#!/usr/bin/env python3
"""Run one contest task against a local vLLM Gemma server.

This is not the official swegemma harness. It uses the same snapshot,
wheels, system prompt, and pytest check so a pod can score a task.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import os
import tarfile
import time
from pathlib import Path

import yaml
from openai import BadRequestError, OpenAI

ROOT = Path("/workspace/gemma")
DATA = ROOT / "data"
RUNS = ROOT / "runs"
MODEL = "google/gemma-4-31b-it-qat-w4a16-ct"
# What the contest sandbox image (data/docker/Dockerfile.sandbox) preinstalls.
BASE_PACKAGES = ["pytest", "pytest-timeout==2.1.0", "typer", "pdm-backend", "setuptools", "wheel",
                 "poetry-core", "hatchling", "flit-core", "editables"]
CONTEXT = 32768
MAX_NUDGES = 3
# Budgets and sampling come from the submission itself, like the contest harness.
EVAL = yaml.safe_load((ROOT / "submission" / "eval_config.yaml").read_text())["evaluation"]
SAMPLING = yaml.safe_load((ROOT / "submission" / "configs" / "sampling.yaml").read_text())
MAX_TOOLS = EVAL["max_tool_calls"]
MAX_TURNS = EVAL["max_turns"]
AGENT_MINUTES = EVAL["max_time_minutes"]
MAX_OUTPUT = SAMPLING.get("max_output_tokens", 16384)
THINKING = SAMPLING.get("thinking_config") or {}
ENABLE_THINKING = bool(THINKING.get("include_thoughts", True)) and str(THINKING.get("thinking_level", "")).upper() != "NONE"
NUDGE = ("Continue working on the task. Use your tools to make progress, and call submit_patch "
         "when your change is complete.")
NUDGE_LENGTH = ("Your previous response reached the token limit while thinking before a tool call was completed. "
                "Do NOT repeat your analysis in thought—keep reasoning under a few sentences and emit your next "
                "tool call immediately, or call submit_patch.")


def task_message(task: dict, repo: Path) -> str:
    """Mirror the contest harness's task prompt (HARNESS_README section 5)."""
    layout = subprocess.run(
        "find . -maxdepth 3 -not -path './.git*' -not -name '__pycache__' -not -name '*.pyc' | head -150",
        shell=True, cwd=repo, text=True, capture_output=True).stdout
    parts = [f"You are evaluating a software engineering task for repository {task['repo']}.\n\n"
             f"Problem Statement:\n{task['problem_statement']}"]
    if task.get("hints_text"):
        parts.append(f"## Hints:\n{task['hints_text']}")
    parts.append("## Task Budget (Session terminates when any budget is exhausted)\n"
                 f"- Time allowance: {float(AGENT_MINUTES)} minutes\n"
                 f"- Tool calls allowance: {MAX_TOOLS} calls\n"
                 f"- Max loop iterations: {MAX_TURNS} turns")
    parts.append("## Execution Environment Rules\n"
                 f"- Single command timeout: {EVAL.get('timeout_seconds', 300)} seconds (commands exceeding this fail without ending the session)\n"
                 "- Command output limit: 5000 characters\n"
                 "- File view limit: 150 lines per read_file call\n"
                 "- File character limit: 10000 characters per read_file call\n"
                 "- Environment is offline (no network/PyPI access). All repository and test dependencies are ALREADY "
                 "pre-installed. Do NOT attempt to run pip install or download packages.\n"
                 "- The shell starts in the repository root. Paths described as /workspace are this directory.")
    parts.append(f"## Workspace Layout\n{layout}")
    return "\n\n".join(parts)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a bash command in the repository root.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file. Lines are 1-indexed and inclusive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace one exact old_string with new_string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["filepath", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_patch",
            "description": "Submit the current git diff. Call this last.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Return remaining tool-call budget.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def load_task(instance_id: str) -> dict:
    for line in (DATA / "tasks.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["instance_id"] == instance_id:
            return row
    raise SystemExit(f"unknown task {instance_id}")


def resolve(repo: Path, filepath: str) -> Path:
    rel = filepath.removeprefix("/workspace/").lstrip("/")
    if ".." in Path(rel).parts:
        raise ValueError("path traversal")
    path = (repo / rel).resolve()
    if not str(path).startswith(str(repo.resolve())):
        raise ValueError("outside repo")
    return path


def clip(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def task_env(repo: Path) -> dict:
    # Each task has its own venv so parallel tasks of one repo don't share installs.
    venv = repo.parent / "venv"
    env = dict(os.environ, VIRTUAL_ENV=str(venv), PATH=f"{venv / 'bin'}:{os.environ['PATH']}")
    env.pop("PYTHONPATH", None)
    return env


def run_cmd(repo: Path, command: str) -> str:
    proc = subprocess.run(
        ["bash", "-lc", command],
        cwd=repo,
        env=task_env(repo),
        text=True,
        capture_output=True,
        timeout=120,
    )
    payload = {
        "status": "ok" if proc.returncode == 0 else "error",
        "exit_code": proc.returncode,
        "stdout": clip(proc.stdout),
        "stderr": clip(proc.stderr),
    }
    return json.dumps(payload)


def tool_call(repo: Path, name: str, args: dict, used: int) -> tuple[str, bool]:
    if name == "get_status":
        return json.dumps({"tool_calls_used": used, "tool_calls_remaining": MAX_TOOLS - used, "max_tool_calls": MAX_TOOLS}), False
    if name == "submit_patch":
        diff = subprocess.run(["git", "add", "-N", "."], cwd=repo)
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=repo, text=True, capture_output=True)
        patch = diff.stdout
        (repo.parent / "agent.patch").write_text(patch)
        files = [ln[6:] for ln in patch.splitlines() if ln.startswith("+++ b/")]
        return json.dumps({"status": "ok", "patch_size": len(patch), "files_changed": len(files)}), True
    if name == "run_command":
        try:
            return run_cmd(repo, args["command"]), False
        except subprocess.TimeoutExpired:
            return json.dumps({"status": "error", "error_type": "TimeoutExceeded"}), False
    if name == "read_file":
        path = resolve(repo, args["filepath"])
        lines = path.read_text(errors="replace").splitlines()
        start = int(args.get("start_line") or 1)
        end = int(args.get("end_line") or min(len(lines), start + 149))
        chunk = lines[start - 1 : end]
        body = "\n".join(chunk)
        if len(body) > 10000:
            body = body[:10000]
        return json.dumps({
            "status": "ok",
            "filepath": args["filepath"],
            "content": body,
            "start_line": start,
            "end_line": start + len(chunk) - 1,
            "total_lines": len(lines),
        }), False
    if name == "edit_file":
        path = resolve(repo, args["filepath"])
        text = path.read_text()
        updated, how = replace_once(text, args["old_string"], args["new_string"])
        if updated is None:
            return json.dumps({"status": "error", "error_message": how}), False
        path.write_text(updated)
        return json.dumps({"status": "ok", "filepath": args["filepath"], "match": how}), False
    if name == "write_file":
        path = resolve(repo, args["filepath"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"])
        return json.dumps({"status": "ok", "filepath": args["filepath"]}), False
    return json.dumps({"status": "error", "error_message": f"unknown tool {name}"}), False


def unescape(s: str) -> str:
    return s.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')


def replace_once(text: str, old: str, new: str) -> tuple[str | None, str]:
    """Exact match, then with escaped newlines decoded, then ignoring per-line indentation
    and trailing whitespace. Stands in for the harness's 3-tier apply_replacement."""
    tiers = [("exact", old, new)]
    if "\\n" in old:
        tiers.append(("unescaped", unescape(old), unescape(new)))
    for how, o, n in tiers:
        count = text.count(o)
        if count == 1:
            return text.replace(o, n, 1), how
        if count > 1:
            return None, f"old_string matched {count} times; include more surrounding lines"
    o_lines = [l.strip() for l in tiers[-1][1].strip("\n").splitlines()]
    n_text = tiers[-1][2]
    lines = text.splitlines(keepends=True)
    hits = [i for i in range(len(lines) - len(o_lines) + 1)
            if [l.strip() for l in lines[i : i + len(o_lines)]] == o_lines]
    if len(hits) == 1 and o_lines:
        i = hits[0]
        if not n_text.endswith("\n") and lines[i + len(o_lines) - 1].endswith("\n"):
            n_text += "\n"
        return "".join(lines[:i]) + n_text + "".join(lines[i + len(o_lines) :]), "whitespace"
    if len(hits) > 1:
        return None, f"old_string matched {len(hits)} times; include more surrounding lines"
    return None, "old_string matched 0 times; re-read the file and copy the lines exactly"


def safe_tool_call(repo: Path, name: str, args: dict, used: int) -> tuple[str, bool]:
    try:
        return tool_call(repo, name, args, used)
    except Exception as exc:
        return json.dumps({"status": "error", "error_type": type(exc).__name__, "error_message": str(exc)}), False


def current_patch(repo: Path) -> str:
    subprocess.run(["git", "add", "-N", "."], cwd=repo)
    return subprocess.run(["git", "diff", "HEAD"], cwd=repo, text=True, capture_output=True).stdout


def prepare(task: dict) -> Path:
    repo = RUNS / task["instance_id"] / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir(parents=True)
    with tarfile.open(DATA / "snapshots" / f"{task['instance_id']}.tgz", "r:gz") as tar:
        tar.extractall(repo, filter="data")
    subprocess.run(["git", "config", "user.email", "agent@eval"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Agent"], cwd=repo, check=True)
    wheels = Path("/wheels")
    if not wheels.exists():
        wheels.symlink_to(DATA / "wheels")
    venv = repo.parent / "venv"
    py = str(venv / "bin" / "python")
    env = task_env(repo)
    subprocess.run(["python3.13", "-m", "venv", str(venv)], check=True)
    subprocess.run([py, "-m", "pip", "install", "-q", *BASE_PACKAGES], env=env, check=True)
    site = next((venv / "lib").glob("python3*/site-packages"))
    for shim in ("imp.py", "telnetlib.py"):
        shutil.copy(DATA / "docker" / shim, site / shim)
    sandbox = DATA / "sandbox"
    subprocess.run(
        [py, str(sandbox / "setup.py"), "--workspace", str(repo), task["repo"].split("/")[-1]],
        cwd=repo,
        env=env,
        check=False,
    )
    # setup.py alone leaves many tasks unable to import their test deps; the
    # contest image has them prebaked. See install_deps.py.
    subprocess.run([py, str(Path(__file__).resolve().parent / "install_deps.py"), str(repo)], env=env, check=False)
    # Like the harness: commit the prepared tree (pytest.ini, conftest.py) as the baseline.
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "--no-verify", "-m", "eval_baseline"], cwd=repo, check=False)
    return repo


def apply_patch(repo: Path, text: str, name: str) -> bool:
    patch_file = repo.parent / name
    patch_file.write_text(text if text.endswith("\n") else text + "\n")
    for extra in ([], ["-3"], ["--ignore-whitespace"], ["--recount"]):
        res = subprocess.run(["git", "apply", *extra, str(patch_file)], cwd=repo, text=True, capture_output=True)
        if res.returncode == 0:
            return True
    res = subprocess.run(["patch", "-p1", "--batch", "--forward", "-i", str(patch_file)], cwd=repo, text=True, capture_output=True)
    if res.returncode != 0:
        print(f"{name} failed to apply")
        print(res.stdout[-1000:], res.stderr[-1000:])
    return res.returncode == 0


def grade(task: dict, repo: Path) -> int:
    targets = [ln[6:] for ln in task["test_patch"].splitlines() if ln.startswith("+++ b/")]
    # The harness discards agent edits to the target test files before applying test_patch.
    subprocess.run(["git", "checkout", "HEAD", "--", *targets], cwd=repo, capture_output=True)
    subprocess.run(["git", "clean", "-f", "--", *targets], cwd=repo, capture_output=True)
    if not apply_patch(repo, task["test_patch"], "test.patch"):
        return 1
    cmd = ["python", "-m", "pytest", *targets, "-p", "no:anyio", "-o", "timeout=0",
           "-o", "norecursedirs=.* build dist venv", "-o", "python_classes=Test* *Test", "-q"]
    env = dict(task_env(repo), PYTHONSAFEPATH="1")
    try:
        proc = subprocess.run(cmd, cwd=repo, env=env, text=True, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        print("pytest timed out")
        return 1
    (repo.parent / "pytest.log").write_text(proc.stdout + "\n" + proc.stderr)
    print(proc.stdout[-2000:])
    print(proc.stderr[-1000:])
    return proc.returncode


def main() -> None:
    global RUNS
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="rich_4077")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    parser.add_argument("--mode", choices=["agent", "gold", "empty"], default="agent",
                        help="gold applies the reference fix, empty grades the untouched repo")
    args = parser.parse_args()
    RUNS = args.runs_dir
    task = load_task(args.task)
    start = time.time()
    repo = prepare(task)
    setup_seconds = time.time() - start
    if args.mode != "agent":
        applied = args.mode == "empty" or apply_patch(repo, task["patch"], "gold.patch")
        code = grade(task, repo) if applied else 1
        result = {"instance_id": args.task, "repo": task["repo"], "resolved": code == 0, "stop": args.mode,
                  "setup_seconds": round(setup_seconds, 1), "total_seconds": round(time.time() - start, 1)}
        (repo.parent / "result.json").write_text(json.dumps(result, indent=1))
        print("RESOLVED" if code == 0 else "FAILED", args.task)
        raise SystemExit(code)
    system = (ROOT / "submission" / "prompts" / "system.md").read_text()
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": task_message(task, repo),
        },
    ]
    submitted = False
    used = 0
    context_used = 0
    prompt_tokens = completion_tokens = 0
    stop = "max_turns"
    turn = -1
    nudges = 0
    for turn in range(MAX_TURNS):
        if used >= MAX_TOOLS:
            stop = "max_tools"
            break
        if time.time() - start - setup_seconds > AGENT_MINUTES * 60:
            stop = "time"
            break
        # Rough prompt size: last reported total plus new tool output at ~3 chars/token.
        new_chars = sum(len(str(m.get("content") or "")) for m in messages[last_len:]) if turn else 0
        room = CONTEXT - context_used - new_chars // 3 - 512
        if room < 1024:
            print(f"turn {turn}: context full ({context_used} tokens used)")
            stop = "context_full"
            break
        last_len = len(messages)
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=min(MAX_OUTPUT, room),
                temperature=SAMPLING.get("temperature", 0.2),
                top_p=SAMPLING.get("top_p", 1.0),
                extra_body={"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}},
            )
        except BadRequestError as exc:
            print(f"turn {turn}: request rejected: {exc}")
            stop = "request_rejected"
            break
        context_used = resp.usage.total_tokens
        prompt_tokens += resp.usage.prompt_tokens
        completion_tokens += resp.usage.completion_tokens
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        calls = msg.tool_calls or []
        if not calls:
            print(f"turn {turn}: no tool call")
            print((msg.content or "")[:500])
            # Like the harness: nudge up to MAX_NUDGES consecutive turns without a tool call.
            nudges += 1
            if nudges > MAX_NUDGES:
                stop = "no_tool_call"
                break
            length = resp.choices[0].finish_reason == "length"
            messages.append({"role": "user", "content": NUDGE_LENGTH if length else NUDGE})
            continue
        nudges = 0
        for call in calls:
            name = call.function.name
            raw = call.function.arguments or "{}"
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
            print(f"turn {turn}: {name}")
            result, done = safe_tool_call(repo, name, payload, used)
            if name not in {"get_status", "submit_patch"}:
                used += 1
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            if done:
                submitted = True
        if submitted:
            stop = "submitted"
            break
    agent_seconds = time.time() - start - setup_seconds
    # Like the contest harness: grade whatever diff is left even without submit_patch.
    patch = current_patch(repo)
    (repo.parent / "agent.patch").write_text(patch)
    code = grade(task, repo) if patch.strip() else 1
    (repo.parent / "messages.json").write_text(json.dumps(messages, indent=1, default=str))
    result = {
        "instance_id": args.task,
        "repo": task["repo"],
        "resolved": code == 0,
        "submitted": submitted,
        "stop": stop,
        "turns": turn + 1,
        "tool_calls": used,
        "patch_chars": len(patch),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "setup_seconds": round(setup_seconds, 1),
        "agent_seconds": round(agent_seconds, 1),
        "total_seconds": round(time.time() - start, 1),
    }
    (repo.parent / "result.json").write_text(json.dumps(result, indent=1))
    print("RESOLVED" if code == 0 else "FAILED", args.task)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
