#!/usr/bin/env python3
"""Build Gemma SFT traces for the contest tasks.

DeepSeek sees the maintainer patch and writes the tool-call trajectory.
Each step is then run against the task snapshot, so the saved tool results
are real. The saved user message is only the problem statement.

  python scripts/make_traces.py --limit 1
  python scripts/make_traces.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "traces" / "traces.jsonl"
SYSTEM = (ROOT / "submission" / "prompts" / "system.md").read_text()
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"

TEACHER = """You write one training trajectory for a coding agent.

The agent is graded on hidden tests and must match the maintainer change exactly. You can see that change. The trajectory you write must not mention that you were shown it.

Return one JSON object and nothing else:
{"steps":[{"thought":"one or two sentences","tool":"run_command"|"read_file"|"edit_file"|"submit_patch","arguments":{}}]}

Rules:
- 4 to 10 steps. One tool per step.
- Start by locating the code. run_command may only be `grep -n` inside the package.
- read_file and edit_file take filepath, relative to the repo root. Lines are 1-indexed and inclusive.
- edit_file old_string must be copied verbatim from the file excerpts and must occur once. Keep each edit under 40 lines.
- The edits together must reproduce the maintainer patch and change nothing else.
- The last step is submit_patch with arguments {}.
- Do not call pytest, pip, or write_file.
"""


def load_env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEEPSEEK_API")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    return key


def load_tasks(task_id: str | None, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in (DATA / "tasks.jsonl").read_text().splitlines() if line.strip()]
    if task_id:
        rows = [row for row in rows if row["instance_id"] == task_id]
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["instance_id"])
    rows = [row for row in rows if row["instance_id"] not in done]
    if limit:
        rows = rows[:limit]
    return rows


def ask(key: str, user: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": TEACHER},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:500]
        raise RuntimeError(f"deepseek {exc.code}: {detail}") from exc
    text = payload["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def files_in_patch(patch: str) -> list[str]:
    return [line[6:] for line in patch.splitlines() if line.startswith("+++ b/")]


def hunk_spans(patch: str) -> dict[str, list[tuple[int, int]]]:
    spans: dict[str, list[tuple[int, int]]] = {}
    current = None
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            spans.setdefault(current, [])
        match = re.match(r"@@ -(\d+)(?:,(\d+))?", line)
        if match and current:
            start = int(match.group(1))
            count = int(match.group(2) or "1")
            spans[current].append((start, max(count, 1)))
    return spans


def excerpts(repo: Path, patch: str) -> str:
    parts = []
    for path, spans in hunk_spans(patch).items():
        file_path = repo / path
        if not file_path.is_file():
            continue
        lines = file_path.read_text(errors="replace").splitlines()
        wanted: set[int] = set()
        for start, count in spans:
            for number in range(max(1, start - 20), min(len(lines), start + count + 20) + 1):
                wanted.add(number)
        numbered = [f"{n}|{lines[n - 1]}" for n in sorted(wanted)[:220]]
        parts.append(f"### {path}\n" + "\n".join(numbered))
    return "\n\n".join(parts)


def resolve(repo: Path, filepath: str) -> Path:
    rel = filepath.removeprefix("/workspace/").lstrip("/")
    if ".." in Path(rel).parts:
        raise ValueError("bad path")
    path = (repo / rel).resolve()
    if not str(path).startswith(str(repo.resolve())):
        raise ValueError("outside repo")
    return path


def normalize(name: str, args: dict) -> dict:
    args = dict(args or {})
    path = args.get("filepath") or args.get("path") or args.get("file") or args.get("file_path")
    if path:
        args["filepath"] = path
    if "old_string" not in args:
        args["old_string"] = args.get("old") or args.get("old_str") or ""
    if "new_string" not in args:
        args["new_string"] = args.get("new") or args.get("new_str") or ""
    if "command" not in args:
        args["command"] = args.get("cmd") or ""
    if name == "read_file":
        args["start_line"] = args.get("start_line") or args.get("start") or 1
        args["end_line"] = args.get("end_line") or args.get("end")
    return args


def run_tool(repo: Path, name: str, args: dict) -> str:
    args = normalize(name, args)
    if name == "run_command":
        command = args.get("command", "")
        if not re.match(r"^grep\s", command):
            return json.dumps({"status": "error", "error_message": "only grep is allowed in this trace"})
        proc = subprocess.run(["bash", "-lc", command], cwd=repo, text=True, capture_output=True, timeout=30)
        return json.dumps({"status": "ok" if proc.returncode in (0, 1) else "error", "stdout": proc.stdout[:4000], "stderr": proc.stderr[:1000], "exit_code": proc.returncode})
    if name == "read_file":
        path = resolve(repo, args["filepath"])
        lines = path.read_text(errors="replace").splitlines()
        start = int(args.get("start_line") or 1)
        end = int(args.get("end_line") or min(len(lines), start + 149))
        chunk = lines[start - 1 : end]
        return json.dumps({"status": "ok", "content": "\n".join(chunk)[:10000], "start_line": start, "end_line": start + len(chunk) - 1, "total_lines": len(lines)})
    if name == "edit_file":
        path = resolve(repo, args["filepath"])
        text = path.read_text()
        old, new = args["old_string"], args["new_string"]
        count = text.count(old)
        if count != 1:
            return json.dumps({"status": "error", "error_message": f"old_string matched {count} times"})
        path.write_text(text.replace(old, new, 1))
        return json.dumps({"status": "ok", "filepath": args["filepath"]})
    if name == "submit_patch":
        subprocess.run(["git", "add", "-N", "."], cwd=repo, check=False)
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=repo, text=True, capture_output=True)
        (repo.parent / "agent.patch").write_text(diff.stdout)
        files = [line[6:] for line in diff.stdout.splitlines() if line.startswith("+++ b/")]
        return json.dumps({"status": "ok", "patch_size": len(diff.stdout), "files_changed": len(files)})
    return json.dumps({"status": "error", "error_message": f"unknown tool {name}"})


def changed_lines(patch: str, prefix: str) -> list[str]:
    return [line[1:] for line in patch.splitlines() if line.startswith(prefix) and not line.startswith(prefix * 3)]


def coverage(gold: str, produced: str) -> float:
    score = []
    for prefix in ("+", "-"):
        wanted = changed_lines(gold, prefix)
        got = set(changed_lines(produced, prefix))
        if wanted:
            score.append(sum(line in got for line in wanted) / len(wanted))
    return min(score) if score else 0.0


def teacher_prompt(task: dict, repo: Path) -> str:
    patch = task["patch"]
    if len(patch) > 40000:
        patch = patch[:40000] + "\n...[patch truncated]"
    return (
        f"Problem statement:\n{task['problem_statement']}\n\n"
        f"File excerpts before the change:\n{excerpts(repo, task['patch'])}\n\n"
        f"Maintainer patch:\n{patch}"
    )


def build(task: dict, repo: Path, key: str) -> dict | None:
    steps = ask(key, teacher_prompt(task, repo)).get("steps") or []
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task["problem_statement"]},
    ]
    submitted = False
    for index, step in enumerate(steps):
        name = step.get("tool")
        args = step.get("arguments") or {}
        if name in {"read_file", "edit_file"}:
            filepath = args.get("filepath") or args.get("path") or args.get("file") or args.get("file_path")
            args = {key: value for key, value in args.items() if key not in {"path", "file", "file_path"}}
            if filepath:
                args["filepath"] = filepath
        try:
            result = run_tool(repo, name, args)
        except Exception as exc:
            result = json.dumps({"status": "error", "error_message": str(exc)})
        messages.append({
            "role": "assistant",
            "content": step.get("thought") or "",
            "tool_calls": [{
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }],
        })
        messages.append({"role": "tool", "tool_call_id": f"call_{index}", "name": name, "content": result})
        if name == "submit_patch":
            submitted = True
            break
        if '"status": "error"' in result and name == "edit_file":
            return None
    if not submitted:
        return None
    produced = (repo.parent / "agent.patch").read_text()
    score = coverage(task["patch"], produced)
    if score < 0.8:
        return None
    return {"instance_id": task["instance_id"], "repo": task["repo"], "coverage": round(score, 3), "messages": messages}


def extract(task: dict, dest: Path) -> None:
    archive = DATA / "snapshots" / f"{task['instance_id']}.tgz"
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    subprocess.run(["git", "config", "user.email", "agent@eval"], cwd=dest, check=False)
    subprocess.run(["git", "config", "user.name", "Agent"], cwd=dest, check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task")
    args = parser.parse_args()
    load_env()
    key = api_key()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    for task in load_tasks(args.task, args.limit):
        print(task["instance_id"], flush=True)
        work = Path(tempfile.mkdtemp(prefix="trace-"))
        repo = work / "repo"
        repo.mkdir()
        row = None
        try:
            extract(task, repo)
            for _ in range(2):
                try:
                    row = build(task, repo, key)
                    break
                except (json.JSONDecodeError, RuntimeError) as exc:
                    print(f"retry {task['instance_id']}: {exc}", flush=True)
                    row = None
        except Exception as exc:
            print(f"reject {task['instance_id']}: {exc}", flush=True)
            row = None
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if row is None:
            print(f"reject {task['instance_id']}", flush=True)
            continue
        with OUT.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        kept += 1
        print(f"kept {task['instance_id']} coverage {row['coverage']}", flush=True)
    print(f"wrote {kept} traces to {OUT}", flush=True)


if __name__ == "__main__":
    main()
