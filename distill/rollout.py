#!/usr/bin/env python3
"""Run one teacher rollout on one contest task and save it as a trace.

The teacher works like the student will: same system prompt, task message,
tools, tool-call and turn budgets, and the same grading. It never sees the
maintainer patch, and its commands run in an offline sandbox container that holds
only the task repo (sandbox.py). Every trace is saved, pass or fail;
train_lora.py keeps the resolved ones (rejection sampling).

  python distill/rollout.py --task rich_4077 --teacher deepseek --sample 0 --out traces/raw

collect.py runs this for many tasks in parallel.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from openai import APIConnectionError, APIStatusError, BadRequestError, OpenAI, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_one_task as rot  # noqa: E402  (budgets, tools and prompts, as the bench uses them)

from sandbox import Sandbox  # noqa: E402

# OpenAI-compatible endpoints. Model names change often, so only DeepSeek has a default
# (the one the earlier trace script used); pass --model for the others.
TEACHERS = {
    "deepseek": {"base_url": "https://api.deepseek.com", "key_env": "DEEPSEEK_API_KEY", "model": "deepseek-flash"},
    "kimi": {"base_url": "https://api.moonshot.ai/v1", "key_env": "MOONSHOT_API_KEY", "model": None},
    "glm": {"base_url": "https://api.z.ai/api/paas/v4", "key_env": "ZAI_API_KEY", "model": None},
    "qwen": {"base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "key_env": "DASHSCOPE_API_KEY", "model": None},
    # Any model OpenRouter serves, e.g. --model z-ai/glm-5.3 or moonshotai/kimi-k3.
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER", "model": None},
    # Any other OpenAI-compatible server, e.g. a self-hosted vLLM: --base-url and --key-env.
    "custom": {"base_url": None, "key_env": "TEACHER_API_KEY", "model": None},
}
SENT_KEYS = {"role", "content", "tool_calls", "tool_call_id"}
# Commands that look for the answer outside the repo (a newer release, another rollout's repo,
# caches, git objects past base_commit). The contest sandbox has none of these, so a trace that
# relies on them teaches the student a habit that can't work; train_lora.py drops such traces.
OUTSIDE = re.compile(r"pip3? (download|install)|uv (pip|add|sync)|site-packages|/root\b|~/|\.cache|"
                     r"find /( |$)|ls /( |$)|/usr/(local/)?lib|/opt/|curl |wget |git fsck|lost-found|git log --all")


def load_env() -> None:
    for env in (rot.ROOT / ".env", rot.ROOT.parent / ".env", Path.cwd() / ".env"):
        if not env.is_file():
            continue
        for line in env.read_text().splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def teacher_label(teacher: str, model: str | None) -> str:
    """Folder and id name for a teacher: the preset, plus the model when it isn't the preset's default."""
    slug = re.sub(r"[^a-z0-9.]+", "-", (model or "").lower()).strip("-")
    if teacher == "openrouter" and model:
        return "or-" + slug
    if model and model != TEACHERS[teacher]["model"]:
        return f"{teacher}-{slug}"
    return teacher


def teacher_config(args) -> dict:
    preset = TEACHERS[args.teacher]
    cfg = {
        "name": teacher_label(args.teacher, args.model),
        "base_url": args.base_url or preset["base_url"],
        "key_env": args.key_env or preset["key_env"],
        "model": args.model or preset["model"],
    }
    if not cfg["base_url"] or not cfg["model"]:
        raise SystemExit(f"teacher {args.teacher} needs --model{' and --base-url' if not cfg['base_url'] else ''}")
    if not os.environ.get(cfg["key_env"]):
        raise SystemExit(f"{cfg['key_env']} is not set (put it in .env)")
    return cfg


def for_api(message: dict, echo_reasoning: bool) -> dict:
    """What the teacher is sent back: OpenAI fields only. Some thinking APIs want their
    reasoning returned within a tool-use turn; --echo-reasoning does that."""
    sent = {k: v for k, v in message.items() if k in SENT_KEYS}
    if echo_reasoning and message.get("reasoning"):
        sent["reasoning_content"] = message["reasoning"]
    return sent


def complete(client: OpenAI, cfg: dict, messages: list[dict], tools: list[dict], args) -> object:
    extra = json.loads(args.extra_body) if args.extra_body else None
    for attempt in range(8):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=[for_api(m, args.echo_reasoning) for m in messages],
                tools=tools,
                tool_choice="auto",
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_body=extra,
            )
            if resp.choices:
                return resp
            # OpenRouter sometimes returns a 200 with an upstream error and no choices.
            wait = min(300, 10 * 2 ** attempt)
            print(f"teacher returned no choices ({str((resp.model_extra or {}).get('error'))[:120]}), retry in {wait}s", flush=True)
            time.sleep(wait)
        except BadRequestError:
            raise
        except APIStatusError as exc:
            # Out of credit, over a spending limit, or a bad key: retrying won't help.
            if exc.status_code in (401, 402, 403):
                raise RuntimeError(f"teacher refused ({exc.status_code}): {str(exc)[:200]}") from exc
            wait = min(300, 10 * 2 ** attempt)
            print(f"teacher error ({type(exc).__name__}), retry in {wait}s", flush=True)
            time.sleep(wait)
        except (RateLimitError, APIConnectionError) as exc:
            wait = min(300, 10 * 2 ** attempt)
            print(f"teacher error ({type(exc).__name__}), retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("teacher unavailable")


def rollout(task: dict, cfg: dict, args) -> dict:
    iid = task["instance_id"]
    workdir = args.runs_dir / f"{iid}__{cfg['name']}__{args.sample}"
    sandbox = Sandbox(iid, f"gd-{iid}-{cfg['name']}-{args.sample}".replace("_", "-").lower(),
                      workdir, rot.ROOT, rot.DATA)
    try:
        return run_agent(task, cfg, args, sandbox)
    finally:
        sandbox.close()
        shutil.rmtree(workdir, ignore_errors=True)


def run_agent(task: dict, cfg: dict, args, sandbox: Sandbox) -> dict:
    iid = task["instance_id"]
    if rot.CODE_TOOLS_ON:
        rot.CODE_INDEX = rot.load_code_index(iid)
    start = time.time()
    sandbox.setup()
    repo = sandbox.repo
    setup_seconds = time.time() - start
    system = (rot.SUBMISSION / "prompts" / "system.md").read_text()
    tools = rot.active_tools()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": rot.task_message(task, repo)}]
    client = OpenAI(base_url=cfg["base_url"], api_key=os.environ[cfg["key_env"]], timeout=600, max_retries=0)

    used = nudges = bad_calls = multi_call_turns = outside_refs = 0
    prompt_tokens = completion_tokens = 0
    cost_usd = 0.0
    submitted = False
    stop = "max_turns"
    turn = -1
    for turn in range(rot.MAX_TURNS):
        if used >= rot.MAX_TOOLS:
            stop = "max_tools"
            break
        try:
            resp = complete(client, cfg, messages, tools, args)
        except BadRequestError as exc:
            print(f"turn {turn}: request rejected: {exc}", flush=True)
            stop = "request_rejected"
            break
        if resp.usage:
            prompt_tokens += resp.usage.prompt_tokens
            completion_tokens += resp.usage.completion_tokens
            # OpenRouter reports the charged cost (after caching) with each response.
            cost_usd += float((resp.usage.model_extra or {}).get("cost") or 0)
        msg = resp.choices[0].message
        extra = msg.model_extra or {}
        record = {"role": "assistant", "content": msg.content or ""}
        reasoning = extra.get("reasoning_content") or extra.get("reasoning")
        if reasoning:
            record["reasoning"] = reasoning
        calls = msg.tool_calls or []
        if calls:
            record["tool_calls"] = [{"id": c.id, "type": "function",
                                     "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                                    for c in calls]
        messages.append(record)
        if not calls:
            # Same nudges as the harness, so traces show how to recover from a stalled turn.
            nudges += 1
            if nudges > rot.MAX_NUDGES:
                stop = "no_tool_call"
                break
            length = resp.choices[0].finish_reason == "length"
            messages.append({"role": "user", "content": rot.NUDGE_LENGTH if length else rot.NUDGE})
            continue
        nudges = 0
        multi_call_turns += len(calls) > 1
        for call in calls:
            name = call.function.name
            try:
                payload = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                payload = {}
                bad_calls += 1
            print(f"turn {turn}: {name}", flush=True)
            if name == "run_command" and OUTSIDE.search(str(payload.get("command", ""))):
                outside_refs += 1
            if name == "run_command" and not rot.missing_args(name, payload):
                result, done = sandbox.run(str(payload["command"])), False
            else:
                # File tools act on the host copy of /workspace; the container sees the same files.
                result, done = rot.safe_tool_call(repo, name, payload, used)
            if name not in {"get_status", "submit_patch"} and not rot.missing_args(name, payload):
                used += 1
            messages.append({"role": "tool", "tool_call_id": call.id, "name": name, "content": result})
            submitted = submitted or done
        if submitted:
            stop = "submitted"
            break

    agent_seconds = time.time() - start - setup_seconds
    patch = rot.current_patch(repo)
    resolved, grade_log = sandbox.grade(patch) if patch.strip() else (False, "empty patch")
    return {
        "id": f"{iid}/{cfg['name']}/{args.sample}",
        "instance_id": iid,
        "repo": task["repo"],
        "teacher": cfg["name"],
        "teacher_model": cfg["model"],
        "sample": args.sample,
        "temperature": args.temperature,
        "resolved": resolved,
        "submitted": submitted,
        "stop": stop,
        "turns": turn + 1,
        "tool_calls": used,
        "bad_calls": bad_calls,
        "multi_call_turns": multi_call_turns,
        "outside_refs": outside_refs,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost_usd, 4),
        "setup_seconds": round(setup_seconds, 1),
        "agent_seconds": round(agent_seconds, 1),
        "system_sha": hashlib.sha256(system.encode()).hexdigest()[:12],
        "patch": patch,
        "grade_log": grade_log,
        "tools": tools,
        "messages": messages,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def out_path(out: Path, teacher: str, iid: str, sample: int) -> Path:
    return out / teacher / f"{iid}__{sample}.json.gz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--sample", type=int, default=0, help="sample index; several per task give more passing traces")
    parser.add_argument("--teacher", choices=sorted(TEACHERS), default="deepseek")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--key-env")
    parser.add_argument("--temperature", type=float, default=0.6, help="some diversity across samples")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--extra-body", help='JSON passed to the API as is, e.g. \'{"thinking": {"type": "disabled"}}\'')
    parser.add_argument("--echo-reasoning", action="store_true", help="send reasoning back as reasoning_content")
    parser.add_argument("--out", type=Path, default=rot.ROOT / "traces" / "raw")
    parser.add_argument("--runs-dir", type=Path, default=rot.ROOT / "runs" / "distill",
                        help="host scratch for repos and venvs (removed after each rollout)")
    args = parser.parse_args()
    load_env()
    cfg = teacher_config(args)
    dest = out_path(args.out, cfg["name"], args.task, args.sample)
    if dest.exists():
        print(f"exists {dest}")
        return
    row = rollout(rot.load_task(args.task), cfg, args)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(row, fh)
    tmp.rename(dest)
    print(("RESOLVED" if row["resolved"] else "FAILED"), row["id"], f"stop={row['stop']} tools={row['tool_calls']}")


if __name__ == "__main__":
    main()
