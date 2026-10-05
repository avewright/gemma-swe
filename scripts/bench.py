#!/usr/bin/env python3
"""Run many contest tasks in parallel against the local vLLM server.

Each task runs scripts/run_one_task.py in its own process. The task repos and
venvs go on local disk (fast). Results, patches, transcripts and logs are
copied to /workspace/gemma/runs/bench/<name>/, which survives pod restarts.
Rerun with the same --name to resume; finished tasks are skipped.

  python scripts/bench.py --name baseline --concurrency 4
  python scripts/bench.py --name smoke --limit 5
  python scripts/bench.py --name rich --repo Textualize/rich
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/workspace/gemma")
SCRIPT = Path(__file__).resolve().parent / "run_one_task.py"
PYTHON = "/opt/vllm-venv/bin/python"  # has the openai package
SCRATCH = Path("/root/bench")

lock = threading.Lock()


def load_tasks(args) -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "data" / "tasks.jsonl").read_text().splitlines() if l.strip()]
    if args.tasks:
        wanted = set(args.tasks)
        rows = [r for r in rows if r["instance_id"] in wanted]
    if args.repo:
        rows = [r for r in rows if r["repo"] == args.repo]
    rows = rows[args.shard_index :: args.num_shards]
    if args.limit:
        rows = rows[: args.limit]
    return rows


def wait_for_server(base_url: str, timeout: int = 1800) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as resp:
                if resp.status == 200:
                    return
        except OSError:
            pass
        if time.time() > deadline:
            sys.exit(f"vLLM not ready at {base_url}")
        print("waiting for vLLM...", flush=True)
        time.sleep(15)


def run_task(task: dict, args, out: Path, scratch: Path) -> dict:
    iid = task["instance_id"]
    log_path = out / "logs" / f"{iid}.log"
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [PYTHON, str(SCRIPT), "--task", iid, "--base-url", args.base_url, "--runs-dir", str(scratch), "--mode", args.mode],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=args.task_timeout * 60)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            timed_out = True

    task_dir = scratch / iid
    result_file = task_dir / "result.json"
    if result_file.exists():
        result = json.loads(result_file.read_text())
    else:
        # Crashed or timed out before writing a result.
        result = {"instance_id": iid, "repo": task["repo"], "resolved": False,
                  "stop": "timeout" if timed_out else f"crash(exit {proc.returncode})"}
    result["wall_seconds"] = round(time.time() - start, 1)

    for name, dest in [("agent.patch", out / "patches" / f"{iid}.patch"),
                       ("messages.json", out / "transcripts" / f"{iid}.json"),
                       ("pytest.log", out / "logs" / f"{iid}.pytest.log")]:
        if (task_dir / name).exists():
            shutil.copy(task_dir / name, dest)
    if not args.keep:
        shutil.rmtree(task_dir, ignore_errors=True)

    with lock:
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
    return result


def summarize(out: Path, total: int) -> dict:
    rows = {}
    for line in (out / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["instance_id"]] = r
    done = list(rows.values())
    resolved = [r for r in done if r.get("resolved")]
    by_repo: dict[str, list[int]] = {}
    for r in done:
        s = by_repo.setdefault(r["repo"], [0, 0])
        s[0] += bool(r.get("resolved"))
        s[1] += 1
    summary = {
        "finished": len(done),
        "total": total,
        "resolved": len(resolved),
        "resolution_rate": round(len(resolved) / len(done), 4) if done else 0.0,
        "by_repo": {k: f"{a}/{b}" for k, (a, b) in sorted(by_repo.items())},
        "stop_reasons": dict(Counter(r.get("stop") for r in done)),
        "mean_agent_seconds": round(sum(r.get("agent_seconds", 0) for r in done) / max(len(done), 1), 1),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in done),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="run name; reuse it to resume")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--tasks", nargs="*", help="specific instance_ids")
    parser.add_argument("--repo", help="only tasks from this repo, e.g. Textualize/rich")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--task-timeout", type=float, default=20, help="minutes per task, setup included")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--mode", choices=["agent", "gold", "empty"], default="agent",
                        help="gold/empty check the benchmark itself: expect ~100%% and ~0%%")
    parser.add_argument("--keep", action="store_true", help="keep repos and venvs in /root/bench")
    args = parser.parse_args()

    out = ROOT / "runs" / "bench" / args.name
    for sub in ("logs", "patches", "transcripts"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").touch()
    scratch = SCRATCH / args.name
    scratch.mkdir(parents=True, exist_ok=True)
    # Keep a copy of the prompt and config this run used.
    shutil.copytree(ROOT / "submission", out / "submission", dirs_exist_ok=True)

    tasks = load_tasks(args)
    done = {json.loads(l)["instance_id"] for l in (out / "results.jsonl").read_text().splitlines()}
    todo = [t for t in tasks if t["instance_id"] not in done]
    print(f"{len(tasks)} tasks, {len(done & {t['instance_id'] for t in tasks})} already done, "
          f"{len(todo)} to run, concurrency {args.concurrency}", flush=True)
    print(f"results: {out}", flush=True)

    wait_for_server(args.base_url)
    finished = 0
    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(run_task, t, args, out, scratch) for t in todo]
        for fut in futures:
            r = fut.result()
            finished += 1
            s = summarize(out, len(tasks))
            mark = "PASS" if r.get("resolved") else "fail"
            print(f"[{finished}/{len(todo)}] {mark} {r['instance_id']:<28} stop={r.get('stop')} "
                  f"{r.get('wall_seconds')}s | {s['resolved']}/{s['finished']} = {s['resolution_rate']:.1%}",
                  flush=True)

    print(json.dumps(summarize(out, len(tasks)), indent=1))


if __name__ == "__main__":
    main()
