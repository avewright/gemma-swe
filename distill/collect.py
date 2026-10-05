#!/usr/bin/env python3
"""Collect teacher traces for many tasks in parallel (rejection sampling, step 1).

Runs distill/rollout.py once per (task, sample) in its own process, like bench.py.
Traces land in traces/raw/<teacher>/<task>__<sample>.json.gz. Rerun the same
command to resume: finished rollouts are skipped. The dev40 holdout is excluded
unless --include-holdout, so the bench stays a fair test of the adapter.

  python distill/collect.py --teacher deepseek --samples 4 --concurrency 8
  python distill/collect.py --teacher openrouter --model z-ai/glm-5.3 --samples 2
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rollout import teacher_label  # noqa: E402
ROOT = Path(os.environ.get("GEMMA_ROOT", HERE.parent))
DATA = Path(os.environ.get("GEMMA_DATA", ROOT / "data"))
HOLDOUT = HERE.parent / "scripts" / "dev40.txt"
lock = threading.Lock()


def load_tasks(args) -> list[dict]:
    rows = [json.loads(l) for l in (DATA / "tasks.jsonl").read_text().splitlines() if l.strip()]
    if not args.include_holdout:
        holdout = set(HOLDOUT.read_text().split())
        rows = [r for r in rows if r["instance_id"] not in holdout]
    if args.tasks:
        rows = [r for r in rows if r["instance_id"] in set(args.tasks)]
    if args.repo:
        rows = [r for r in rows if r["repo"] == args.repo]
    if args.reverse:
        rows = rows[::-1]
    return rows[: args.limit] if args.limit else rows


def run(job: tuple[str, int], args, passthrough: list[str], log_dir: Path) -> dict:
    iid, sample = job
    start = time.time()
    with open(log_dir / f"{iid}__{sample}.log", "w") as log:
        proc = subprocess.Popen(
            [args.python, str(HERE / "rollout.py"), "--task", iid, "--sample", str(sample),
             "--out", str(args.out), "--runs-dir", str(args.runs_dir), *passthrough],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, GEMMA_ROOT=str(ROOT), GEMMA_DATA=str(DATA)),
        )
        try:
            proc.wait(timeout=args.job_timeout * 60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    tail = (log_dir / f"{iid}__{sample}.log").read_text().strip().splitlines()[-1:] or [""]
    status = tail[0].split()[0] if tail[0].startswith(("RESOLVED", "FAILED", "exists")) else f"crash({proc.returncode})"
    # rollout.py cleans up after itself; this covers a rollout killed on timeout.
    teacher = args.label
    subprocess.run(["docker", "rm", "-f", f"gd-{iid}-{teacher}-{sample}".replace("_", "-").lower()], capture_output=True)
    subprocess.run(["rm", "-rf", str(args.runs_dir / f"{iid}__{teacher}__{sample}")])
    return {"task": iid, "sample": sample, "status": status, "seconds": round(time.time() - start)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="deepseek")
    parser.add_argument("--model", help="teacher model id (required for teachers without a default)")
    parser.add_argument("--samples", type=int, default=4, help="rollouts per task")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--repo")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reverse", action="store_true",
                        help="work from the end of the task list, e.g. to cover different tasks than another run")
    parser.add_argument("--include-holdout", action="store_true", help="also use the dev40 bench tasks (not recommended)")
    parser.add_argument("--job-timeout", type=float, default=40, help="minutes per rollout, setup included")
    parser.add_argument("--out", type=Path, default=ROOT / "traces" / "raw")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs" / "distill")
    parser.add_argument("--python", default=sys.executable)
    args, passthrough = parser.parse_known_args()
    passthrough = ["--teacher", args.teacher, *(["--model", args.model] if args.model else []), *passthrough]
    # Same label rollout.py files traces under (openrouter traces are per model).
    args.label = teacher_label(args.teacher, args.model)

    tasks = load_tasks(args)
    teacher_dir = args.out / args.label
    jobs = [(t["instance_id"], s) for t in tasks for s in range(args.samples)
            if not (teacher_dir / f"{t['instance_id']}__{s}.json.gz").exists()]
    log_dir = args.out.parent / "logs" / args.label
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} tasks x {args.samples} samples, {len(jobs)} rollouts to run, concurrency {args.concurrency}", flush=True)

    counts: Counter = Counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(run, j, args, passthrough, log_dir) for j in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with lock:
                counts[result["status"]] += 1
            print(f"[{n}/{len(jobs)}] {result['status']:<9} {result['task']}#{result['sample']} "
                  f"{result['seconds']}s | {dict(counts)}", flush=True)
    print("done. Next: python distill/pack.py", flush=True)


if __name__ == "__main__":
    main()
