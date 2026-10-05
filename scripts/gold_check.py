#!/usr/bin/env python3
"""Grade the gold (or empty) patch for tasks with swegemma's own verify_task, to check the grading environment.

  python gold_check.py --data /workspace/data --tasks requests_7205 rich_3472 [--empty]
"""
import argparse
import asyncio
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--tasks", nargs="+", required=True)
p.add_argument("--empty", action="store_true", help="grade an empty patch instead (should fail)")
p.add_argument("--out", type=Path, default=Path("/workspace/evals/gold"))
args = p.parse_args()

from adk_submission import ModelRegistry
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.evaluate import Evaluator
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks

limits, gen = build_submission_limits()
cfg = EvalConfig(tasks_path=args.data / "tasks.jsonl", snapshots_dir=args.data / "snapshots", results_dir=args.out,
                 submission_dir=Path("/workspace/gemma/variants/v7-config"), models=ModelRegistry(), sandbox="subprocess",
                 wheels_dir=args.data / "wheels", graph_dir=str(args.data / "graphs"), embeddings_dir=str(args.data / "embeddings"),
                 limits=limits, generation_constraints=gen)
ev = Evaluator(cfg)
tasks = {t.instance_id: t for t in load_tasks(args.data / "tasks.jsonl")}
for tid in args.tasks:
    t = tasks[tid]
    snap, base, patch = resolve_task_snapshot_paths(cfg.snapshots_dir, t.instance_id, t.repo)
    r = asyncio.run(verify_task(ev.docker, cfg, t, snap, base_snapshot_path=base, patch_path=patch,
                                agent_patch="" if args.empty else t.patch))
    tail = "\n".join((r.test_output or "").strip().splitlines()[-6:])
    print(json.dumps({"task": tid, "patch": "empty" if args.empty else "gold", "resolved": r.resolved,
                      "exit_code": r.test_exit_code, "error": getattr(r, "error_message", None)}), flush=True)
    print("   " + tail.replace("\n", "\n   "), flush=True)
