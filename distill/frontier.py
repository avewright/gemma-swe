#!/usr/bin/env python3
"""Compare teachers on the same tasks and mark the cost/quality Pareto frontier.

Quality is the share of rollouts that give a usable training trace: one that passes
quality.issues(), the same gate train_lora.py applies.
Cost is what OpenRouter charged per rollout. A teacher is on the frontier if no other
teacher is at least as good on both and better on one.

  python distill/frontier.py --tasks-file distill/pilot_tasks.txt
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import quality

ROOT = Path(os.environ.get("GEMMA_ROOT", Path(__file__).resolve().parents[1]))


def usable(r: dict) -> bool:
    return not quality.issues(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "traces" / "raw")
    parser.add_argument("--tasks-file", type=Path, help="only these tasks, so teachers are compared like for like")
    parser.add_argument("--samples", type=int, default=1, help="only sample indices below this")
    args = parser.parse_args()
    wanted = set(args.tasks_file.read_text().split()) if args.tasks_file else None

    by_teacher: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(args.raw.glob("*/*.json.gz")):
        with gzip.open(path, "rt") as fh:
            r = json.load(fh)
        if (wanted is None or r["instance_id"] in wanted) and r["sample"] < args.samples:
            by_teacher[r["teacher_model"] if r["teacher"].startswith("or-") else r["teacher"]].append(r)

    stats = []
    for teacher, rows in by_teacher.items():
        n = len(rows)
        good = [r for r in rows if usable(r)]
        # Average over rollouts that report a cost (older traces and direct APIs don't).
        priced = [r["cost_usd"] for r in rows if r.get("cost_usd")]
        cost = sum(priced) / len(priced) if priced else 0.0
        stats.append({
            "teacher": teacher, "n": n, "resolved": sum(r["resolved"] for r in rows) / n,
            "usable": len(good) / n, "cost": cost,
            "cost_per_usable": cost * n / len(good) if good else float("inf"),
            "calls": sum(r["tool_calls"] for r in good) / len(good) if good else float("nan"),
            "minutes": sum(r["agent_seconds"] for r in rows) / n / 60,
            "priced": bool(priced),
        })
    for s in stats:
        s["frontier"] = s["priced"] and not any(
            o is not s and o["priced"] and o["usable"] >= s["usable"] and o["cost"] <= s["cost"]
            and (o["usable"] > s["usable"] or o["cost"] < s["cost"]) for o in stats)

    print(f"{'teacher':<34} {'n':>3} {'resolved':>8} {'usable':>7} {'$/rollout':>9} {'$/usable':>9} "
          f"{'calls':>6} {'min':>5}  frontier")
    for s in sorted(stats, key=lambda s: (-s["usable"], s["cost"])):
        print(f"{s['teacher']:<34} {s['n']:>3} {s['resolved']:>8.0%} {s['usable']:>7.0%} "
              f"{s['cost']:>9.3f} {s['cost_per_usable']:>9.3f} {s['calls']:>6.1f} {s['minutes']:>5.1f}  "
              f"{'*' if s['frontier'] else ('(no cost data)' if not s['priced'] else '')}")
    small = min((s["n"] for s in stats), default=0)
    if small < 20:
        print(f"\nOnly {small} rollouts per teacher: a one-task difference moves pass rate by {1 / max(small, 1):.0%}.")


if __name__ == "__main__":
    main()
