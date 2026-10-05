#!/usr/bin/env python3
"""Re-grade failed traces that have a patch, with the current grading rule (sandbox.py).

Use after grading changes, e.g. the gold-baseline rule for tests that need the network.
Updates each trace file in place and marks it "regraded".

  python distill/regrade.py                      # every failed trace with a patch
  python distill/regrade.py --tasks requests_7502
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_one_task as rot  # noqa: E402
from sandbox import Sandbox  # noqa: E402


def regrade(path: Path, runs: Path) -> str:
    with gzip.open(path, "rt") as fh:
        row = json.load(fh)
    name = f"gd-regrade-{path.stem.split('.')[0]}-{row['teacher']}".replace("_", "-").lower()[:120]
    sb = Sandbox(row["instance_id"], name, runs / f"regrade-{row['teacher']}-{path.stem}", rot.ROOT, rot.DATA)
    try:
        sb.setup()
        sb.close()  # grading only; no agent container needed
        resolved, log = sb.grade(row["patch"])
    finally:
        sb.close()
        shutil.rmtree(sb.workdir, ignore_errors=True)
    if resolved:
        row.update(resolved=True, grade_log=log, regraded=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as fh:
            json.dump(row, fh)
        tmp.rename(path)
    return f"{'RESOLVED' if resolved else 'still failed'}  {row['id']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=rot.ROOT / "traces" / "raw")
    parser.add_argument("--runs-dir", type=Path, default=rot.ROOT / "runs" / "distill")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    todo = []
    for path in sorted(args.raw.glob("*/*.json.gz")):
        with gzip.open(path, "rt") as fh:
            row = json.load(fh)
        if not row["resolved"] and row["patch"].strip() and (not args.tasks or row["instance_id"] in args.tasks):
            todo.append(path)
    print(f"{len(todo)} failed traces with a patch to re-grade", flush=True)
    with ThreadPoolExecutor(args.concurrency) as pool:
        for line in pool.map(lambda p: regrade(p, args.runs_dir), todo):
            print(line, flush=True)


if __name__ == "__main__":
    main()
