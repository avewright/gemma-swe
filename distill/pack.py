#!/usr/bin/env python3
"""Pack raw traces into one zstd-compressed Parquet file (rejection sampling, step 2).

Raw rollouts (traces/raw/<teacher>/*.json.gz) stay the source of truth; this
rebuilds traces/packed/traces.parquet from all of them. Failed rollouts are kept
too (resolved=False), for later preference training or error analysis.
Conversations are stored as JSON strings: the long, repetitive system prompts and
tool outputs compress well with zstd. Load with pandas, pyarrow or
datasets.load_dataset("parquet", data_files=...).

  python distill/pack.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("GEMMA_ROOT", Path(__file__).resolve().parents[1]))
JSON_COLUMNS = ("messages", "tools")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "traces" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "traces" / "packed" / "traces.parquet")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.raw.glob("*/*.json.gz")):
        with gzip.open(path, "rt") as fh:
            row = json.load(fh)
        for key in JSON_COLUMNS:
            row[key] = json.dumps(row[key], ensure_ascii=False)
        rows.append(row)
    if not rows:
        raise SystemExit(f"no traces under {args.raw}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, args.out, compression="zstd", compression_level=9, row_group_size=256)

    raw_bytes = sum(p.stat().st_size for p in args.raw.glob("*/*.json.gz"))
    by_task: dict[str, int] = defaultdict(int)
    for r in rows:
        by_task[r["instance_id"]] += r["resolved"]
    passed = [r for r in rows if r["resolved"]]
    print(f"{len(rows)} rollouts, {len(passed)} resolved ({len(passed) / len(rows):.0%})")
    print(f"tasks with a passing trace: {sum(v > 0 for v in by_task.values())}/{len(by_task)}")
    for teacher in sorted({r["teacher"] for r in rows}):
        mine = [r for r in rows if r["teacher"] == teacher]
        print(f"  {teacher}: {sum(r['resolved'] for r in mine)}/{len(mine)} resolved, "
              f"mean tool calls on passes {sum(r['tool_calls'] for r in mine if r['resolved']) / max(1, sum(r['resolved'] for r in mine)):.1f}")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB; raw gz {raw_bytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
