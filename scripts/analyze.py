#!/usr/bin/env python3
"""Sort a benchmark run's tasks into failure categories.

  python3 scripts/analyze.py <run-name> [<run-name> ...]

Reads runs/bench/<run>/{results.jsonl,transcripts,patches}. The first matching
category wins, in this order:
  resolved            tests passed
  crash               runner crashed or server died (infrastructure, rerun it)
  no_edit             patch touches no source file: the agent never changed code
  broken_edit         a patched .py file does not compile
  stray_files         patch adds scratch files (repro scripts, tmp/)
  out_of_time         stopped by the time budget with an edit in place
  out_of_calls        stopped by the tool-call or turn budget with an edit in place
  wrong_fix           submitted (or ended) with a compiling edit, tests still fail
Also reports tool errors, repeated commands, and thinking volume per task.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/gemma")
SCRATCH = re.compile(r"(^|/)(repro|reproduce|test_repro|debug|scratch)[^/]*\.py$|^tmp/")


def patch_files(patch: str) -> dict[str, str]:
    """Map each file the patch adds or modifies to its added lines."""
    files: dict[str, list[str]] = {}
    current = None
    for line in patch.splitlines():
        if line.startswith("+++ "):
            current = None if line.endswith("/dev/null") else line[6:]
            if current:
                files[current] = []
        elif current and line.startswith("+") and not line.startswith("+++"):
            files[current].append(line[1:])
    return {k: "\n".join(v) for k, v in files.items()}


def broken_python(run: Path, iid: str, files: list[str]) -> list[str]:
    """Patched .py files that no longer parse, checked against the kept repo when present."""
    bad = []
    for f in files:
        if not f.endswith(".py"):
            continue
        path = Path("/root/bench") / run.name / iid / "repo" / f
        if not path.exists():
            continue
        try:
            ast.parse(path.read_text())
        except SyntaxError:
            bad.append(f)
    return bad


def transcript_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    msgs = json.loads(path.read_text())
    calls, errors, thinking = [], 0, 0
    for m in msgs:
        if m.get("role") == "assistant":
            thinking += len(m.get("reasoning") or m.get("reasoning_content") or "")
            for c in m.get("tool_calls") or []:
                fn = c.get("function", {})
                calls.append((fn.get("name"), fn.get("arguments")))
        elif m.get("role") == "tool" and '"status": "error"' in str(m.get("content", ""))[:40]:
            errors += 1
    names = Counter(n for n, _ in calls)
    repeats = sum(n - 1 for n in Counter(calls).values() if n > 1)
    literal_newlines = sum(1 for n, a in calls if n == "edit_file" and a and "\\\\n" in a)
    return {"calls": len(calls), "tool_errors": errors, "repeated_calls": repeats,
            "edits": names["edit_file"] + names["write_file"], "thinking_chars": thinking,
            "literal_newline_edits": literal_newlines}


def categorize(run: Path, r: dict) -> str:
    if r.get("resolved"):
        return "resolved"
    if str(r.get("stop", "")).startswith("crash") or r.get("stop") == "server_down":
        return "crash"
    patch_path = run / "patches" / f"{r['instance_id']}.patch"
    patch = patch_path.read_text() if patch_path.exists() else ""
    files = patch_files(patch)
    source = [f for f in files if not SCRATCH.search(f)]
    if not source:
        return "no_edit"
    if broken_python(run, r["instance_id"], source):
        return "broken_edit"
    if len(source) < len(files):
        return "stray_files"
    if r.get("stop") == "time":
        return "out_of_time"
    if r.get("stop") in ("max_tools", "max_turns"):
        return "out_of_calls"
    return "wrong_fix"


def main() -> None:
    for name in sys.argv[1:]:
        run = ROOT / "runs" / "bench" / name
        rows = {}
        for line in (run / "results.jsonl").read_text().splitlines():
            r = json.loads(line)
            rows[r["instance_id"]] = r
        cats = Counter()
        print(f"== {name}: {len(rows)} tasks")
        print(f"{'task':<16} {'category':<13} {'stop':<14} {'calls':>5} {'edits':>5} {'errs':>4} {'rept':>4} {'think':>6} {'secs':>5}")
        for iid, r in sorted(rows.items()):
            cat = categorize(run, r)
            cats[cat] += 1
            s = transcript_stats(run / "transcripts" / f"{iid}.json")
            flag = " literal-\\n" if s.get("literal_newline_edits") else ""
            print(f"{iid:<16} {cat:<13} {str(r.get('stop')):<14} {s.get('calls', 0):>5} {s.get('edits', 0):>5} "
                  f"{s.get('tool_errors', 0):>4} {s.get('repeated_calls', 0):>4} {s.get('thinking_chars', 0):>6} "
                  f"{r.get('agent_seconds') or 0:>5.0f}{flag}")
        scored = len(rows) - cats["crash"]
        print(f"\nresolved {cats['resolved']}/{scored}" + (f" ({cats['crash']} crashed, excluded)" if cats["crash"] else ""))
        for cat, n in cats.most_common():
            print(f"  {cat:<13} {n}")
        print()


if __name__ == "__main__":
    main()
