#!/usr/bin/env python3
"""Audit the usable traces before training: dataset stats plus red flags quality.py doesn't gate.

  python distill/audit.py            # report
  python distill/audit.py --show 3   # also print 3 random traces, condensed, to read by hand
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
import re
import statistics
from pathlib import Path

import quality

ROOT = quality.ROOT
HOLDOUT = set((Path(__file__).resolve().parents[1] / "scripts" / "dev40.txt").read_text().split())
# Phrases that would mean the teacher knew or hinted at the answer key, or saw the harness.
# ("hidden tests" and "the maintainers' fix" are the system prompt's own words, so not flagged.)
LEAK = re.compile(r"gold (patch|fix)|test_patch|reference (patch|solution)|/g/|/data/|tasks\.jsonl|"
                  r"gemma-distill|docker", re.I)
SANDBOX = re.compile(r"/venv\b")
TEACHERISM = re.compile(r"as an ai\b|i cannot assist|i'm sorry, but|language model", re.I)


def load() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "traces" / "raw").glob("*/*.json.gz")):
        with gzip.open(path, "rt") as fh:
            rows.append(json.load(fh))
    return rows


def text_of(m: dict) -> str:
    parts = [m.get("content") or "", m.get("reasoning") or ""]
    for c in m.get("tool_calls") or []:
        parts.append(c["function"]["arguments"] or "")
    return "\n".join(parts)


def condensed(r: dict) -> str:
    out = [f"### {r['id']}  calls={r['tool_calls']}  verified={quality.verified(r['messages'])}"]
    for m in r["messages"][2:]:
        if m["role"] == "assistant":
            th = " ".join((m.get("content") or m.get("reasoning") or "").split())[:160]
            for c in m.get("tool_calls") or []:
                a = json.loads(c["function"]["arguments"] or "{}")
                arg = a.get("command") or a.get("filepath") or ""
                out.append(f"  - {th!r}\n      {c['function']['name']}: {str(arg)[:140]!r}")
                th = ""
        elif m["role"] == "tool" and '"status": "error"' in m["content"]:
            out.append(f"      ! error: {m['content'][:120]}")
    out.append("  patch files: " + ", ".join(quality.patch_files(r["patch"])))
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=0)
    args = parser.parse_args()
    rows = load()
    good = [r for r in rows if not quality.issues(r)]
    gold = quality.gold_patches()
    current_prompt = hashlib.sha256((ROOT / "submission" / "prompts" / "system.md").read_bytes()).hexdigest()[:12]

    print(f"rollouts {len(rows)} | usable {len(good)} | tasks with a usable trace "
          f"{len({r['instance_id'] for r in good})} of {len(set(gold) - HOLDOUT)} non-holdout")
    print("holdout traces among usable:", sum(r["instance_id"] in HOLDOUT for r in good))
    print("by repo:", dict(collections.Counter(r["repo"] for r in good)))
    print("by teacher:", dict(collections.Counter(r["teacher_model"] for r in good)))
    shas = collections.Counter(r["system_sha"] for r in good)
    print(f"system prompts: {dict(shas)} (current submission prompt = {current_prompt})")
    calls = [r["tool_calls"] for r in good]
    print(f"tool calls: median {statistics.median(calls)}, max {max(calls)}; "
          f"verified after last edit: {sum(quality.verified(r['messages']) for r in good)}/{len(good)}")

    steps = thought_steps = err_results = 0
    names = collections.Counter()
    flags = collections.defaultdict(list)
    for r in good:
        for m in r["messages"][2:]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                steps += 1
                thought_steps += bool((m.get("content") or m.get("reasoning") or "").strip())
                names.update(c["function"]["name"] for c in m["tool_calls"])
                t = text_of(m)
                if LEAK.search(t):
                    flags["mentions answer key or harness"].append((r["id"], LEAK.search(t).group()))
                if SANDBOX.search(t):
                    flags["uses sandbox-only path /venv"].append((r["id"], "/venv"))
                if TEACHERISM.search(t):
                    flags["assistant-style refusal/boilerplate"].append((r["id"], TEACHERISM.search(t).group()))
            elif m["role"] == "tool" and '"status": "error"' in m["content"]:
                err_results += 1
        g = gold.get(r["instance_id"])
        if g and not set(quality.patch_files(r["patch"])) & set(quality.patch_files(g)):
            flags["patch touches none of the gold patch's files"].append((r["id"], ",".join(quality.patch_files(r["patch"]))))
    print(f"steps {steps}, with a thought {thought_steps / steps:.0%}; tool results that were errors {err_results}")
    print("tools used:", dict(names))
    print("\nflags (not gated, review by hand):" if flags else "\nno flags")
    for kind, items in flags.items():
        print(f"  {kind}: {len(items)}")
        for iid, what in items[:6]:
            print(f"     {iid}  [{what}]")

    if args.show:
        random.seed(1)
        for r in random.sample(good, min(args.show, len(good))):
            print("\n" + condensed(r))


if __name__ == "__main__":
    main()
