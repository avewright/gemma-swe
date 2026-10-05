#!/usr/bin/env python3
"""Per-variant metrics from harness traces + results, for comparing configs on the same tasks.

Outcome counts only tasks whose gold patch passes in our environment (--gold). Behaviour metrics come
from every trace. Pass --group to restrict to one experiment's folder (e.g. v8).

  python scripts/analyze_evals.py runs/pod-evals --gold runs/pod-evals/gold_rich.jsonl --group v8
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


SCHEMA = {"run_command": {"command"}, "read_file": {"filepath", "start_line", "end_line"},
          "edit_file": {"filepath", "old_string", "new_string", "allow_multiple"}, "write_file": {"filepath", "content"},
          "submit_patch": set(), "get_status": set(), "get_code_neighbors": {"node", "edge_type", "max_neighbors"},
          "get_code_subgraph": {"nodes"}, "search_similar_code": {"query", "k"}}


def garbled(fn: str, args, obs: str) -> bool:
    """Tool call whose arguments arrived corrupted (Gemma's call delimiters broke)."""
    if "mandatory input parameters are not present" in obs:
        return True
    if not isinstance(args, dict) or fn not in SCHEMA:
        return False
    fp = str(args.get("filepath", ""))
    return bool(set(args) - SCHEMA[fn]) or any(c in fp for c in '`",') or "start_line" in fp


def trace_metrics(path: Path) -> dict:
    t = json.loads(path.read_text())
    calls, empty_submits, edit_fail, n_garbled = [], 0, 0, 0
    out_tok = []
    last_t = 0.0
    for s in t.get("steps", []):
        if s.get("source") != "agent":
            continue
        out_tok.append((s.get("metrics") or {}).get("completion_tokens") or 0)
        last_t = (s.get("extra") or {}).get("elapsed_s") or last_t
        obs = (s.get("observation") or {}).get("content") or ""
        for tc in s.get("tool_calls") or []:
            calls.append((tc["function_name"], json.dumps(tc.get("arguments"), sort_keys=True)))
            n_garbled += garbled(tc["function_name"], tc.get("arguments"), obs)
            if tc["function_name"] == "submit_patch" and '"files_changed": 0' in obs:
                empty_submits += 1
            if tc["function_name"] == "edit_file" and '"status": "error"' in obs:
                edit_fail += 1
    n = len(calls)
    b2b = sum(1 for i in range(1, n) if calls[i] == calls[i - 1])
    return {"calls": n, "b2b_repeat": b2b / n if n else 0, "any_repeat": (n - len(set(calls))) / n if n else 0,
            "reads": sum(f == "read_file" for f, _ in calls), "edits": sum(f == "edit_file" for f, _ in calls),
            "edit_fail": edit_fail, "empty_submits": empty_submits, "secs": last_t, "garbled": n_garbled,
            "out_tok_per_turn": st.mean(out_tok) if out_tok else 0,
            "first_edit": next((i + 1 for i, (f, _) in enumerate(calls) if f == "edit_file"), None)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("evals", type=Path)
    p.add_argument("--gold", type=Path)
    p.add_argument("--group", help="only runs under <evals>/<group>/")
    p.add_argument("--per-task", action="store_true")
    args = p.parse_args()
    gold = {}
    if args.gold and args.gold.exists():
        gold = {g["task"]: g["resolved"] for g in map(json.loads, (l for l in args.gold.read_text().splitlines() if l.startswith("{")))}
    root = args.evals / args.group if args.group else args.evals
    results = {}
    for f in root.rglob("results.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                results[(str(f.parent), r["task"])] = r  # one results.jsonl per run directory
    rows = defaultdict(list)
    for tr in root.rglob("trace_*.json"):
        variant, task = tr.parents[2].name, tr.stem.removeprefix("trace_")
        m = trace_metrics(tr)
        r = results.get((str(tr.parents[2]), task), {})
        m.update(task=task, resolved=r.get("resolved"), graded=gold.get(task, True), error=r.get("error", ""))
        rows[variant].append(m)
    print(f"{'variant':<18}{'runs':>5}{'solved/graded':>15}{'calls':>7}{'b2b rep':>9}{'any rep':>9}{'reads':>7}"
          f"{'edits':>7}{'edit fail':>10}{'garbled':>8}{'empty sub':>10}{'1st edit':>9}{'secs':>7}{'tok/turn':>9}{'timeouts':>9}{'overflow':>9}")
    for v, ms in sorted(rows.items()):
        graded = [m for m in ms if m["graded"] and m["resolved"] is not None]
        solved = sum(1 for m in graded if m["resolved"])
        med = lambda k: st.median([m[k] for m in ms]) if ms else 0
        fe = [m["first_edit"] for m in ms if m["first_edit"]]
        print(f"{v:<18}{len(ms):>5}{f'{solved}/{len(graded)}':>15}{med('calls'):>7.0f}{st.mean(m['b2b_repeat'] for m in ms):>9.0%}"
              f"{st.mean(m['any_repeat'] for m in ms):>9.0%}{med('reads'):>7.0f}{med('edits'):>7.0f}{sum(m['edit_fail'] for m in ms):>10}{sum(m['garbled'] for m in ms)/max(sum(m['calls'] for m in ms),1):>8.1%}"
              f"{sum(m['empty_submits'] for m in ms):>10}{(st.median(fe) if fe else float('nan')):>9.0f}{med('secs'):>7.0f}"
              f"{st.mean(m['out_tok_per_turn'] for m in ms):>9.0f}{sum('timeout' in m['error'] for m in ms):>9}{sum('ContextWindow' in m['error'] for m in ms):>9}")
    if args.per_task:
        tasks = sorted({m["task"] for ms in rows.values() for m in ms})
        vs = sorted(rows)
        print("\n" + f"{'task':<13}" + "".join(f"{v:>20}" for v in vs))
        for t in tasks:
            cells = []
            for v in vs:
                m = next((m for m in rows[v] if m["task"] == t), None)
                cells.append("-" if not m else f"{'PASS' if m['resolved'] else 'fail'}{'' if m['graded'] else '*'} {m['calls']}c {m['secs']:.0f}s")
            print(f"{t:<13}" + "".join(f"{c:>20}" for c in cells))
        print("* = gold patch fails in our environment; outcome not counted")


if __name__ == "__main__":
    main()
