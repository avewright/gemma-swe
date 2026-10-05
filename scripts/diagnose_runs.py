#!/usr/bin/env python3
"""Explain why runs fail and why tool calls repeat, from harness traces + the gold patch.

Per run: how far the agent got (found the gold file? edited it? tested? submitted?) and why the run ended.
Per repeated tool call: what the previous identical call returned, context size, position, nudges, streak
length and what broke the streak. Writes a markdown report and a JSONL of every repeat event.

  python scripts/diagnose_runs.py runs/pod-evals --group e8 --data data/tasks.jsonl \
      --gold runs/pod-evals/gold.jsonl --out experiments/diag_e8.md
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

TRUNC = 4900  # run_command output is capped at 5000 chars by the harness


def gold_files(patch: str) -> set[str]:
    return {l[6:].strip() for l in patch.splitlines() if l.startswith("+++ b/")}


def result_class(fn: str, obs: str) -> str:
    try:
        d = json.loads(obs)
    except Exception:
        d = {}
    if "mandatory input parameters are not present" in obs:
        return "garbled call (missing arg)"
    st_ = d.get("status")
    if st_ == "error":
        et, det = d.get("error_type", ""), d.get("details") or {}
        if et == "CommandError":
            out = (det.get("stdout") or "") + (det.get("stderr") or "")
            if not out.strip():
                return "no output + exit 1 (e.g. grep no match)"
            if "SyntaxError" in out or "IndentationError" in out:
                return "command error: syntax error"
            if "No module named" in out:
                return "command error: missing module"
            if "Traceback" in out:
                return "command error: python traceback"
            return "command error: other"
        return {"FileEditError": "edit failed", "FileReadError": "read failed",
                "TimeoutExceeded": "command timeout"}.get(et, f"error: {et}")
    if fn == "run_command":
        out = (d.get("stdout") or "") + (d.get("stderr") or "")
        if not out.strip():
            return "ok, empty output"
        if len(out) >= TRUNC:
            return "ok, output truncated (≥5000 chars)"
        return "ok, useful output"
    if fn == "read_file":
        return "read ok (truncated)" if d.get("is_truncated") else "read ok"
    if fn == "submit_patch":
        return "submit (empty patch)" if d.get("files_changed") == 0 else "submit ok"
    return "ok" if st_ == "ok" else "other"


def mentions_file(call_args: dict, obs: str, files: set[str]) -> bool:
    text = json.dumps(call_args) + obs
    return any(f in text or f.rsplit("/", 1)[-1] in json.dumps(call_args) for f in files)


def analyze_trace(path: Path, task: dict, result: dict) -> tuple[dict, list[dict]]:
    t = json.loads(path.read_text())
    files = gold_files(task.get("patch", ""))
    calls = []  # (idx, fn, args, key, obs, prompt_tokens, elapsed, preceded_by_nudge)
    nudge_pending = False
    for s in t.get("steps", []):
        if s.get("source") == "user" and s.get("step_id", 0) > 2:
            nudge_pending = True
            continue
        if s.get("source") != "agent":
            continue
        obs = (s.get("observation") or {}).get("content") or ""
        m = s.get("metrics") or {}
        for tc in s.get("tool_calls") or []:
            args = tc.get("arguments") or {}
            key = (tc["function_name"], json.dumps(args, sort_keys=True))
            calls.append({"i": len(calls) + 1, "fn": tc["function_name"], "args": args, "key": key, "obs": obs,
                          "pt": m.get("prompt_tokens") or 0, "t": (s.get("extra") or {}).get("elapsed_s") or 0,
                          "nudge": nudge_pending, "text": (s.get("message") or "")[:300]})
            nudge_pending = False
    # --- progress against the gold patch
    touched = [c for c in calls if isinstance(c["args"], dict) and mentions_file(c["args"], c["obs"], files)]
    first_seen = min((c["i"] for c in touched), default=None)
    edits_gold = [c for c in calls if c["fn"] == "edit_file" and isinstance(c["args"], dict)
                  and any(str(c["args"].get("filepath", "")).lstrip("/").removeprefix("workspace/") == f for f in files)]
    edits_other = [c for c in calls if c["fn"] in ("edit_file", "write_file") and c not in edits_gold]
    ran_tests = any(c["fn"] == "run_command" and "pytest" in json.dumps(c["args"]) for c in calls)
    submitted = any(c["fn"] == "submit_patch" and '"files_changed": 0' not in c["obs"] for c in calls)
    err = result.get("error", "") or ""
    if "ContextWindow" in err:
        end = "context overflow"
    elif "timeout" in err.lower():
        end = "time cap"
    elif calls and calls[-1]["fn"] == "submit_patch":
        end = "submitted"
    elif len(calls) >= 95:
        end = "call cap"
    else:
        end = "stopped (nudges/turn end)"
    resolved = result.get("resolved")
    if resolved:
        stage = "SOLVED"
    elif first_seen is None:
        stage = "never looked at the gold file"
    elif not edits_gold and not edits_other:
        stage = "saw gold file, made no edit"
    elif not edits_gold:
        stage = "edited only other files (wrong location)"
    elif not any("\"status\": \"ok\"" in c["obs"] for c in edits_gold):
        stage = "edits to gold file all failed"
    else:
        stage = "edited gold file, fix wrong/incomplete"
    tid = path.stem.removeprefix("trace_")
    vdir = path.parents[2]
    patch_f, test_f = vdir / "patches" / f"{tid}.patch", vdir / "test_outputs" / f"{tid}.log"
    agent_patch = patch_f.read_text() if patch_f.exists() else None
    test_out = test_f.read_text() if test_f.exists() else ""
    failed_tests = re.findall(r"^FAILED (\S+)", test_out, re.M) + re.findall(r"^ERROR (\S+)", test_out, re.M)
    agent_files = gold_files(agent_patch) if agent_patch else set()
    run_extra = {"agent_patch_files": sorted(agent_files), "patch_hits_gold": bool(agent_files & files),
                 "patch_extra_files": sorted(agent_files - files), "failed_tests": failed_tests[:8],
                 "test_tail": "\n".join(test_out.strip().splitlines()[-4:]), "have_patch": agent_patch is not None,
                 "agent_patch": agent_patch or "", "gold_patch": task.get("patch", "")}
    run = {"task": tid, "calls": len(calls), "resolved": resolved, "stage": stage,
           "end": end, "first_saw_gold_at": first_seen, "gold_files": sorted(files),
           "edits_gold": len(edits_gold), "edits_other": len(edits_other), "ran_tests": ran_tests, "submitted": submitted,
           "peak_prompt": max((c["pt"] for c in calls), default=0), "secs": calls[-1]["t"] if calls else 0, **run_extra}
    # --- repeat events, grouped into streaks
    events, i = [], 1
    while i < len(calls):
        if calls[i]["key"] != calls[i - 1]["key"]:
            i += 1
            continue
        start = i - 1
        j = i
        while j + 1 < len(calls) and calls[j + 1]["key"] == calls[start]["key"]:
            j += 1
        first = calls[start]
        breaker = calls[j + 1] if j + 1 < len(calls) else None
        earlier_same = sum(1 for c in calls[:start] if c["key"] == first["key"])
        events.append({"task": run["task"], "tool": first["fn"], "streak": j - start + 1,
                       "prev_result": result_class(first["fn"], first["obs"]),
                       "same_result_each_time": len({calls[k]["obs"] for k in range(start, j + 1)}) == 1,
                       "at_call": first["i"], "position": round(first["i"] / max(len(calls), 1), 2),
                       "prompt_tokens": first["pt"], "after_nudge": any(calls[k]["nudge"] for k in range(start + 1, j + 1)),
                       "had_text": bool(any(calls[k]["text"].strip() for k in range(start + 1, j + 1))),
                       "earlier_identical_calls": earlier_same,
                       "broken_by": ("run ended: " + end) if breaker is None else
                                    f"{breaker['fn']}" + (" (nudge)" if breaker["nudge"] else ""),
                       "cmd": (first["args"].get("command") if isinstance(first["args"], dict) else None) or
                              json.dumps(first["args"])[:120]})
        i = j + 1
    run["repeat_streaks"] = len(events)
    run["repeated_calls"] = sum(e["streak"] - 1 for e in events)
    return run, events


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("evals", type=Path)
    p.add_argument("--group")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--gold", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    tasks = {r["instance_id"]: r for r in map(json.loads, args.data.read_text().splitlines()) if r}
    gold_ok = {}
    if args.gold and args.gold.exists():
        gold_ok = {g["task"]: g["resolved"] for g in map(json.loads, (l for l in args.gold.read_text().splitlines() if l.startswith("{")))}
    root = args.evals / args.group if args.group else args.evals
    results = {}
    for f in root.rglob("results.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                results[(str(f.parent), r["task"])] = r  # one results.jsonl per run directory
    runs, events = defaultdict(list), defaultdict(list)
    for tr in sorted(root.rglob("trace_*.json")):
        variant, task = tr.parents[2].name, tr.stem.removeprefix("trace_")
        run, ev = analyze_trace(tr, tasks.get(task, {}), results.get((str(tr.parents[2]), task), {}))
        run["graded"] = gold_ok.get(task, True)
        runs[variant].append(run)
        for e in ev:
            e["variant"] = variant
        events[variant].extend(ev)
    L = [f"# Diagnostics: {args.group or 'all runs'}", "",
         "Generated by `scripts/diagnose_runs.py`. Outcomes count only tasks whose gold patch passes here.", ""]
    # ---- 1. where runs fail
    L += ["## 1. How far each run got (vs the gold patch)", "",
          "| variant | runs | solved | never looked at gold file | saw it, no edit | edited wrong file | gold edits all failed | edited gold file, fix wrong | median call when gold file first seen |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v, rs in sorted(runs.items()):
        g = [r for r in rs if r["graded"]]
        c = Counter(r["stage"] for r in g)
        fs = [r["first_saw_gold_at"] for r in g if r["first_saw_gold_at"]]
        L.append(f"| {v} | {len(g)} | {c['SOLVED']} | {c['never looked at the gold file']} | {c['saw gold file, made no edit']} | "
                 f"{c['edited only other files (wrong location)']} | {c['edits to gold file all failed']} | "
                 f"{c['edited gold file, fix wrong/incomplete']} | {st.median(fs) if fs else '–'} |")
    L += ["", "## 2. Why each run ended", "", "| variant | submitted | time cap | call cap | context overflow | stopped (nudges/turn end) | ran tests | median peak prompt tokens |", "|---|---|---|---|---|---|---|---|"]
    for v, rs in sorted(runs.items()):
        c = Counter(r["end"] for r in rs)
        L.append(f"| {v} | {c['submitted']} | {c['time cap']} | {c['call cap']} | {c['context overflow']} | {c['stopped (nudges/turn end)']} | "
                 f"{sum(r['ran_tests'] for r in rs)}/{len(rs)} | {st.median(r['peak_prompt'] for r in rs):.0f} |")
    # ---- 3. repeats
    L += ["", "## 3. Why tool calls repeat", "",
          "A *streak* is a run of identical consecutive tool calls. For each streak: what the first call returned, "
          "whether the result was identical every time, context size, nudges, and what finally broke it.", ""]
    for v, ev in sorted(events.items()):
        if not ev:
            L += [f"### {v}: no repeat streaks", ""]
            continue
        rep = sum(e["streak"] - 1 for e in ev)
        total_calls = sum(r["calls"] for r in runs[v])
        L += [f"### {v}: {len(ev)} streaks, {rep} repeated calls ({rep / max(total_calls, 1):.0%} of {total_calls} calls)", "",
              "| previous result | streaks | repeated calls | median streak | max streak | identical result each time | median prompt tokens | after a nudge |",
              "|---|---|---|---|---|---|---|---|"]
        by = defaultdict(list)
        for e in ev:
            by[e["prev_result"]].append(e)
        for k, es in sorted(by.items(), key=lambda kv: -sum(e["streak"] - 1 for e in kv[1])):
            L.append(f"| {k} | {len(es)} | {sum(e['streak'] - 1 for e in es)} | {st.median(e['streak'] for e in es):.0f} | "
                     f"{max(e['streak'] for e in es)} | {sum(e['same_result_each_time'] for e in es)}/{len(es)} | "
                     f"{st.median(e['prompt_tokens'] for e in es):.0f} | {sum(e['after_nudge'] for e in es)} |")
        pos = Counter("early (first third)" if e["position"] < 0.34 else "middle" if e["position"] < 0.67 else "late (last third)" for e in ev)
        brk = Counter(e["broken_by"] for e in ev)
        tools = Counter(e["tool"] for e in ev)
        L += ["", f"- **Tool repeated:** {dict(tools.most_common())}",
              f"- **Where in the run:** {dict(pos)}",
              f"- **Streak broken by:** {dict(brk.most_common(6))}",
              f"- **Gemma wrote any text during the streak:** {sum(e['had_text'] for e in ev)}/{len(ev)}",
              f"- **The same call had also been made earlier in the run (before the streak):** {sum(e['earlier_identical_calls'] > 0 for e in ev)}/{len(ev)}",
              "", "Longest streaks:", ""]
        for e in sorted(ev, key=lambda e: -e["streak"])[:5]:
            L.append(f"- {e['task']}: ×{e['streak']} `{str(e['cmd'])[:110]}` after *{e['prev_result']}* at call {e['at_call']}, "
                     f"{e['prompt_tokens']:,} prompt tokens → broken by {e['broken_by']}")
        L.append("")
    # ---- 3b. wrong fixes: what the hidden tests said and how the patch differs from gold
    wrong = [(v, r) for v, rs in sorted(runs.items()) for r in rs
             if r["graded"] and not r["resolved"] and r.get("have_patch") and r["agent_patch"].strip()]
    if wrong:
        L += ["## 3b. Runs that produced a patch but failed", "",
              "Which hidden tests failed, which files the patch touched vs the gold patch, and both diffs.", ""]
        for v, r in wrong:
            L += [f"### {v} / {r['task']} ({r['stage']}, ended: {r['end']})", "",
                  f"- **Patch files:** {r['agent_patch_files']} · **gold files:** {r['gold_files']} · "
                  f"extra files touched: {r['patch_extra_files'] or 'none'}",
                  f"- **Failed hidden tests:** {r['failed_tests'] or '(none listed)'}",
                  f"- **Test output tail:** `{r['test_tail'][-300:].replace(chr(10), ' ⏎ ')}`", "",
                  "<details><summary>agent patch vs gold patch</summary>", "", "```diff", r["agent_patch"][:3000], "```", "",
                  "```diff", r["gold_patch"][:2000], "```", "</details>", ""]
    # ---- 4. per run table
    L += ["## 4. Every run", "", "| variant | task | result | stage | ended | calls | repeated | gold file first seen at call | gold-file edits | other edits | tests | secs |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, rs in sorted(runs.items()):
        for r in sorted(rs, key=lambda r: r["task"]):
            res = "PASS" if r["resolved"] else ("fail" if r["graded"] else "fail*")
            L.append(f"| {v} | {r['task']} | {res} | {r['stage']} | {r['end']} | {r['calls']} | {r['repeated_calls']} | "
                     f"{r['first_saw_gold_at'] or '–'} | {r['edits_gold']} | {r['edits_other']} | {'yes' if r['ran_tests'] else 'no'} | {r['secs']:.0f} |")
    L += ["", "\\* gold patch fails in our environment; not counted."]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    ev_path = args.out.with_suffix(".repeats.jsonl")
    ev_path.write_text("".join(json.dumps(e) + "\n" for evs in events.values() for e in evs))
    print(f"wrote {args.out} and {ev_path} ({sum(len(v) for v in runs.values())} runs, {sum(len(v) for v in events.values())} streaks)")


if __name__ == "__main__":
    main()
