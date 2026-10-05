#!/usr/bin/env python3
"""Build a self-contained HTML viewer for swegemma agent trajectories.

Finds every trace_<task>.json under the evals dir (the harness's own ATIF traces), joins the run's
results.jsonl row and the task (issue text, gold patch), flags common mistakes, and writes one HTML file.
Contest data stays local: open the file in a browser, don't publish it.

  python scripts/trace_viewer.py runs/pod-evals --data data/tasks.jsonl --out runs/trajectories.html
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LIMIT = 12000  # characters kept per tool result / argument / thought


def clip(s, n=LIMIT):
    if s is None:
        return ""
    s = s if isinstance(s, str) else json.dumps(s, indent=1)
    return s if len(s) <= n else s[:n] + f"\n… [{len(s) - n:,} more characters]"


def parse_result(content: str) -> dict:
    try:
        d = json.loads(content)
        return d if isinstance(d, dict) else {"raw": content}
    except Exception:
        return {"raw": content}


THOUGHT_RE = re.compile(r"<\|channel\>thought(.*?)(<channel\|>|$)", re.S)


def split_thought(text: str) -> tuple[str, str]:
    """Gemma sometimes writes its thought channel into the reply text; pull it out."""
    thoughts = [m.group(1).strip() for m in THOUGHT_RE.finditer(text or "")]
    clean = THOUGHT_RE.sub("", text or "").strip()
    return "\n\n".join(t for t in thoughts if t), clean


def build_run(trace_path: Path, results: dict, tasks: dict) -> dict:
    t = json.loads(trace_path.read_text())
    task_id = trace_path.stem.removeprefix("trace_")
    variant = trace_path.parent.parent.parent.name
    row = results.get((variant, task_id), {})
    steps, prev_call, flags_total = [], None, {}
    first_ts = None
    for s in t.get("steps", []):
        src = s.get("source")
        if src == "system":
            continue
        extra = s.get("extra") or {}
        elapsed = extra.get("elapsed_s")
        msg = s.get("message") or ""
        leaked, text = split_thought(msg if isinstance(msg, str) else json.dumps(msg))
        thought = s.get("reasoning_content") or ""
        if leaked:
            thought = (thought + "\n\n" + leaked).strip()
        flags = []
        if leaked:
            flags.append("thought leak")
        calls = []
        for tc in s.get("tool_calls") or []:
            name, args = tc.get("function_name"), tc.get("arguments") or {}
            key = (name, json.dumps(args, sort_keys=True))
            if key == prev_call:
                flags.append("repeat")
            prev_call = key
            fp = str(args.get("filepath", "")) if isinstance(args, dict) else ""
            if "`" in fp or "," in fp or "start_line" in fp:
                flags.append("malformed args")
            calls.append({"name": name, "args": {k: clip(v) for k, v in args.items()} if isinstance(args, dict) else clip(args)})
        obs = s.get("observation") or {}
        result = None
        if isinstance(obs, dict) and obs.get("content") is not None:
            res = parse_result(obs["content"])
            status = res.get("status", "")
            if status == "error":
                et = res.get("error_type", "error")
                flags.append({"FileEditError": "edit failed", "FileReadError": "read failed",
                              "CommandError": "command failed", "TimeoutExceeded": "timeout"}.get(et, et))
            if calls and calls[0]["name"] == "submit_patch" and res.get("files_changed") == 0:
                flags.append("empty submit")
            if calls and calls[0]["name"] == "run_command" and re.search(r"pip install|No module named", json.dumps(res)):
                flags.append("env/pip")
            result = {"status": status, "body": clip(json.dumps(res, indent=1) if "raw" not in res else res["raw"])}
        if src == "agent" and not calls and text:
            flags.append("no tool call")
        m = s.get("metrics") or {}
        for f in flags:
            flags_total[f] = flags_total.get(f, 0) + 1
        steps.append({"id": s.get("step_id"), "src": src, "t": round(elapsed, 1) if elapsed else None,
                      "text": clip(text), "thought": clip(thought), "calls": calls, "result": result, "flags": flags,
                      "pt": m.get("prompt_tokens"), "ct": m.get("completion_tokens")})
    task = tasks.get(task_id, {})
    n_calls = sum(len(s["calls"]) for s in steps)
    return {"id": f"{variant}/{task_id}/{trace_path.parent.parent.parent.parent.name}", "variant": variant, "task": task_id,
            "group": trace_path.parents[3].name, "resolved": row.get("resolved"), "error": row.get("error", ""),
            "duration": row.get("duration") or (steps[-1]["t"] if steps and steps[-1]["t"] else None),
            "calls": n_calls, "patch_chars": row.get("patch_chars"), "flags": flags_total,
            "statement": clip(task.get("problem_statement", ""), 20000), "gold": clip(task.get("patch", ""), 20000),
            "repo": task.get("repo", ""), "steps": steps}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("evals", type=Path, help="dir holding harness results (searched recursively)")
    p.add_argument("--data", type=Path, required=True, help="tasks.jsonl")
    p.add_argument("--gold", type=Path, help="optional gold_check output (lines of JSON) to mark tasks whose grading is reliable")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    tasks = {r["instance_id"]: r for r in map(json.loads, args.data.read_text().splitlines()) if r}
    results = {}
    for f in args.evals.rglob("results.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                results[(r["variant"], r["task"])] = r
    gold = {}
    if args.gold and args.gold.exists():
        for line in args.gold.read_text().splitlines():
            if line.startswith("{"):
                g = json.loads(line)
                gold[g["task"]] = bool(g["resolved"])
    runs = [build_run(f, results, tasks) for f in sorted(args.evals.rglob("trace_*.json"))]
    for r in runs:
        r["gold_ok"] = gold.get(r["task"])
    data = json.dumps(runs).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"{len(runs)} runs -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemma Trajectories</title>
<style>
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e2de;--accent:#2f5bd3;--ok:#1f7a45;--bad:#b3261e;--warn:#9a6700;
--thought:#f3effa;--thought-ink:#4b3a78;--call:#eef3fd;--res:#f6f6f4;--flag:#fde8e6;--flag-ink:#9b2218;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--panel:#1d1d1b;--ink:#ecebe7;--muted:#a09f99;--line:#33332f;--accent:#8fb0ff;--ok:#6cc58f;--bad:#ff8a80;--warn:#e6b450;
--thought:#26213a;--thought-ink:#cfc4f5;--call:#1c2438;--res:#232321;--flag:#3a1f1c;--flag-ink:#ffb4ab}}
:root[data-theme="dark"]{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe7;--muted:#a09f99;--line:#33332f;--accent:#8fb0ff;--ok:#6cc58f;--bad:#ff8a80;--warn:#e6b450;
--thought:#26213a;--thought-ink:#cfc4f5;--call:#1c2438;--res:#232321;--flag:#3a1f1c;--flag-ink:#ffb4ab}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{display:flex;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:5;flex-wrap:wrap}
header h1{font-size:15px;margin:0 8px 0 0}select,input{background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:5px 8px;font:inherit}
.wrap{display:grid;grid-template-columns:330px 1fr;min-height:calc(100vh - 52px)}
@media (max-width:820px){.wrap{grid-template-columns:1fr}#list{max-height:40vh}}
#list{border-right:1px solid var(--line);overflow:auto;max-height:calc(100vh - 52px);position:sticky;top:52px}
.run{padding:9px 14px;border-bottom:1px solid var(--line);cursor:pointer}.run:hover{background:var(--res)}.run.sel{background:var(--call);box-shadow:inset 3px 0 var(--accent)}
.run .top{display:flex;justify-content:space-between;gap:8px;font-weight:600}.run .meta{color:var(--muted);font-size:12px;margin-top:2px}
.badge{font-size:11px;padding:1px 7px;border-radius:10px;font-weight:600;white-space:nowrap}.ok{background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok)}
.bad{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}.na{background:var(--res);color:var(--muted)}
main{padding:16px 20px;min-width:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.kpis{display:flex;gap:18px;flex-wrap:wrap}.kpi b{display:block;font-size:18px}.kpi span{color:var(--muted);font-size:12px}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.chip{background:var(--flag);color:var(--flag-ink);font-size:12px;padding:2px 8px;border-radius:10px}
details>summary{cursor:pointer;color:var(--muted);font-weight:600;margin:4px 0}pre{white-space:pre-wrap;word-break:break-word;font:12px/1.45 var(--mono);margin:6px 0 0;max-height:420px;overflow:auto}
.step{border-left:2px solid var(--line);margin-left:8px;padding:2px 0 10px 16px;position:relative}.step:before{content:"";position:absolute;left:-6px;top:8px;width:10px;height:10px;border-radius:50%;background:var(--line)}
.step.flagged:before{background:var(--bad)}.shead{display:flex;gap:10px;align-items:center;color:var(--muted);font-size:12px;flex-wrap:wrap}
.thought{background:var(--thought);color:var(--thought-ink);border-radius:8px;padding:8px 10px;margin-top:6px}.say{margin-top:6px}
.call{background:var(--call);border-radius:8px;padding:8px 10px;margin-top:6px}.call .fn{font:600 13px var(--mono);color:var(--accent)}
.res{background:var(--res);border-radius:8px;padding:6px 10px;margin-top:6px;border-left:3px solid var(--line)}.res.error{border-left-color:var(--bad)}.res.ok{border-left-color:var(--ok)}
.flagtag{background:var(--flag);color:var(--flag-ink);font-size:11px;padding:1px 7px;border-radius:9px;font-weight:600}
.user{background:var(--res);border-radius:8px;padding:8px 10px}.empty{color:var(--muted);padding:40px;text-align:center}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:700}
</style></head><body>
<header><h1>Gemma trajectories</h1>
<select id="fVariant"><option value="">All variants</option></select>
<select id="fTask"><option value="">All tasks</option></select>
<select id="fOut"><option value="">Any outcome</option><option value="1">Resolved</option><option value="0">Failed</option></select>
<select id="fFlag"><option value="">Any step</option></select>
<label style="font-size:12px;color:var(--muted)"><input type="checkbox" id="fGold"> only reliably graded</label>
<span id="count" style="color:var(--muted);font-size:12px"></span></header>
<div class="wrap"><div id="list"></div><main id="main"><div class="empty">Pick a run on the left.</div></main></div>
<script>
const RUNS = __DATA__;
const $ = s => document.querySelector(s), esc = s => String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const uniq = a => [...new Set(a)].sort();
uniq(RUNS.map(r => r.variant)).forEach(v => $("#fVariant").add(new Option(v, v)));
uniq(RUNS.map(r => r.task)).forEach(v => $("#fTask").add(new Option(v, v)));
uniq(RUNS.flatMap(r => Object.keys(r.flags))).forEach(v => $("#fFlag").add(new Option("steps flagged: " + v, v)));
let sel = null;
function outcome(r){ if (r.resolved === true) return '<span class="badge ok">resolved</span>';
  if (r.resolved === false) return '<span class="badge bad">failed</span>'; return '<span class="badge na">no result</span>'; }
function filtered(){ const v=$("#fVariant").value, t=$("#fTask").value, o=$("#fOut").value, f=$("#fFlag").value, g=$("#fGold").checked;
  return RUNS.filter(r => (!v||r.variant===v)&&(!t||r.task===t)&&(!o||String(+!!r.resolved)===o)&&(!f||r.flags[f])&&(!g||r.gold_ok===true)); }
function renderList(){ const rs = filtered(); $("#count").textContent = `${rs.length} runs · ${rs.filter(r=>r.resolved).length} resolved`;
  $("#list").innerHTML = rs.map(r => { const nf = Object.values(r.flags).reduce((a,b)=>a+b,0);
    return `<div class="run ${sel===r.id?'sel':''}" data-id="${esc(r.id)}"><div class="top"><span>${esc(r.task)}</span>${outcome(r)}</div>
    <div class="meta">${esc(r.variant)} · ${r.calls} calls · ${r.duration?Math.round(r.duration)+'s':'?'} · ${nf} flags${r.gold_ok===false?' · grading unreliable':''}</div></div>`; }).join("") || '<div class="empty">No runs match.</div>';
  document.querySelectorAll(".run").forEach(el => el.onclick = () => { sel = el.dataset.id; renderList(); renderRun(RUNS.find(r => r.id === sel)); }); }
function renderArgs(c){ if (typeof c.args !== "object") return `<pre>${esc(c.args)}</pre>`;
  if (c.name === "run_command") return `<pre>$ ${esc(c.args.command)}</pre>`;
  if (c.name === "edit_file") return `<div class="lbl" style="margin-top:6px">${esc(c.args.filepath)}</div><div class="lbl">old_string</div><pre>${esc(c.args.old_string)}</pre><div class="lbl">new_string</div><pre>${esc(c.args.new_string)}</pre>`;
  return `<pre>${esc(JSON.stringify(c.args, null, 1))}</pre>`; }
function renderRun(r){ const onlyFlag = $("#fFlag").value;
  const chips = Object.entries(r.flags).sort((a,b)=>b[1]-a[1]).map(([k,v]) => `<span class="chip">${esc(k)} × ${v}</span>`).join("");
  const steps = r.steps.map(s => {
    if (s.src === "user") return `<div class="step"><div class="shead"><b>#${s.id} user</b></div><div class="user"><pre>${esc(s.text)}</pre></div></div>`;
    const flags = s.flags.map(f => `<span class="flagtag">${esc(f)}</span>`).join(" ");
    return `<div class="step ${s.flags.length?'flagged':''}"><div class="shead"><b>#${s.id}</b>${s.t!=null?`<span>${s.t}s</span>`:""}
      ${s.pt?`<span>${s.pt.toLocaleString()} prompt tok · ${s.ct} out</span>`:""} ${flags}</div>
      ${s.thought?`<details class="thought" ${s.thought.length<600?'open':''}><summary>thinking (${s.thought.length.toLocaleString()} chars)</summary><pre>${esc(s.thought)}</pre></details>`:""}
      ${s.text?`<div class="say"><pre>${esc(s.text)}</pre></div>`:""}
      ${s.calls.map(c => `<div class="call"><span class="fn">${esc(c.name)}</span>${renderArgs(c)}</div>`).join("")}
      ${s.result?`<details class="res ${esc(s.result.status)}" ${s.result.status==='error'?'open':''}><summary>result: ${esc(s.result.status||'ok')}</summary><pre>${esc(s.result.body)}</pre></details>`:""}</div>`; }).join("");
  $("#main").innerHTML = `<div class="card"><div class="shead" style="font-size:13px"><b style="color:var(--ink);font-size:16px">${esc(r.task)}</b> ${outcome(r)}
      <span>${esc(r.variant)}</span><span>${esc(r.repo)}</span>${r.gold_ok===false?'<span class="badge na">gold patch fails here too: grading unreliable</span>':''}</div>
    <div class="kpis" style="margin-top:10px"><div class="kpi"><b>${r.calls}</b><span>tool calls</span></div><div class="kpi"><b>${r.duration?Math.round(r.duration)+'s':'?'}</b><span>agent time</span></div>
      <div class="kpi"><b>${r.patch_chars ?? '?'}</b><span>patch chars</span></div><div class="kpi"><b>${r.steps.filter(s=>s.flags.length).length}</b><span>flagged steps</span></div></div>
    ${chips?`<div class="chips">${chips}</div>`:""}${r.error?`<div style="color:var(--bad);margin-top:8px">${esc(r.error)}</div>`:""}
    <details><summary>Issue text</summary><pre>${esc(r.statement)}</pre></details><details><summary>Gold patch (the maintainers' fix)</summary><pre>${esc(r.gold)}</pre></details></div>
    <div class="card">${steps}</div>`; }
["#fVariant","#fTask","#fOut","#fFlag","#fGold"].forEach(id => $(id).onchange = renderList);
renderList();
</script></body></html>"""

if __name__ == "__main__":
    main()
