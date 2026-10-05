#!/usr/bin/env python3
"""Measure how often the model emits a malformed edit, per system-prompt variant.

Replays real transcripts up to the turn where the model made a malformed
edit_file call, then resamples that turn N times under each system prompt and
counts well-formed vs malformed tool calls.

  /opt/vllm-venv/bin/python scripts/edit_probe.py --run dev40-v1 -n 20 \
      --prompt v1=prompts-archive/system.v1-repo-specific.md --prompt v3=variants/v3/prompts/system.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_one_task as rt  # noqa: E402  (tool schema, sampling, model name)

from openai import OpenAI  # noqa: E402

ROOT = Path("/workspace/gemma")
REQUIRED = {"edit_file": {"filepath", "old_string", "new_string"}, "write_file": {"filepath", "content"},
            "run_command": {"command"}, "read_file": {"filepath"}}


def well_formed(name: str, raw: str) -> bool:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return False
    need = REQUIRED.get(name)
    return need is None or (need <= set(args) and all(isinstance(args[k], str) for k in need))


def cut_points(transcript: list[dict]) -> list[list[dict]]:
    """Prefixes ending just before each malformed edit_file call (first per task only)."""
    for i, m in enumerate(transcript):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            if fn["name"] == "edit_file" and not well_formed("edit_file", fn["arguments"]):
                return [transcript[:i]]
    return []


def clean(msgs: list[dict]) -> list[dict]:
    out = []
    for m in msgs:
        m = {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id")}
        if m["role"] == "assistant":
            m.setdefault("content", "")
        out.append(m)
    return out


def sample(client, messages) -> str:
    resp = client.chat.completions.create(
        model=rt.MODEL, messages=messages, tools=rt.TOOLS, tool_choice="auto", max_tokens=4096,
        temperature=rt.SAMPLING.get("temperature", 0.2), top_p=rt.SAMPLING.get("top_p", 1.0),
        extra_body={"chat_template_kwargs": {"enable_thinking": rt.ENABLE_THINKING}})
    calls = resp.choices[0].message.tool_calls or []
    if not calls:
        return "no_call"
    c = calls[0].function
    ok = well_formed(c.name, c.arguments)
    return f"{c.name}:{'ok' if ok else 'MALFORMED'}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dev40-v1")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--prompt", action="append", required=True, help="label=path/to/system.md")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = ap.parse_args()
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    prefixes = []
    for f in sorted((ROOT / "runs" / "bench" / args.run / "transcripts").glob("*.json")):
        for p in cut_points(json.loads(f.read_text())):
            prefixes.append((f.stem, clean(p)))
    print(f"{len(prefixes)} cut points: {[t for t, _ in prefixes]}")
    for spec in args.prompt:
        label, path = spec.split("=", 1)
        system = (ROOT / path).read_text()
        jobs = [[{"role": "system", "content": system}] + p[1:] for _, p in prefixes for _ in range(args.n)]
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda m: sample(client, m), jobs))
        counts = Counter(results)
        bad = sum(n for k, n in counts.items() if k.endswith("MALFORMED"))
        print(f"{label:<10} malformed {bad}/{len(results)} ({100 * bad / len(results):.0f}%)  {dict(counts.most_common())}")


if __name__ == "__main__":
    main()
