"""What makes a trace good enough to train on. Used by train_lora.py and frontier.py.

Passing the hidden tests is necessary but not sufficient: the student copies habits,
so a trace must also be clean. issues() returns why a trace is rejected (empty = keep);
rank() orders the keepers of one task, best first.
"""

from __future__ import annotations

import json
import os
import re
from functools import cache
from pathlib import Path

ROOT = Path(os.environ.get("GEMMA_ROOT", Path(__file__).resolve().parents[1]))
DATA = Path(os.environ.get("GEMMA_DATA", ROOT / "data"))

CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
# Files an agent leaves behind by accident; the contest grades them as part of the patch.
STRAY = re.compile(r"(^|/)(repro\w*|debug\w*|scratch\w*|tmp\w*|test_(repro|fix|debug|issue)\w*)\.py$"
                   r"|\.(orig|rej|bak|swp)$|(^|/)(\.agent\.patch|agent\.patch)$", re.I)
TEST_RUN = re.compile(r"\bpytest\b|python3? (-m )?\S*test|python3? /tmp/|python3? -c")
# /venv exists only in our sandbox (the contest installs into the system Python), and looking for
# an installed or newer copy of the code "that might contain the fix" is answer-hunting.
SANDBOX_ONLY = re.compile(r"/venv\b")
HUNTING = re.compile(r"(contain|has|have|find|look for|already)[^.\n]{0,40}\b(fix|solution|answer|implementation)\b"
                     r"[^.\n]{0,60}(site-packages|installed|elsewhere|upstream|newer|pip|release)", re.I)
# Answering from memory of the real project's history; the student can't, and shouldn't claim to.
RECALL = re.compile(r"\bupstream (fix|commit|change|patch|version|code|implementation|rich|fastapi|requests)|"
                    r"recall (from|that) the (actual|real|upstream)|from memory of|in the (actual|real) (rich|fastapi|requests) repo", re.I)
MAX_NON_ENGLISH = 0.2   # share of thoughts; above this the trace is dropped
BLOAT = 6               # patch may change at most this many times the gold patch's lines (min 200)


@cache
def gold_patches() -> dict[str, str]:
    path = DATA / "tasks.jsonl"
    if not path.exists():
        return {}
    return {r["instance_id"]: r["patch"] for r in map(json.loads, path.read_text().splitlines()) if r}


def changed_lines(patch: str) -> int:
    return sum(1 for l in patch.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---")))


def patch_files(patch: str) -> list[str]:
    return [l[6:] for l in patch.splitlines() if l.startswith("+++ b/")]


def thoughts(messages: list[dict]) -> list[str]:
    return [(m.get("content") or m.get("reasoning") or "")[:400]
            for m in messages if m["role"] == "assistant" and m.get("tool_calls")]


def non_english(text: str) -> bool:
    return len(CJK.findall(text)) >= 3


def clean_thought(text: str) -> str:
    """A thought the student may learn, or "" if it recalls the upstream fix from memory, hunts for
    the answer outside the repo, mentions the sandbox-only /venv, or isn't in English. Teachers
    know these famous repos' history; the grade confirms the fix, but the student shouldn't learn
    to claim that knowledge. The step's tool call is still trained."""
    if non_english(text) or RECALL.search(text) or HUNTING.search(text) or SANDBOX_ONLY.search(text):
        return ""
    return text


def verified(messages: list[dict]) -> bool:
    """Ran a test or a script after its last edit, before submitting."""
    last_edit = checked = -1
    for i, m in enumerate(messages):
        for c in (m.get("tool_calls") or []) if m["role"] == "assistant" else []:
            name, args = c["function"]["name"], c["function"]["arguments"] or ""
            if name in ("edit_file", "write_file"):
                last_edit = i
            elif name == "run_command" and TEST_RUN.search(args):
                checked = i
    return checked > last_edit >= 0


def issues(row: dict, max_outside_refs: int = 2, require_submit: bool = True) -> list[str]:
    found = []
    if not row["resolved"]:
        found.append("not resolved")
    if require_submit and not row["submitted"]:
        found.append("never called submit_patch")
    if row.get("bad_calls"):
        found.append("malformed tool arguments")
    if (row.get("outside_refs") or 0) > max_outside_refs:
        found.append("kept looking outside the repo")
    if any(STRAY.search(f) for f in patch_files(row["patch"])):
        found.append("stray files in patch")
    gold = gold_patches().get(row["instance_id"])
    if gold and changed_lines(row["patch"]) > max(200, BLOAT * changed_lines(gold)):
        found.append("patch far larger than needed")
    messages = row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"])
    # Actions are judged here; thoughts are scrubbed per step by clean_thought() instead.
    actions = "\n".join(c["function"]["arguments"] or "" for m in messages if m["role"] == "assistant"
                        for c in m.get("tool_calls") or [])
    if SANDBOX_ONLY.search(actions):
        found.append("uses the sandbox-only /venv path")
    th = thoughts(messages)
    if th and sum(map(non_english, th)) / len(th) > MAX_NON_ENGLISH:
        found.append("thinks in another language")
    return found


def rank(row: dict) -> tuple:
    """Sort key among a task's keepers: verified first, then fewest calls, then fewest turns."""
    messages = row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"])
    return (not verified(messages), row["tool_calls"], row["turns"])
