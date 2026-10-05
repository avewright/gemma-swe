#!/usr/bin/env python3
"""Setup and grading steps that sandbox.py runs inside a gemma-distill container.

These containers see the contest data; the agent's container never does.

  setup --task ID               build /workspace (repo) and /venv, as the harness does
  grade --task ID --patch FILE  apply the agent's patch to a fresh copy and run the hidden tests
  grade --task ID --gold        the same for the maintainer patch (which tests fail offline anyway)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_one_task as rot  # noqa: E402

WORKSPACE = Path("/workspace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["setup", "grade"])
    parser.add_argument("--task", required=True)
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--gold", action="store_true", help="grade the maintainer patch (the offline baseline)")
    args = parser.parse_args()
    task = rot.load_task(args.task)
    if args.step == "setup":
        rot.prepare(task, WORKSPACE)
        return
    patch = task["patch"] if args.gold else args.patch.read_text()
    applied = bool(patch.strip()) and rot.apply_patch(WORKSPACE, patch, "agent.patch")
    code = rot.grade(task, WORKSPACE) if applied else 1
    # Which tests failed, so sandbox.py can compare with the gold patch's own offline failures.
    log = Path("/pytest.log").read_text() if Path("/pytest.log").exists() else ""
    failing = sorted({m.group(1) for m in re.finditer(r"^(?:FAILED|ERROR) (\S+)", log, re.M)})
    summary = re.findall(r"^.*\d+ (?:passed|failed|errors?)\b.* in [\d.]+s.*$", log, re.M)
    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|errors?)\b", summary[-1] if summary else "")}
    print("SUMMARY " + json.dumps({"applied": applied, "code": code, "failing": failing, "passed": counts.get("passed", 0)}))
    print("RESOLVED" if code == 0 else "FAILED")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
