#!/usr/bin/env python3
"""Build a Hugging Face dataset from the traces, and optionally push it as a PRIVATE repo.

The traces contain contest task statements, repo code and patches. The contest rules
forbid sharing competition data with anyone who has not joined, so the repo is created
private and this script refuses to make it public.

  python distill/push_hf.py                                   # build traces/hf/ only
  python distill/push_hf.py --push --repo <user>/<name>       # needs HF_TOKEN (env or .env)

Two configs:
  sft       traces/sft/train.jsonl, the training set (messages/tools as JSON strings)
  rollouts  traces/packed/traces.parquet, every rollout, pass or fail
Load: load_dataset("<user>/<name>", "sft", split="train", token=...)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("GEMMA_ROOT", Path(__file__).resolve().parents[1]))


def build(out: Path) -> dict:
    shutil.rmtree(out, ignore_errors=True)
    (out / "sft").mkdir(parents=True)
    (out / "rollouts").mkdir(parents=True)
    rows = [json.loads(l) for l in (ROOT / "traces" / "sft" / "train.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        r["messages"] = json.dumps(r["messages"], ensure_ascii=False)
        r["tools"] = json.dumps(r["tools"], ensure_ascii=False)
    pq.write_table(pa.Table.from_pylist(rows), out / "sft" / "train-00000-of-00001.parquet", compression="zstd")
    rollouts = pq.read_table(ROOT / "traces" / "packed" / "traces.parquet")
    pq.write_table(rollouts, out / "rollouts" / "train-00000-of-00001.parquet", compression="zstd")
    card = refresh_card(ROOT / "traces" / "sft" / "README.md", rows)
    header = """---
configs:
- config_name: sft
  default: true
  data_files:
  - split: train
    path: sft/*.parquet
- config_name: rollouts
  data_files:
  - split: train
    path: rollouts/*.parquet
tags:
- agent
- swe
- distillation
---

"""
    usage = """
## Loading

```python
import json
from datasets import load_dataset
ds = load_dataset("REPO", "sft", split="train", token=True)   # private repo
ex = ds[0]
messages, tools = json.loads(ex["messages"]), json.loads(ex["tools"])
```
`messages` and `tools` are JSON strings because their shapes vary per step.
The `rollouts` config has every rollout (pass and fail) with `resolved`, `cost_usd`,
`patch` and the full conversation, for preference data or analysis.
"""
    (out / "README.md").write_text(header + card + usage)
    return {"sft": len(rows), "rollouts": rollouts.num_rows}


NAMES = {"deepseek-v4-pro": "DeepSeek V4 Pro", "deepseek/deepseek-v4-pro-0813": "DeepSeek V4 Pro",
         "xiaomi/mimo-v2.6-flash": "MiMo v2.6 Flash", "z-ai/glm-5.3": "GLM-5.3", "z-ai/glm-5.3-flash": "GLM-5.3 Flash",
         "qwen/qwen3.8-max-0902": "Qwen 3.8 Max"}


def refresh_card(path: Path, rows: list[dict]) -> str:
    """Rewrite the card's "This build" line from the rows actually exported."""
    import collections
    import statistics
    tasks = {r["instance_id"] for r in rows}
    teachers = collections.Counter(NAMES.get(r["teacher"], "other") for r in rows)
    named = ", ".join(f"{k} ({v})" for k, v in teachers.most_common() if k != "other")
    other = f", and {teachers['other']} from other teachers" if teachers["other"] else ""
    line = (f"**This build:** {len(rows)} examples from {len(tasks)} tasks (fastapi, rich, requests). "
            f"{sum(r['n_trained_tokens'] for r in rows) // 1000}k trained tokens,\nmedian "
            f"{statistics.median(r['n_tokens'] for r in rows) / 1000:.1f}k tokens per example. Teachers: {named}{other}.")
    text = path.read_text()
    start = text.index("**This build:**")
    text = text[:start] + line + text[text.index("\n\n", start):]
    path.write_text(text)
    return text


def token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for env in (ROOT.parent.parent / ".env", ROOT / ".env"):
        if env.is_file():
            for line in env.read_text().splitlines():
                key, _, value = line.partition("=")
                if key.strip() in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_KEY") and value.strip():
                    return value.strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "traces" / "hf")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--repo", help="e.g. avewright/gemma-swe-distill")
    args = parser.parse_args()
    counts = build(args.out)
    print(f"built {args.out}: sft {counts['sft']} examples, rollouts {counts['rollouts']}")
    if not args.push:
        return
    if not args.repo:
        raise SystemExit("--repo is required with --push")
    tok = token()
    if not tok:
        raise SystemExit("no HF token: set HF_TOKEN in the environment or in .env")
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(args.repo, repo_type="dataset").private:
        raise SystemExit(f"{args.repo} exists and is public; refusing to upload contest-derived data to it")
    readme = args.out / "README.md"
    readme.write_text(readme.read_text().replace('"REPO"', f'"{args.repo}"'))
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(args.out),
                      commit_message=f"Traces: sft {counts['sft']}, rollouts {counts['rollouts']}")
    print(f"pushed (private): https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
