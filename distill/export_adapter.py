#!/usr/bin/env python3
"""Put a trained LoRA adapter into a new submission variant (rejection sampling, step 4).

Copies a variant, adds adapters/<adapter-name>/ with only the two files the
harness accepts, and points the root agent at it. The sub-agent keeps the base
model. Checks the harness limits: rank <= 128, .safetensors only, < 3 GiB total.

  python distill/export_adapter.py --adapter runs/lora/lora-v1/adapter --name v5-lora
Then bench it before spending a submission (see distill/README.md).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(os.environ.get("GEMMA_ROOT", Path(__file__).resolve().parents[1]))
MAX_BYTES = 3 * 1024 ** 3
MAX_RANK = 128


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True, help="PEFT output dir")
    parser.add_argument("--name", required=True, help="new variant name, e.g. v5-lora")
    parser.add_argument("--from-variant", type=Path, default=ROOT / "variants" / "v4-compact")
    parser.add_argument("--adapter-name", default="main_lora")
    args = parser.parse_args()

    config = json.loads((args.adapter / "adapter_config.json").read_text())
    if config.get("r", 0) > MAX_RANK:
        raise SystemExit(f"rank {config['r']} is over the harness limit of {MAX_RANK}")
    weights = args.adapter / "adapter_model.safetensors"
    if not weights.exists():
        raise SystemExit(f"{weights} missing (save with safe_serialization=True)")

    dest = ROOT / "variants" / args.name
    if dest.exists():
        raise SystemExit(f"{dest} already exists")
    shutil.copytree(args.from_variant, dest)
    adapter_dir = dest / "adapters" / args.adapter_name
    adapter_dir.mkdir(parents=True)
    shutil.copy(args.adapter / "adapter_config.json", adapter_dir)
    shutil.copy(weights, adapter_dir)

    agent = dest / "agent.yaml"
    text = agent.read_text()
    if re.search(r"^adapter:", text, re.M):
        raise SystemExit(f"{agent} already names an adapter")
    agent.write_text(re.sub(r"^(model: .*)$", rf"\1\nadapter: {args.adapter_name}", text, count=1, flags=re.M))

    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    if total >= MAX_BYTES:
        shutil.rmtree(dest)
        raise SystemExit(f"variant would be {total / 1024 ** 3:.2f} GiB, over the 3 GiB limit")
    print(f"wrote {dest} ({total / 1e6:.0f} MB, rank {config.get('r')}, adapter {args.adapter_name})")
    print(agent.read_text())


if __name__ == "__main__":
    main()
