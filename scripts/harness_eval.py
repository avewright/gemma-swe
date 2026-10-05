#!/usr/bin/env python3
"""Evaluate submission variants with the official swegemma harness, like the Kaggle scorer.

Starts one vLLM server for gemma-4-31b-it-qat-w4a16-ct (settings from the organizers' getting-started
notebook), then runs each variant in its own thread. Each variant goes through its tasks one at a time,
as the scorer does; variants share the server (vLLM batches them). Results are written as tasks finish:

  <out>/<variant>/results.jsonl       one line per task (resolved, duration, tool calls, patch size, error)
  <out>/<variant>/harness/...         swegemma's own results dir (traces/, logs/, patches/, test_outputs/)

  python harness_eval.py --model /workspace/models/gemma-4-31b-it-qat-w4a16-ct --data /workspace/data \
      --variants variants/v7-config variants/v8-control --tasks $(cat dev40.txt) --out /workspace/evals/run1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "1200")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TARGET = "gemma-4-31b-it-qat-w4a16-ct"
lock = threading.Lock()


def eval_settings(variant: Path) -> dict:
    """Read eval_config.yaml the way the organizers' notebook does. The scorer's default is no limit."""
    import yaml
    f = variant / "eval_config.yaml"
    raw = yaml.safe_load(f.read_text()) if f.exists() else {}
    sec = (raw or {}).get("evaluation", raw or {})
    turns = sec.get("max_turns", sec.get("max_llm_calls"))
    return {
        "timeout_seconds": int(sec.get("timeout_seconds", 300)),
        "max_tool_calls": int(sec["max_tool_calls"]) if "max_tool_calls" in sec else None,
        "max_time_minutes": float(sec["max_time_minutes"]) if "max_time_minutes" in sec else None,
        "max_turns": int(turns) if turns is not None else None,
    }


def run_variant(variant: Path, tasks: list, models, adapters, args) -> None:
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator

    name = variant.name
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    res_file = out / "results.jsonl"
    if res_file.exists():
        done = {json.loads(l)["task"] for l in res_file.read_text().splitlines() if l.strip()}
    settings = eval_settings(variant)
    limits, gen_constraints = build_submission_limits()
    kw = dict(
        tasks_path=args.data / "tasks.jsonl",
        snapshots_dir=args.data / "snapshots",
        results_dir=out / "harness",
        submission_dir=variant,
        models=models,
        sandbox="subprocess",
        timeout_seconds=settings["timeout_seconds"],
        max_tool_calls=settings["max_tool_calls"],
        max_turns=settings["max_turns"],
        limits=limits,
        generation_constraints=gen_constraints,
        adapter_manifest=adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=args.compaction_interval, overlap_size=2, token_threshold=args.compaction_threshold, event_retention_size=5),
        graph_dir=str(args.data / "graphs"),
        embeddings_dir=str(args.data / "embeddings"),
        wheels_dir=args.data / "wheels",
        verbose=False,
    )
    # The scorer's default is no per-task time limit; EvalConfig wants a number.
    kw["max_time_minutes"] = settings["max_time_minutes"] if settings["max_time_minutes"] is not None else args.default_minutes
    evaluator = Evaluator(EvalConfig(**kw))
    print(f"[{name}] settings {settings}; {len(tasks) - len(done)} tasks to run", flush=True)
    for i, task in enumerate(tasks, 1):
        if task.instance_id in done:
            continue
        start = time.time()
        row = {"task": task.instance_id, "variant": name}
        try:
            r = asyncio.run(evaluator.evaluate_task(task=task, task_index=i, total_tasks=len(tasks)))
            # Keep the agent's patch and the hidden-test output, to explain wrong fixes later.
            (out / "patches").mkdir(exist_ok=True)
            (out / "patches" / f"{task.instance_id}.patch").write_text(r.agent_patch or "")
            (out / "test_outputs").mkdir(exist_ok=True)
            (out / "test_outputs" / f"{task.instance_id}.log").write_text(r.test_output or "")
            row.update(resolved=bool(r.resolved), exit_code=r.test_exit_code, tool_calls=r.tool_calls,
                       duration=round(r.duration_seconds, 1), patch_chars=len(r.agent_patch or ""),
                       error=str(getattr(r, "error_message", None) or "")[:300])
        except Exception as e:  # keep going; a crash on one task shouldn't end the run
            row.update(resolved=False, error=f"{type(e).__name__}: {e}"[:300], trace=traceback.format_exc()[-1500:])
        row["wall"] = round(time.time() - start, 1)
        with lock:
            with res_file.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            n = sum(1 for _ in res_file.open())
            ok = sum(json.loads(l)["resolved"] for l in res_file.open())
            print(f"[{name}] {n}/{len(tasks)} {task.instance_id}: resolved={row['resolved']} "
                  f"wall={row['wall']}s calls={row.get('tool_calls')} | {ok}/{n} resolved", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="local path to gemma-4-31b-it-qat-w4a16-ct")
    p.add_argument("--data", type=Path, required=True, help="competition data dir (tasks.jsonl, snapshots/, graphs/, ...)")
    p.add_argument("--variants", type=Path, nargs="+", required=True)
    p.add_argument("--tasks", nargs="*", help="instance ids (default: all)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--default-minutes", type=float, default=30.0,
                   help="per-task cap when a variant has no max_time_minutes (the scorer has none; this bounds our spend)")
    p.add_argument("--external-server", action="store_true",
                   help="use a vLLM already running on 127.0.0.1:8000 (scripts/serve.py) instead of starting one")
    p.add_argument("--compaction-threshold", type=int, default=14336,
                   help="scorer value per the current HARNESS_README (2026-09-25+): 14,336")
    p.add_argument("--compaction-interval", type=int, default=5,
                   help="scorer value per the current HARNESS_README: 5 (older copies said 15)")
    args = p.parse_args()

    import litellm
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    from swegemma.models import load_tasks

    litellm.drop_params = True
    tasks = load_tasks(args.data / "tasks.jsonl")
    if args.tasks:
        want = set(args.tasks)
        tasks = [t for t in tasks if t.instance_id in want]
    variants = [v.resolve() for v in args.variants]
    args.out.mkdir(parents=True, exist_ok=True)
    for v in variants:
        shutil.copytree(v, args.out / v.name / "submission", dirs_exist_ok=True)

    # One server for all variants; adapters from every variant are mounted.
    manifests = [discover_adapters(str(v), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS) for v in variants]
    cfg = VllmConfig(model=args.model, port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
                     max_model_len=32768, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
                     enable_auto_tool_choice=True, enable_lora=True, max_loras=8, max_lora_rank=128,
                     tensor_parallel_size=args.tp, startup_timeout=60 * 20)
    server = VllmServer(cfg, adapter_manifest=manifests[0])
    if not args.external_server:
        server.start()
    print(f"vLLM at {server.base_url} ({'external' if args.external_server else 'started here'})", flush=True)
    models = server.create_model_registry(aliases=[TARGET], model_prefix="openai/", api_key="EMPTY")

    threads = [threading.Thread(target=run_variant, args=(v, tasks, models, m, args), name=v.name)
               for v, m in zip(variants, manifests)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not args.external_server:
        server.stop()
    print("all variants done", flush=True)


if __name__ == "__main__":
    main()
