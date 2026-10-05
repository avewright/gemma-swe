#!/usr/bin/env python3
"""Start vLLM for gemma-4-31b-it-qat-w4a16-ct with the scorer's settings and keep it running.

Pair with `harness_eval.py --external-server` so single-task runs don't reload the model.

  python serve.py --model /workspace/models/gemma-4-31b-it-qat-w4a16-ct
"""
import argparse
import os
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "1200")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--tp", type=int, default=1)
p.add_argument("--gpu-mem", type=float, default=0.90)
args = p.parse_args()

from adk_submission import VllmConfig, VllmServer

cfg = VllmConfig(model=args.model, port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
                 max_model_len=32768, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
                 enable_auto_tool_choice=True, enable_lora=True, max_loras=8, max_lora_rank=128,
                 tensor_parallel_size=args.tp, startup_timeout=60 * 20)
server = VllmServer(cfg)
server.start()
print(f"SERVER_READY {server.base_url}", flush=True)
while True:
    time.sleep(3600)
