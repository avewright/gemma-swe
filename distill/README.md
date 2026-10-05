# Distillation: teacher traces → LoRA adapter

Rejection sampling. A frontier teacher (DeepSeek, Kimi, GLM or Qwen through their
OpenAI-compatible APIs) solves the contest tasks as a real agent: same system prompt,
task message, tools and 40-call budget as the student, and it never sees the
maintainer fix. Only rollouts that pass the hidden tests are used to train a LoRA.

The teacher's commands run in a sandbox shaped like the contest's (`sandbox.py`): a
Docker container with no network, 4 GiB and 2 CPUs, holding only the task repo at
`/workspace` and its venv at `/venv`. Setup and grading run in separate containers, and
grading starts from a fresh copy (like the harness's Container B). This matters: in an
unsandboxed trial the teacher downloaded a newer release from PyPI, read other rollouts'
repos, and found `tasks.jsonl`, which holds the gold patches.

```
collect.py ──► traces/raw/<teacher>/<task>__<sample>.json.gz   (every rollout, pass or fail)
pack.py    ──► traces/packed/traces.parquet                     (zstd Parquet, one row per rollout)
train_lora.py ──► runs/lora/<name>/adapter                      (PEFT LoRA, safetensors)
export_adapter.py ──► variants/<name>/adapters/main_lora        (submission-ready)
```

The dev40 bench tasks (`scripts/dev40.txt`) are excluded from collection by default,
and training refuses traces from them, so the bench stays a fair test.

## Collect on a laptop (Docker)

Collection needs Docker, not a GPU. The API calls go out from the host; each rollout's
commands run in its own offline container.

```bash
cd gemma-swe
distill/collect_local.sh --teacher deepseek --samples 1 --limit 3      # trial
distill/collect_local.sh --teacher deepseek --samples 4 --concurrency 4
distill/.venv/bin/python distill/pack.py
```

`collect_local.sh` makes `distill/.venv` (openai, pyyaml, numpy, pyarrow), builds the
`gemma-distill` image (linux/amd64; emulated on Apple Silicon), reads keys from the
top-level `.env` and data from the top-level `data/`. Measured on an M-series Mac: about
3 minutes of setup plus 1.5–6 minutes of agent time per rollout, and roughly 300k–800k
prompt tokens per rollout (most of it repeated context the API can cache).

## Train and bench on the GPU pod

```bash
cd /workspace/gemma
pip install -r distill/requirements.txt          # in the training env; collect only needs openai
# API key in /workspace/gemma/.env, e.g. DEEPSEEK_API_KEY=...

# 1-2. Collect and pack (above; or here, if the pod can run Docker). Copy traces/ over.
#    Other teachers: --teacher kimi|glm|qwen --model <model id>, or --teacher custom
#    --base-url <url> --key-env <VAR> --model <id> for any OpenAI-compatible server.
#    Extra API fields: --extra-body '{"thinking": {"type": "enabled"}}'; add --echo-reasoning
#    if the API wants reasoning sent back during tool use.

# 3. Check the rendering, then train (QLoRA, one process over all visible GPUs)
python distill/train_lora.py --dry-run          # tokenizer only; writes runs/lora/lora-v1/example.txt
python distill/train_lora.py --name lora-v1

# 4. Bench against v4 before spending a submission
vllm serve google/gemma-4-31b-it-qat-w4a16-ct ... --enable-lora --max-lora-rank 16 \
  --lora-modules main_lora=/workspace/gemma/runs/lora/lora-v1/adapter
SERVED_MODEL=main_lora python scripts/bench.py --name lora-v1 --submission variants/v4-compact \
  --concurrency 3 --tasks $(cat scripts/dev40.txt)

# 5. Package
python distill/export_adapter.py --adapter runs/lora/lora-v1/adapter --name v5-lora
```

`example.txt` shows one rendered training example with the trained spans inside `⟦ ⟧`.
Only the student's own output is trained: for each step, the thinking channel, the tool
call, and the `<|tool_response>` token it stops on. Prompts, tool outputs, nudges and
stalled turns are masked.

What training keeps: resolved, submitted traces with well-formed calls and at most 2
commands that look outside `/workspace`, up to 2 per task (fewest tool calls first).
Parallel tool calls become consecutive single-call steps, since the student makes one
call per step. Traces longer than `--max-len` have their oldest tool outputs elided
(the last 5 stay), like the harness's own context compaction; all steps are still trained.

## Check these before a big run

1. **LoRA on the quantized model.** Kaggle serves the int4 QAT build; the adapter is
   trained on the bf16 weights (`--base google/gemma-4-31b-it`) with QLoRA. Train a
   few steps (`--epochs 0.05`), serve it with `--enable-lora`, and confirm vLLM loads it
   and the bench still produces tool calls, before a full run.
2. **Tool declarations.** Traces use the tool schemas from `scripts/run_one_task.py`,
   which approximate the harness's. The contest harness builds its own from the ADK
   tool functions. Differences in names or descriptions become a train/test mismatch.
3. **Earlier thoughts.** Training shows every earlier thought in context
   (`preserve_thinking`); at inference the harness likely drops them. And the second of a
   split parallel call has no thought, while at inference Gemma always opens one. Both are
   usually small effects, but they are differences.
4. **Data volume.** 129 public tasks minus the 40 holdout leaves 89. The first sandboxed
   trial passed 1 of 3; with 4 samples each, expect on the order of 100 passing traces. That is enough
   for a format-and-habits LoRA, not for new skills. More needs other task sources with
   runnable environments, which this pipeline does not fetch yet.
5. **Teacher model ids** change often. Only `deepseek` has a default (`deepseek-flash`,
   from the earlier trace script); pass `--model` for the others.
