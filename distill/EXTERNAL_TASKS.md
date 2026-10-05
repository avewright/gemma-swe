# External data for the contest repos (TODO: run MiMo Pro, pull traces on the pod)

Updated 2026-09-28. Goal: more in-distribution training data on the contest repos
(fastapi 67 tasks, rich 48, requests 13, httpx 1), beyond the 89 non-holdout contest tasks
(45 have a quality trace as of this date). The goal is to win: data on these repos is welcome,
and only the dev40 bench tasks are excluded, so the bench stays a useful measure.

Everything below was found by reading only the id/repo columns remotely. **Pull the real data on
the GPU pod, not the laptop.**

## 1. Ready-made trajectories on contest repos: nvidia/Open-SWE-Traces

[nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces), CC-BY-4.0,
42.6 GB, v1.2 (2026-09-26). Harnesses: OpenHands, SWE-agent, mini-swe-agent. Models: Qwen3.6/3.8-27B,
Qwen3.5-122B, MiniMax-M2.5, DeepSeek-V4-Flash. Rows carry `messages`, `tools`, `resolved`
(1/0/-1), and `metadata.reference_patch`.

**481 rows on contest repos, all on 28 rich tasks and 5 httpx tasks (none on fastapi or requests):**
rich 253 resolved / 130 failed / 33 unknown; httpx 27 / 32 / 6. Every one of the 28 rich tasks
has at least one resolved trace. The exact rows (file, instance_id, repo, resolved) are in
`open_swe_traces_contest_rows.json`, so the pod can read just those shards.

9 of the 28 rich tasks are **contest tasks** with resolved traces: rich_2943, 3006, 3067, 3105 (dev40,
drop), 3130, 3180, 3296, 3518, 3718.

For volume, the whole set has **~115k resolved Python trajectories** (by config, resolved Python rows:
mini-swe-agent qwen38_27b scale-swe 27k, sweagent qwen36_27b scale-swe 19k, mini-swe-agent
qwen36_27b scale-swe 14k, plus 3–7k per swe-rebench-v2 config). The `openhands/*_flash` and
`openhands/qwen36_27b` configs report `resolved` 0 everywhere; treat those labels as missing.

Caveats: the tools are OpenHands/SWE-agent/mini-swe-agent tools (bash + str_replace editor), not ours
(`run_command`, `read_file`, `edit_file`, `submit_patch`), and trajectories often exceed our 40-call
budget. Convert, keep resolved traces with ≤40 calls that end in a submit, and run them through
`quality.py`.

## 2. Other trajectory datasets: almost nothing on these repos

| Dataset | Contest-repo rows |
|---|---|
| nebius/SWE-rebench-openhands-trajectories (67k) | 0 |
| nvidia/SWE-Hero-openhands-trajectories (34k) | 0 |
| SWE-bench/SWE-smith-trajectories (102k) | 0 |
| nebius/SWE-agent-trajectories (80k) | 0 |
| nvidia/SWE-Zero-openhands-trajectories (318k) | fastapi 59 rows / 22 tasks, rich 142 / 50 (execution-free: never runs tests) |
| AlienKevin/SWE-ZERO-96K-trajectories | rich 84 rows / 28 tasks (execution-free) |
| OpenHandsCommunity/SWE-bench-devin-full | requests 9 (old Devin runs) |
| nvidia/Nemotron-SFT-SWE-v3.5 | unknown (no repo column; scan message text on the pod) |

Not checked (no parquet export or no id column): FineEnvs/repo2rlenv-swe-flow, ByteDance-Seed/Multi-SWE-RL,
ibragim-bad/swe_rebench_07_2026_trajectories, Dorothydu/SWE-Dev, R2E-Gym/R2EGym-SFT-Trajectories,
SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K, hamishivi/swerl-tmax-15k,
JierunChen/swerebench-filtered-openhands-minimax-m2_5-corrected-completions.

## 3. Tasks we can run MiMo Pro on: 369, listed in `external_tasks.jsonl`

| Source | fastapi | rich | requests | httpx |
|---|---|---|---|---|
| princeton-nlp/SWE-bench | 28 | | 44 | |
| nebius/SWE-rebench-V2 (pre-built Docker images) | | 28 | | |
| nebius/SWE-rebench-V2-PRs (install instructions only) | | 52 | | 47 |
| SWE-Gym/SWE-Gym-Raw | 2 | 13 | | 154 |
| SWE-bench-Live/SWE-bench-Live | | | 1 | |
| **Total unique** | **30** | **93** | **45** | **201** |

De-duplicated by instance_id; `also_in` lists other datasets that carry the same task (e.g.
internlm/SWE-Fixer-Train-110K, PrimeIntellect/SWE-rebench-V2, whitecircle/swe-rebench-v2-clean-python-tasks).
No hits in R2E-Gym, SWE-smith, SWE-bench_Pro, SWE-bench-extra, SWE-rebench (v1), SWE-rebench-leaderboard,
SWE-Dev-train, SWE-Lego, SWE-Synth, AweAI-Team/Scale-SWE.

**27 contest tasks are in here** (26 rich, requests_6644), matched on PR number and `base_commit`; see
`contest_overlap.json`. 8 of them are dev40 and must be dropped from anything we import or run:
rich_3052, 3063, 3105, 3469, 3472, 3486, 3506, requests_6644. Otherwise, fastapi, requests and httpx
public tasks are older than the contest's (fastapi public ≤9468 vs contest 5077–15800, mostly 12942+).

## 4. Mine the repos ourselves (biggest pool)

Commits on the default branch that change both source `.py` files and tests (from blobless clones):

| Repo | All time | Since 2023 |
|---|---|---|
| fastapi/fastapi | 339 | 159 |
| Textualize/rich | 394 | 72 |
| psf/requests | 309 | 25 |
| encode/httpx | 443 | 55 |

About 1,485 candidate tasks, 10× the contest's 129 and in its exact format (`base_commit`, `patch`,
`test_patch`, `problem_statement`). They'd be built the way the contest built its tasks: split each PR into
source patch and test patch, write the problem statement from the linked issue or PR description,
and grade with FAIL_TO_PASS tests. Recent PRs share the contest tasks' dependency versions, so the
contest's `snapshots/` and `wheels/` may cover many of them. Exclude the 129 contest instance ids
(and definitely dev40).

## Plan

1. **Pod first.** Both gemma pods (`runpod-gemma`, `runpod-gemma-lora`) refused SSH on 2026-09-28.
2. Pull the 481 Open-SWE-Traces rows on contest repos (item 1), convert to our tools, gate, add to SFT.
   Cheapest win: no rollouts needed.
3. Write the task adapter so `collect.py`/`rollout.py`/`sandbox.py` accept SWE-bench-format tasks
   (repo at `base_commit`, deps offline, FAIL_TO_PASS/PASS_TO_PASS grading). Start with the 28
   SWE-rebench-V2 rich tasks (Docker images exist), then SWE-bench requests/fastapi.
4. Run **MiMo v2.6 Pro**: `--teacher openrouter --model xiaomi/mimo-v2.6-pro --samples 2`.
   Pilot: `requests_7309` resolved in 7 calls, $0.0066, 118 s. Even at 10× that, 369 tasks × 2 ≈ $50.
5. Build the item-4 mined tasks, especially fastapi since 2023 (159), since fastapi is 67 of 129
   contest tasks and has the fewest public tasks.
6. Gate with `quality.py`, update `traces/sft/train.jsonl` + `snapshot_ids.json`, push to
   `avewright/gemma-swe-distill-traces`. For volume, optionally add ~115k general resolved Python traces
   from Open-SWE-Traces, weighted below the in-repo data.
