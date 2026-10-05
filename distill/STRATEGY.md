# Plan to win: dataset, prompt, skills, budget

Written 2026-09-28. Leaderboard: top 0.15, a large group at 0.13, us 0.06 (v4-compact);
v6 (v5-fast + LoRA) pending. Deadline 2026-12-02. One submission a day, ~12 h to score.

## What the rules and the organizers say (drives everything below)

- **The hidden test set is ~120 tasks from private repositories**, split evenly between the public and
  private leaderboards. Same pipeline as the public 129: commits that change code + tests, the issue as the
  prompt, fail-to-pass verified, and a frontier model can pass it or get within one test of passing. So the
  test repos are *not* fastapi/rich/requests/httpx. What transfers is general skill on unfamiliar Python
  repos plus habits fitted to this harness, not knowledge of the four public repos.
- **Tasks run sequentially; hitting 12 h errors the whole submission** (organizer, 2026-09-24 and
  2026-09-28: fix "planned", not live). Setup time counts; validation doesn't.
  120 × (5 min + setup) is right at the limit.
- **Model:** only `gemma-4-31b-it-qat-w4a16-ct` (int4 QAT) on 4× L4. LoRA loading was fixed 2026-09-27.
- **Distillation from external LLMs: no ruling yet** (organizers "conferring" since 2026-09-24; thread
  742807). External data must be public and free or cheap. Prefer open-weight teachers
  (DeepSeek, GLM, MiniMax, Kimi, and MiMo if its weights are open); drop closed-model traces (Qwen 3.8 Max,
  4 traces) until there's a ruling.
- **Parallel sub-agents aren't supported.**
- **The real harness is available:** Kaggle dataset `metric/gemma-4-developer-agent-wheelhouse`
  (adk_submission 0.2.11, adk_eval_core 0.1.0, google_adk 1.36.1, swegemma 0.2.7 per the forum) and the
  organizer notebook `ryanholbrook/getting-started-gemma-4-developer-agent`. Known local gotcha: clear
  `$TMPDIR/swegemma_sp_cache_v8` when changing wheels (thread 744029).

## 1. Budget: fix first (free, and may be the biggest lever)

- Set `max_time_minutes: 4` (120 × ~5 min ≈ 10 h, leaving ~2 h margin for setup and slow tasks). An
  errored run scores nothing.
- Every second counts: low thinking (`thinking_level: low`, `thinking_budget: 512`), short prompt, no
  analyzer sub-agent (it doubles context and turns). v5-fast already does this; v6's score shows whether it
  helped.
- Measure turn latency on real L4s (or scale from the pod GPU) before trusting any per-task time.

## 2. Dataset: generalization over repo memorization

Target: 1–3k verified trajectories across hundreds of repos, in our exact tool format, short.

| Tier | Source | How | Size | Weight |
|---|---|---|---|---|
| A | Contest 129 (non-dev40) | existing pipeline; already 92 traces / 45 tasks | ~100–150 | high |
| A | **SWE-rebench-V2 Python tasks (diverse repos, Docker images)** | our sandbox + harness prompt/tools/budget; teachers MiMo Pro, DeepSeek V4 Pro, GLM | 500–1,500 | high |
| B | nvidia/Open-SWE-Traces, resolved Python (MiniMax M2.5, Qwen3.8-27B; ~115k available) | convert bash/str_replace to run_command/read_file/edit_file; append submit_patch | 1–3k after filtering | lower (or first stage) |
| B | 481 Open-SWE-Traces rows on rich/httpx (incl. 9 contest tasks) | same conversion | ~280 resolved | as B |

Tier A mirrors the hidden set best: PR-mined tasks on repos the model hasn't seen, graded by hidden tests,
with our prompt, tools and budget. SWE-rebench-V2 already ships images and FAIL_TO_PASS lists, so the task
adapter is mostly image + test-command plumbing.

**Selection for speed** (5-minute budget, ~20 turns realistic):
- resolved, called `submit_patch`, passes `quality.py`;
- ≤ 20 tool calls (prefer ≤ 12), first edit by call ~8, one targeted test run, short thoughts;
- per task keep the 1–2 shortest passing traces (the student copies pace as well as correctness);
- drop dev40 tasks (keep the bench honest) and closed-model teachers pending the ruling.

**Training:** re-render every example with the shipping system prompt (`--system-prompt`), rank 16–32,
1–2 epochs, bench vs. no-LoRA on dev40 with the real harness before submitting.

## 3. System prompt (benchmaxx the harness, not the repos)

Keep it short (tokens cost time). Add what the pipeline guarantees:
- The grading tests are edits to **existing** test files, and each whole file must pass. So: find the test
  file for the module you're changing (`grep -rln <symbol> tests/`) early. Its fixtures and style show what
  the new test will call. Don't break the other tests in it.
- The fix is usually small and local (a frontier model can solve it). New API: mirror the nearest sibling
  feature's names, defaults, error messages; exact identifiers from the issue.
- Partial or unsubmitted diffs are still graded. Get an edit in early and keep improving it.
- `hints_text`, if present, is maintainers' discussion; use it.
- Unchanged: /tmp for scratch, no test/pytest.ini/conftest edits, submit last, `get_status` free.

## 4. Skills (untested; verify with the real harness first)

Skills load into the container; `run_skill_script` is an ADK tool, not one of the 9 budget-gated harness
tools, so it probably doesn't count toward `max_tool_calls` (it does use time). Two generic scripts could
collapse ~5–8 turns into 2:
- `locate`: given identifiers from the issue, print ranked definitions with a few lines of context, plus
  related test files and their fixtures, in one call.
- `finish`: compile changed files, run the related test files quietly, list stray/untracked files and edits
  to tests/pytest.ini/conftest, show `git diff --stat`.

Risks: the skill instruction ADK injects costs ~500 tokens; Gemma may not call them reliably unless
training traces use them. Make the same scripts callable via `run_command` so traces and inference match.
Test on the pod with swegemma before any submission.

## Order of work

1. Pod up (4× L4 if available, so timing is real) + install the wheelhouse; reproduce v4/v5 on dev40
   with the real harness. This is our only offline signal.
2. Submit budget fix (v5-fast, `max_time_minutes: 4`) if v6 doesn't already beat 0.13.
3. Build Tier A on SWE-rebench-V2 (task adapter + teacher runs) while converting Tier B.
4. Prompt v7 + optional skills, benched on dev40.
5. Train LoRA v7 on A+B with the v7 prompt; bench; submit.
