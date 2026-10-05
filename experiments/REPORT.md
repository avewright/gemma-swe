# Gemma 4 31B agent experiments

Living log. Each experiment changes one thing, runs fixed tasks, and reports outcomes plus behaviour
from the traces. Newest at the bottom. Last updated 2026-09-29.

## Setup (applies to every experiment unless stated)

- **Harness:** the official scorer stack, installed from the organizers' wheelhouse: swegemma 0.2.7,
  adk_submission 0.2.11, google-adk 1.36.1, vLLM 0.19.1, transformers 5.13.1. Sandbox `subprocess`, as in the
  organizers' notebook. Runner: `scripts/harness_eval.py` (one vLLM server, variants in parallel threads,
  tasks one at a time per thread). Analysis: `scripts/analyze_evals.py`. Viewer: `scripts/trace_viewer.py`.
- **Model:** `gemma-4-31b-it-qat-w4a16-ct`, the exact Kaggle model files (`google/gemma-4/other/.../2`),
  served with the scorer's vLLM settings (gemma4 tool/reasoning parsers, max_model_len 32768, LoRA enabled).
- **Hardware:** 1× RTX 6000 Ada 48 GB (RunPod), not Kaggle's 4× L4. Several streams share the GPU, so
  per-turn latency is not Kaggle's; compare seconds only between variants in the same experiment.
- **Tasks:** dev40 holdout. Each run is **one sample** per task and variant. At temperature 1.0 outcomes are
  random, so one pass or fail per task is weak evidence; read differences of 1–2 tasks as noise.

### Grading reliability in our environment

Gold patches graded with the harness's own `verify_task` (`scripts/gold_check.py`):

| Gold passes (outcome counted) | Gold fails (outcome not counted) |
|---|---|
| rich_3052, 3063, 3105, 3472, 3506, 3676, 3777, 3905, 3930, 3938, 3942, 3953 | rich_3469, rich_3486 (required test node fails); requests_7205 (4 pre-existing test failures); fastapi_14463, fastapi_14964 (test collection errors) |

The pod's Python needs pytest installed alongside the harness (the sandbox venv inherits site-packages);
Kaggle's image has it. fastapi/requests failures match forum reports of missing test dependencies.

### Metric definitions

- **solved/graded:** tasks resolved / tasks whose gold passes here.
- **b2b repeat:** share of tool calls identical (name + arguments) to the call just before.
- **any repeat:** share of tool calls identical to any earlier call in the run.
- **reads, edits:** median `read_file` / `edit_file` calls per run. **edit fail:** total failed edits.
- **empty sub:** `submit_patch` calls that submitted nothing. **1st edit:** median call index of the first edit.
- **secs:** median agent seconds. **tok/turn:** mean output tokens per model turn.

---

## E1. Sampling temperature 0.2 vs 1.0 (thinking off)

**Question.** Traces at temperature 0.2 looked stuck in loops. Does Gemma's recommended sampling
(temperature 1.0, top_p 0.95, top_k 64) remove them?

**Variants.** `v7-config` (temperature 0.2, top_p 0.95) vs `v7-temp1` (same bundle, temperature 1.0, top_p 0.95,
top_k 64). Both: thinking off, max_output_tokens 8192, 12 min / 100 calls per task.
**Tasks.** requests_7205, rich_3472, fastapi_14463, fastapi_14964, rich_3105, rich_3486 (6 tasks; only
rich_3105 and rich_3472 grade reliably).

| variant | runs | solved/graded | calls | b2b repeat | any repeat | reads | edits | empty sub | secs | tok/turn |
|---|---|---|---|---|---|---|---|---|---|---|
| v7-config (T=0.2) | 6 | 0/2 | 92 | **45%** | **54%** | 4 | 1 | 2 | 355 | 88 |
| v7-temp1 (T=1.0) | 6 | 0/2 | 55 | **1%** | **8%** | 12 | 2 | 9 | 511 | 118 |

Per task, b2b repeat at T=0.2 → T=1.0: fastapi_14463 53% → 0%, fastapi_14964 57% → 0%, requests_7205 19% → 0%,
rich_3105 82% → 0%, rich_3472 61% → 8%, rich_3486 0% → 0%.

**Findings.**
1. **At T=0.2 Gemma degenerates into repeating the previous tool call** (up to 17 identical calls in a row;
   rich_3472 made 99 calls and never opened a file). Loops appeared in 5 of 6 tasks.
2. **T=1.0 removes the loops** (b2b repeat 45% → 1%) and triples file reads. This is a robust behavioural effect
   (5 of 5 looping tasks fixed), not a sampling fluke.
3. **No change in solves on this set** (0/2 both). The loops were hiding a second failure: with loops gone,
   Gemma still fails to turn exploration into a fix on vague tasks (rich_3472: right file read, wrong hypothesis,
   ~70 calls grepping an unrelated class; rich_3105: the issue text is only "Fix #3104", unsolvable from text).
4. Other failure modes seen: `read_file` arguments packed into the path string (requests_7205, 3×); repeated
   `submit_patch` on an empty diff (rich_3472, 9×); empty `<|channel>thought` blocks leaking into replies with
   thinking off.

**Caveat.** 6 tasks, one sample each, only 2 reliably graded. Behavioural metrics are solid; solve rates are not.

## E1b. Spot check (partial, stopped early)

`v7-temp1` on rich_3905: **resolved in 9 calls / 70 s**, the first solve in any configuration. rich_3953: failed
(27 calls). The run (v7-temp1 vs v7-temp1-think on 12 rich tasks) was stopped to free the GPU for E2.

---

## E2. v8-control vs v8-best

**Question.** Does the temperature fix improve the best public bundle (the 0.12 notebook's analyzer + coder)?

**Variants.** `v8-control`: exact rebuild of the 0.12 public notebook (analyzer sub-agent + coder,
temperature 0.2, top_k 40, thinking off, 8192 output tokens, no eval_config). `v8-best`: identical except
temperature 1.0, top_p 0.95, top_k 64, plus a 12 min / 100-call cap. For fairness v8-control is also capped at
12 min here (runner default); its call cap is the harness default.
**Tasks.** The 12 reliably graded rich dev40 tasks. 6 streams, no queueing. Finished 11:45 EDT.

| task | v8-control (T=0.2) | v8-best (T=1.0) |
|---|---|---|
| rich_3052 | fail, 43 calls, 540 s | fail, 31 calls, 263 s |
| rich_3063 | **context overflow**, 49 calls | fail, 12-min cap, 73 calls |
| rich_3105 | **context overflow**, 126 calls | **context overflow**, 66 calls |
| rich_3472 | **context overflow**, 189 calls | fail, 58 calls, 214 s |
| rich_3506 | fail, 23 calls, 183 s | fail, 62 calls, 517 s |
| rich_3676 | **context overflow**, 111 calls | **PASS**, 90 calls, 520 s |
| rich_3777 | **context overflow**, 98 calls | **PASS**, 44 calls, 215 s |
| rich_3905 | **PASS**, 24 calls, 145 s | **PASS**, 25 calls, 153 s |
| rich_3930 | **context overflow**, 174 calls | fail, 94 calls, 489 s |
| rich_3938 | **context overflow**, 95 calls | fail, 95 calls, 634 s |
| rich_3942 | fail, 116 calls, 399 s | fail, 52 calls, 339 s |
| rich_3953 | **context overflow**, 15 calls | **context overflow**, 79 calls |
| **solved** | **1 / 12** | **3 / 12** |
| **context overflows** | **9 / 12** | **2 / 12** |

(Tool-call counts include the analyzer sub-agent's calls.)

**Findings.**
1. **v8-best solved 3 of 12, v8-control 1 of 12.** The one control solve (rich_3905) was also solved by v8-best.
   With one sample per task, +2 tasks is suggestive, not proven (a sign test on the 2 discordant tasks can't
   reach significance). But it fits the mechanism from E1, so I'd act on it.
2. **New failure mode: context overflow kills the session.** vLLM rejects any request where prompt +
   `max_output_tokens` (8192) exceeds 32,768, so once the conversation passes ~24.5k tokens every call fails
   with `ContextWindowExceededError`, the session ends, and **the patch is empty** in every such run. The
   harness's history compaction didn't prevent it (our runner uses the organizers' notebook threshold, 14,336;
   HARNESS_README says the scorer uses 32,768, which would be worse).
3. **The loops cause most overflows.** 9 of 12 control runs overflowed (T=0.2, repeated calls pile up in the
   history) vs 2 of 12 at T=1.0. Three of v8-best's gains (rich_3676, rich_3777) are tasks where the control
   overflowed.
4. v8-best's remaining failures: 2 overflows, 1 time cap, 6 wrong or incomplete fixes (patches produced but
   tests fail). The next lever after sampling is fix quality and context budget.

**Decision.** Submit v8-best tonight (scheduled 20:02 EDT).

**Next experiments.**
- **E3: `max_output_tokens` 8192 → 4096.** With thinking off, replies are ~35–150 tokens on average, so 4096
  probably costs nothing and leaves ~28.6k for the conversation, cutting overflows. Need to check the largest
  single turn first (long `edit_file`/`write_file` payloads).
  *Checked (all 2,850 model turns in E1–E2):* only 6 turns exceeded 1,024 output tokens, 3 exceeded 2,048 and
  **2 exceeded 4,096 (0.07%)**: single oversized edits (4,684 tokens in rich_3506, 4,164 in rich_3105). At 4,096
  those would be cut off and the harness would nudge for smaller edits, so the cap costs almost nothing.
  Peak prompt tokens per run (median / max): v8-control 24,304 / 24,567 (pinned at the ~24.5k ceiling),
  v8-best 19,982 / 29,369, v7-temp1 17,070 / 28,608, v7-config 14,618 / 15,027. (Maxima above 24.5k likely come
  from steps whose usage sums more than one model call; to verify.)
- **E4: thinking on vs off** at T=1.0 (bounded by the same output cap), for the wrong-fix failures.
- **E5: repeat samples** (3 per task) to turn 3/12 vs 1/12 into a rate with an error bar.

---

## E3. max_output_tokens 8192 vs 4096, with a second v8-best sample

**Question.** Does reserving less output (4096) cut context overflows and so raise solves, at no cost?

**Variants.** `v8-best` (8192) vs `v8-best-4k` (identical except `max_output_tokens: 4096`). The v8-best arm is
also a second independent sample of v8-best on the same tasks, which measures run-to-run variation (E5).
**Tasks.** The same 12 reliably graded rich tasks. 6 streams, no queueing. Started 12:05 EDT, finished ~12:50.

| task | v8-best run 1 (E2) | v8-best run 2 (E3) | v8-best-4k (E3) |
|---|---|---|---|
| rich_3052 | fail | fail | fail |
| rich_3063 | fail (time cap) | fail | fail |
| rich_3105 | overflow | fail | fail |
| rich_3472 | fail | fail | fail |
| rich_3506 | fail | **PASS** | fail |
| rich_3676 | **PASS** | **PASS** | fail |
| rich_3777 | **PASS** | fail | **PASS** |
| rich_3905 | **PASS** | **PASS** | **PASS** |
| rich_3930 | fail | fail | fail |
| rich_3938 | fail | fail | fail |
| rich_3942 | fail | fail | fail |
| rich_3953 | overflow | fail | fail |
| **solved** | **3 / 12** | **3 / 12** | **2 / 12** |
| **context overflows** | 2 | 2 | **0** |

E3 behaviour (analyze_evals): v8-best median 80 calls, 379 s, 67 failed edits in total, 1 empty submit;
v8-best-4k median 94 calls, 517 s, 41 failed edits, 0 empty submits.

**Findings.**
1. **v8-best is stable at about 25%: 3/12 twice (6/24).** v8-control was 1/12. The tasks solved vary between
   samples (rich_3905 and rich_3676 twice; rich_3777 and rich_3506 once each), as expected at temperature 1.0.
   Per-task solve probability is what matters, not single outcomes.
2. **4096 eliminated context overflows (2 → 0)** with 2/12 solved vs 3/12. One task is within sampling noise
   (v8-best alone varied by 2 tasks per task-set between samples), so there's no measured solve effect either
   way. Mechanistically it removes a crash mode for ~0.07% truncated turns, and it matters more if the scorer
   compacts at 32,768 as HARNESS_README states.
3. **Failed edits are the next big cost:** 41–67 failed `edit_file` calls per 12 runs (old_string didn't match).
   Each costs a call and a turn.

**Decision.** Submit **v8-best** (user: "whatever is best in experiments"). It has the best measured solve rate
(6/24, replicated); 4096's benefit is mechanistic, not yet measured. Re-test 4096 with more samples later.

---

## A1. Analysis: why edits fail (all runs so far, no new runs)

Across every trace from E1–E3 (7 variants, 60 runs): **455 `edit_file` calls, 203 failed (45%).**

| failure | count | share of failures |
|---|---|---|
| **exact retry of the edit that just failed** | **159** | **78%** |
| old_string not in the file (first attempt) | 38 | 19% |
| literal `\\n` instead of real newlines | 5 | 2% |
| old_string identical to new_string | 1 | <1% |

200 of 203 errors were "old_string not found". So the problem is mostly not bad first attempts but **Gemma
resending the identical failing call**, the same degeneration as the E1 command loops, now on edits. The v8
prompt invites it: *"If an edit fails twice, re-read the exact lines and retry"* allows one blind retry.

**Hypothesis for E6 (prompt):** replace that line with "If `edit_file` fails, never resend it. Re-read the exact
lines (`sed -n 'A,Bp' file`) and copy `old_string` from that output; after a second failure, apply the change with
a short python script via `run_command`." Expected: fewer failed edits and fewer wasted calls. Measure: failed
edits per run, exact-retry count, solves.

---

## E4. Thinking on vs off, with a third v8-best sample

**Question.** With loops fixed, most v8-best failures are wrong or incomplete fixes on vague issues. Does
thinking help Gemma form the right hypothesis, and does it pay for its extra time and context?

**Variants.** `v8-best` (thinking off) vs `v8-best-think` (identical except `include_thoughts: true`, which turns
thinking on; its length is bounded only by `max_output_tokens` 8192). The v8-best arm is a third sample.
**Tasks.** The same 12 reliably graded rich tasks. 6 streams, no queueing. Started ~12:55 EDT.

| | solved | 12-min cap hit | overflows | median calls | median secs | output tok/turn | failed edits |
|---|---|---|---|---|---|---|---|
| v8-best (3rd sample) | **3 / 12** | 2 | 1 | 41 | 346 | 133 | 41 |
| v8-best-think | **1 / 12** | **7** | 2 | 24 | 673 | **451** | 4 |

Per task, v8-best-think solved only rich_3905 (in 401 s vs 101 s without thinking).

**Findings.**
1. **Thinking hurts under the time budget.** 3.4× more output per turn roughly halves the turns that fit in 12
   minutes; 7 of 12 runs hit the cap. It made fewer blind retries (4 failed edits vs 41), so it may reason better
   per step, but it never gets far enough. Kaggle's 12-hour total leaves ~6 min per task on average, so there's no
   room to give thinking more time. **Keep thinking off.**
2. **v8-best replicates a third time: 3/12 again → 9/36 = 25%** (95% CI roughly 12–42%). Per task across the three
   samples: rich_3905 3/3, rich_3676 3/3, rich_3777 2/3, rich_3506 1/3, the other 8 tasks 0/3. The solvable set is
   narrow: gains now need to come from the 8 tasks it never solves.

---

## Summary table (12 reliably graded rich tasks)

| config | samples | solved | rate |
|---|---|---|---|
| v8-control (0.12 notebook) | 1 | 1/12 | 8% |
| **v8-best** (T=1.0) | 4 | **11/48** | **23%** |
| v8-best-4k | 1 | 2/12 | 17% |
| v8-best-think | 1 | 1/12 | 8% |
| v8-best-edit | 1 | 3/11 | 27% |


---

## E6. No blind edit retries (prompt), with a fourth v8-best sample

**Question.** Does forbidding repeated calls and blind edit retries (A1) cut wasted calls and raise solves?

**Variants.** `v8-best` vs `v8-best-edit`: identical except one prompt rule. v8-best says *"If an edit fails
twice, re-read the exact lines and retry with a smaller unique snippet."* v8-best-edit replaces it with *"If
`edit_file` fails, never send the same call again. Re-read the exact lines (`sed -n 'A,Bp' FILE`) and copy
`old_string` from that output. If it fails a second time, make the change with a short python script via
`run_command` that … asserts the old text occurs exactly once …"* plus *"Never repeat a command or tool call you
have already made; its result will not change. If you are stuck, try a different approach."*
**Tasks.** Same 12 rich tasks. Started ~13:50 EDT. Pod shut down at ~14:35 with 23/24 runs done
(v8-best-edit on rich_3930 unfinished).

| | solved | failed edits | b2b repeat | any repeat | median calls | median secs |
|---|---|---|---|---|---|---|
| v8-best (4th sample) | 2 / 12 | 17 | 8% | 22% | 61 | 219 |
| v8-best-edit | 3 / 11 | 23 | 18% | 28% | 38 | 207 |

Solved: v8-best rich_3777, rich_3905; v8-best-edit rich_3506, rich_3777, rich_3905.

**Findings.**
1. **The prompt rule had no measurable effect.** Failed edits (17 vs 23) and repeats (8% vs 18%) did not fall;
   Gemma doesn't follow "never repeat a call". Solves differ by one task (noise). Prompting alone won't fix the
   retry loops; this points to training (a LoRA on traces that never retry a failed edit) or scaffolding.
2. **v8-best's 4th sample: 2/12.** Across four samples: **11/48 = 23%** (95% CI ~12–37%). Per task: rich_3905 4/4,
   rich_3777 3/4, rich_3676 3/4, rich_3506 1/4, the other 8 tasks 0/4.

---

## Status (2026-09-29 ~14:40 EDT)

- Pod terminated (RunPod balance $10.63). Rebuilding one takes ~30 min: `scripts/serve.py`, `scripts/harness_eval.py`,
  `scripts/gold_check.py`, plus pytest from the competition wheels and the pinned versions noted in Setup.
- **Submitted tonight:** v8-best (scheduled 20:02 EDT).
- **Conclusions so far:** temperature 1.0 (loops + overflows) is the one change with a large, replicated effect;
  thinking hurts under the time budget; 4096 output and the anti-retry prompt are neutral on solves.
- **Next ideas, by expected value:** (1) LoRA on filtered traces that model the missing habits (no repeated calls,
  no blind edit retries, reproduce-then-fix on vague issues); (2) more samples of v8-best vs v8-best-4k to settle
  4096; (3) study the 8 never-solved tasks in the viewer to target the next failure mode.

---

## K1. Kaggle result: v8-best = 0.08 (2026-09-30)

Submission 56691354 (v8-best). **Public score 0.08** = 5 of 58 public tasks, our best (v4 0.06, v6 0.05, v7 0.05 ≈ 3–4
tasks). Below the pod-based guess (0.14, range 0.09–0.21). +1–2 tasks is within leaderboard noise (SD ≈ 1.7 tasks),
and it sits in the same band as the 0.12 notebook it's based on (0.12 for its author, 0.08 for a copier).

**Leading hypothesis for the pod/Kaggle gap:** the scorer compacts history at 32,768 tokens (HARNESS_README), not
14,336 as in our runner, so context overflows at 8192 reserved output tokens would be more frequent on Kaggle.
**Next:** E8 = v8-best on the pod with `--compaction-threshold 32768`, counting overflows; if they rise sharply,
submit v8-best-4k.

---

## A2. Analysis: why Gemma repeats tool calls (all 109 runs, no new runs)

7,170 tool calls, **1,520 back-to-back repeats**. What the previous (identical) call returned:

| previous result | share | T=0.2 / T=1.0 | cause |
|---|---|---|---|
| success with useful output | **50%** | 488 / 268 | **pure copy loop** (degeneration) |
| grep found nothing (exit 1, empty output) | **16%** | **232 / 4** | **misreading**: the harness reports "no match" as `status: error`, Gemma retries |
| garbled tool call (corrupted arguments) | 11% | 0 / 168 | resending a malformed call |
| edit failed (old_string not found) | 9% | 52 / 90 | blind edit retry |
| command error with output | 8% | 38 / 78 | resending without fixing |
| success with empty output | 4% | 0 / 56 | retrying for output |
| resubmit / bad path | 3% | – | – |

**Garbled tool calls are a cost of randomness:** 6.4% of all calls at T=1.0 (357 / 5,613) vs 0.8% at T=0.2
(13 / 1,557). Arguments containing quotes/backslashes/backticks (regexes, docstrings) break Gemma's tool-call
delimiters when a wrong token is sampled: keys merge (`new_string` swallows `old_string`), line ranges end up in
`filepath`, fragments become bogus keys (`_escape`).

**Implications.** Three different causes: degeneration (~50%, fixed by randomness or penalties), misreading
empty-grep as failure (~16%, plausibly fixable by prompt), and retrying failures unchanged (~25%, a habit for
training). T=1.0 breaks loops but garbles more calls, so the target is the lowest randomness without loops.

---

## A3. Diagnosis of E2 (v8-control vs v8-best) with `scripts/diagnose_runs.py`

Full output: `experiments/diag_e2.md`; every repeat streak with context in `experiments/diag_e2.repeats.jsonl`.
A *streak* = consecutive identical tool calls. "Gold file" = a file the maintainers' patch touches.

**v8-control (T=0.2): 19 streaks, 565 repeated calls = 52% of all calls.**
- Streaks are huge: `grep -rn "import" rich/cells.py` ×160 (after useful output); a no-match grep ×108; the same
  `read_file` 710–730 ×44; a repro heredoc ×74. **Every streak got the identical result every time** (19/19).
- They start early (11/19 in the first third), no nudge precedes them (0/19), and **Gemma writes no text during
  any streak** (0/19): no reasoning at all, pure continuation.
- 6/19 streaks end only when the **context overflows**; the rest end when Gemma finally reads a file or calls the
  analyzer. 10/19 repeat a call it had already made earlier in the run.
- Progress: in **7/12 runs Gemma saw the gold file but never edited**: loops consume the run before it acts.
  Runs ended: 4 submitted, 8 context overflow. Only 3/12 ran tests.

**v8-best (T=1.0): 13 streaks, 89 repeated calls = 10%.**
- The dominant surviving loop: **a command failing with a SyntaxError, resent unchanged** (2 streaks, 66 calls). In
  rich_3953 a `python3 -c "..."` with a unicode escape broke shell quoting; the identical SyntaxError came back and
  Gemma resent it **66 times** until the context overflowed. Randomness doesn't help when Gemma can't see the fix.
- Other streaks: resending garbled edit calls (3 streaks), resubmitting (×8), a few short copy loops.
- Progress on the 9 failures is spread out: never looked at gold file 1, saw it but no edit 2, edited wrong file 2,
  all gold-file edits failed 2, edited gold file but fix wrong 2. No single bottleneck. 7/12 ran tests.
- "Gold file first seen" is median call 1 in both arms: the analyzer's first report names it. **Localization is not
  the main problem; acting on it is.**

**Implications.** (1) At T=0.2, loops are pure continuation with identical inputs, so anything that makes the
next context differ (sampling, penalties) should break them. (2) At T=1.0, the residual loop is "error → resend
unchanged", i.e. Gemma doesn't read error output. Candidates: a prompt line about quoting (`python3 - <<'EOF'`
heredocs instead of `-c`), or training. (3) Since the analyzer localizes well, improving what happens after it
(edit precision, verification) matters more than search.

---

## E8. Temperature sweep

**Question.** What's the lowest-randomness setting without loops? Measures solves, repeat rate, garbled calls,
failed edits and overflows.
**Variants.** v8-best's prompts, agents and caps (12 min / 100 calls), top_p 0.95, top_k 64, thinking off,
8192 output tokens; only sampling differs: `sweep-t02` (T=0.2), `sweep-t05`, `sweep-t07`, `sweep-t10` (T=1.0 =
v8-best), `sweep-t02-fp03` (T=0.2 + frequency_penalty 0.3), `sweep-split` (analyzer sub-agent at T=1.0 to explore,
coder at T=0.2 to edit: the per-agent version of "high temperature first, then low").
**Setup change:** compaction threshold **32,768** (scorer value per HARNESS_README), up from 14,336.
**Tasks.** The 12 reliably graded rich tasks; one stream per arm (6 streams), tasks one at a time.
Diagnosis: `experiments/diag_e8.md`; metrics: `experiments/metrics_e8.txt`.

| arm | solved | b2b repeat | garbled* | failed edits | overflows | time cap | ran tests | runs that edited the gold file but fix was wrong |
|---|---|---|---|---|---|---|---|---|
| T=0.2 | 2/12 | 21% | 0% | 12 | 3 | 2 | 3/12 | 3 |
| T=0.2 + freq_penalty 0.3 | 3/12 | 35% | 24% | 2 | 4 | 1 | 3/12 | 1 |
| T=0.5 | 2/12 | 29% | 11% | 87 | **5** | 1 | 3/12 | 4 |
| T=0.7 | 1/12 | 23% | 8% | 78 | 4 | 2 | 5/12 | 3 |
| **T=1.0** | **3/12** | **14%** | 1% | 79 | 4 | 0 | **8/12** | **6** |
| split (analyzer 1.0 / coder 0.2) | 2/12 | 41% | 11% | 36 | 2 | 1 | 4/12 | 2 |

\* Garbled calls are dominated by one task, rich_3938: Gemma writes `new_string` with escaped newlines (`\\n`),
the extra backslashes break its tool-call format, the parser drops `old_string`, and Gemma resends the broken
call dozens of times. That's a task-triggered formatting bug plus the retry habit, not a temperature effect.

**Findings.**
1. **Context overflow is the dominant failure under the scorer's compaction setting: 22/72 runs (31%)**, in every
   arm. Every overflowed run peaked at 19,967–24,564 prompt tokens, right under 32,768 − 8,192 = 24,576
   (the fatal request itself isn't logged). **With compaction at 32,768, the prompt can never reach the compaction
   threshold before vLLM rejects it, so compaction effectively never fires on the scorer.**
2. **Overflow costs solvable tasks.** rich_3676 passed 3/4 times for v8-best at compaction 14,336 (E2–E6); at
   32,768 it **overflowed in 5 of 6 arms**, some in only 24 steps. This is the best explanation so far for the
   pod-vs-Kaggle gap (pod 23–25% at 14,336 vs Kaggle 0.08).
3. **Temperature barely moves solves once overflow is in play** (1–3/12 everywhere). Repeats still fall with
   temperature (T=1.0 lowest, 14%).
4. **T=1.0 gets furthest when it survives:** 0 runs never looked at the gold file, 8/12 ran tests, and 6 of its
   failures edited the gold file but the fix was wrong. Its remaining failures are fix quality, not search.
5. **The split idea loops most (41%):** the coder at T=0.2 still searches, and low-temperature search loops.

**Decision.** Keep T=1.0. The next lever is the context budget: E9 tests smaller `max_output_tokens` and
shorter tool output under compaction 32,768.

---

## E9. Context budget under the scorer's compaction (running)

**Question.** Under compaction 32,768 (where compaction never fires before overflow), does reserving fewer output
tokens and producing shorter tool output stop the overflows, and does that recover solves (e.g. rich_3676)?
**Variants.** All v8-best at T=1.0: `v8-best` (8192, a second sample next to E8's T=1.0 arm), `v8-best-4k` (4096),
`v8-best-2k` (2048), `v8-best-2k-lean` (2048 + prompt rule: "every tool result stays in your context… pipe searches
through `| head -40`, use `grep -n` instead of `cat`, read at most 60 lines at a time, run tests as `… -x -q 2>&1 |
tail -20`"). Only 3 of 2,850 earlier replies exceeded 2,048 tokens.
**Setup.** Compaction 32,768; 12 rich tasks; one stream per arm. Started ~10:52 EDT.

*Results pending.*

---

## E10 / E11. v9 candidate and replication (queued, running unattended)

**v9-cand** = v8-best-2k-lean + three prompt rules aimed at diagnosed repeat causes:
1. "`grep` exits with code 1 and prints nothing when there are no matches. That is an answer ('not found'), not a
   failure: never rerun the same search, change it." (A2: ~16% of repeats followed an empty grep.)
2. "For Python with quotes, backslashes or several lines, use a heredoc (`python3 - <<'EOF'` … `EOF`) instead of
   `python3 -c "..."`. If a command fails with a SyntaxError or other error, read the error and fix the command;
   never resend it unchanged." (A3: a SyntaxError from shell quoting was resent 66 times.)
3. "In `edit_file`, `old_string` and `new_string` must contain real line breaks, never the two characters `\n`."
   (E8: rich_3938's garbled-call loops.)

- **E10** (started 10:58 EDT): two independent runs of v9-cand on the 12 rich tasks.
- **E11** (starts automatically when E9 ends): a third v9-cand run, plus second runs of v8-best-2k and v8-best-4k and
  a third of v8-best, so each finalist has ≥2 samples under compaction 32,768.

Decision rule for the next submission: the finalist with the most solves over its samples, with overflows near zero;
ties go to fewer repeats/garbled calls. All under compaction 32,768.

---

## A4. What the failed patches contain (E8–E10, 120 failed runs with saved patches)

| failed run's patch | runs |
|---|---|
| source change in a gold file, still failed (fix wrong) | **38** |
| lost to context overflow | **34** |
| **only scratch files (repro*.py, test_*.py), no source change** | **28** |
| empty patch | 16 |
| source change in the wrong file | 4 |

- **44 runs (37%) never landed a source change.** Gemma spends the run on reproduction scripts (rich_3953 at 4k
  left 37 files `reproduce_zwj_spans_*.py`) while its edits to real code fail. The v8 prompt (from the 0.12
  notebook) has a dedicated "Reproduce: write a minimal script to /tmp/repro.py" step; Gemma writes these into the
  repo instead of /tmp and over-invests in them.
- **Stray scratch files don't cost the score:** 10 of 24 passing runs also had them.
- rich_3930's gold patch regenerates 26 Unicode-table files: effectively unsolvable; treat as a ceiling task.

## A5. fastapi grading on the pod (blocked, parked)

Added the missing test wheels (typing_inspection, inline_snapshot, dirty_equals + executing, asttokens) in a separate
`/workspace/data_plus`. fastapi gold patches still fail collection: `Router.__init__() got an unexpected keyword`.
Cause: the subprocess sandbox creates its venv with system site-packages, so it inherits the harness env's recent
fastapi/starlette (google-adk depends on them), which shadow the old versions each task expects. The organizers say
the real scorer uses a Docker sandbox, which doesn't have this problem; RunPod pods can't run Docker. Fixing it needs
an isolated sandbox interpreter. Parked; rich stays the graded set. (Also: TMPDIR must not be under /workspace, since
the subprocess sandbox rewrites `/workspace` paths.)

## E12. No-repro workflow and analyzer-only thinking (queued after E11)

All v8-best-2k (T=1.0, 2048 output tokens, compaction 32,768), 2 samples each on the 12 rich tasks:
- `v8-best-2k-norepro`: step 3 "Reproduce: write /tmp/repro.py…" replaced by "Read the tests: open the existing
  test file for the code you're changing and read how it exercises that code; the hidden tests will be added there.
  Do not write reproduction scripts." and the repro rerun removed from step 5. Targets the 44 no-source-change runs
  by changing the workflow rather than adding a prohibition.
- `v8-best-2k-athink`: the analyzer sub-agent thinks (include_thoughts: true, 8192 tokens, its own context); the
  coder stays at thinking off / 2048. Targets the 38 wrong-fix runs.
- `v8-best-2k`: control (plus E11's sample → 3 samples total).

## Incident: vLLM stall (2026-09-30 ~16:57 UTC)

vLLM stopped scheduling (0 running, 4 waiting, GPU 0%, KV cache 0%) at ~16:57 UTC during E11, and the pod's sshd
then became unresponsive for ~10 minutes. Two E11 results written during the stall (rich_3930 for v9-cand and
v8-best-2k: "session timeout" with 19–87 calls) were removed as invalid; in-flight tasks were killed without
writing results. vLLM was restarted and E11 resumed at 17:27 UTC (the runner skips finished tasks), followed by
E12 (`/workspace/queue3.sh`). A watchdog (`/workspace/watchdog.sh`) now logs vLLM load every 30 s to
`/workspace/watchdog.log` and restarts vLLM after 3 minutes of stall; any result overlapping a logged STALL
should be excluded.

---

## F1. Scoring-environment bugs reported on the forum (2026-09-29/30), with organizer replies

1. **Tool results are double-JSON-encoded** (thread 744272). swegemma tools return a JSON string; ADK wraps it as
   `{"result": ...}` and LiteLLM serializes again, so a quote in code reaches Gemma as `\\\"` and a newline as `\\n`.
   Gemma copies escapes into `old_string` → "old_string not found": another team traced 39 of 62 failed edits to
   this. Explains much of our 45% edit-failure rate (A1) and rich_3938's escaped-newline loop (E8), and inflates
   context (overflows). Organizer: "Addressing now."
2. **With any LoRA adapter, the 4×L4 KV cache drops to ~7,600 tokens and longer requests hang** (744331):
   vLLM preallocates max_loras 8 × rank 128. Our v6 LoRA submission (0.05) likely stalled on most tasks. Organizer
   will set LoRA parameters from the submission. **Don't ship a LoRA until confirmed fixed.**
3. **Thinking mode drops the thought between tool calls** (744354): ADK sends `reasoning_content`, vLLM 0.19.1 reads
   only `reasoning`. Thinking is crippled; part of why E4 found thinking hurts. Organizer: patch planned.
4. **thinking_budget and seed are not sent** (744566), matching our code reading. Organizer: will patch.
5. **~120 hidden tasks, both splits in one 12 h run** (744258): ~6 min per task including setup.
6. **LoRA adapters were silently wiped** by a vLLM double-registration bug (743508); fixed in wheelhouse v23+.
7. Others also see poor local-vs-LB correlation (744319: "~0.2 local, <0.12 LB").

**Implications.** The scoring environment is changing; re-test thinking and edit-failure rates after the patches
land (watch the wheelhouse version, now v25; ours is the Sep 27 download). A line-number edit helper (E14 idea)
sidesteps the escaping bug regardless of the fix. Hold off on LoRA until the KV-cache fix is confirmed.

---

## E14. Line-number editing via a skill (2026-09-30)

**Why.** 45% of `edit_file` calls fail, mostly "old_string not found": tool results reach Gemma double-escaped (F1),
Gemma copies the escapes into `old_string`, then resends the same failed call. Line numbers aren't affected by
escaping, so editing by line range removes the need to reproduce existing text at all.

**How.** `variants/v8-best-2k-lines` = v8-best-2k with a skill `skills/edit-lines` and **`edit_file` and
`write_file` removed from the coder** (structural, since prompt rules are ignored):
- `scripts/show_lines.py FILE START END`: numbered lines.
- `scripts/edit_lines.py FILE START END NEW_TEXT`: replaces the range (END = START−1 inserts, "" deletes);
  rejects edits that break Python syntax (file unchanged); repairs literal `\n` when the text has no real line
  breaks; prints the edited region with numbers.
Mechanics (read from adk_submission/swegemma/ADK 1.36.1): swegemma passes an `AdkSandboxCodeExecutor`, so ADK's
`run_skill_script` materializes the skill in a temp dir inside the task sandbox and runs the script with
`sys.argv` = the args list; scripts use absolute `/workspace/...` paths. Unit-tested locally (replace, insert,
literal-\n repair, syntax rejection). Prompt step 4 now says to edit with the skill; the retry rule says to read
the error and fix, not resend.

**Test plan.** Smoke test on rich_3905 / rich_3777 (does Gemma load and use the skill?), then a full A/B vs
v8-best-2k on the 12 rich tasks: failed edits, runs with no source change, overflows, solves.

**Smoke tests (rich_3905, rich_3777).**
1. *v1:* the prompt said "show the lines with `show_lines.py`"; Gemma called a tool named `show_lines` /
   `show_lines.py`. **An unknown tool name is fatal in the harness** ("Sandbox execution error: Tool 'show_lines'
   not found"): both sessions died with empty patches. This never happens in normal configs (0 of 305 runs), so any
   skill must spell out the exact call. Fix: prompt and SKILL.md give the literal
   `run_skill_script(skill_name="edit-lines", file_path="scripts/show_lines.py", args=[...])`.
2. *v2:* both tasks **passed** and Gemma called `run_skill_script` correctly, but every skill call crashed: Gemma
   wrapped args in extra quotes/backticks (`int('"734"')`), and the scripts assumed a real `/workspace` (the pod's
   subprocess sandbox lives in a temp dir; Kaggle's Docker sandbox has a real one). **Gemma then fell back to
   `sed -i '734,738c\ ...'` (a line-range replace via run_command) and that edit landed:** line-number editing suits it.
3. *v3 (queued for the full test):* `_common.py` strips quotes/backticks, accepts ints, drops a `/workspace/`
   prefix and resolves paths against `$PWD` (the sandbox's real cwd), then `/workspace`. Unit-tested. Removed a
   stray `__pycache__` (`.pyc` isn't an allowed submission extension).

**Full test:** `/workspace/queue4.sh` runs 2 samples of v8-best-2k-lines on the 12 rich tasks after E12;
E12's two v8-best-2k samples are the comparison.

**Schedule change (14:51 EDT).** Analyzer-thinking arm of E12 stopped after 4 runs (0/4 solved, 3/4 hit the 12-min
cap, 510 output tokens per turn vs ~100): same failure as E4. E12 resumed with no-repro + 2k control only, and E14
(2 samples of v8-best-2k-lines) started in parallel (`/workspace/queue5.sh`).

---

## E12 + E14 results (2026-09-30 ~16:20 EDT; E14 at 17/24 runs)

| arm | solved | 12-min cap | overflows |
|---|---|---|---|
| **v8-best-2k (control, 2 samples)** | **4/24** | 7 | 2 |
| v8-best-2k-norepro (2 samples) | 5/24 | 5 | 5 |
| v8-best-2k-athink (stopped) | 0/6 | 5 | 1 |
| v8-best-2k-lines (E14, 17 of 24 runs) | 1/17 | **10** | 3 |

Per task (P pass, o overflow, t time cap, f fail), control / norepro / lines: rich_3506 ff / **PP** / tt;
rich_3676 oP / to / tt; rich_3777 fP / Po / tt; rich_3905 PP / PP / fP.

**Findings.**
1. **No-repro ties the control** (5 vs 4 of 24, noise), with more overflows (5 vs 2). It's the only arm that solved
   rich_3506 (twice). Not adopted tonight; worth another sample later.
2. **The edit-lines skill loses on time.** 10/17 runs hit the 12-minute cap, including tasks the control solves
   (rich_3777, rich_3676, rich_3506 timed out in both samples). Mechanically it works after the fixes (39/75 edits
   succeeded, all show_lines calls worked); its failures are syntax rejections (Gemma replaces a range that cuts a
   block or mis-indents), which the guard catches, not "not found". But each skill call adds overhead, each edit
   needs a separate show-lines call, and rejections add rounds. Net: slower, fewer solves. Not adopted.
3. Analyzer thinking: same time failure as E4 (stopped early).

**Decision.** Submit **v8-best-2k** tonight (scheduled 00:02 UTC via `runs/submit_tonight.sh`). Pod deleted
automatically after E14 finishes (`runs/shutdown_after_e14.sh`).

**Forum.** Posted thread 744692 ("Compaction can't trigger at token_threshold 32,768, and overflow then discards the
patch"). Check for the organizer's answer on the scorer's threshold before the next session.

**E14 final (all 24 runs):** solved 1/24, 12-min cap 12, overflows 3. Pod deleted 16:32 EDT after the final sync.

---

## F2. Wheelhouse update of 2026-09-30 (20:33 UTC): most reported bugs fixed; scoring broken since

Diffed against our Sep 27 wheels (adk_submission 0.2.11 → 0.2.12, adk_eval_core and swegemma 0.2.7 rebuilt):
- **Escaping fixed:** swegemma tools now return dicts at the ADK boundary (`parse_tool_response`) and use
  `json.dumps(..., ensure_ascii=False)`, so results reach the model JSON-encoded once.
- **Thinking fixed:** a LiteLLM patch copies `reasoning_content` into `reasoning`, so thoughts survive between tool
  calls; `reasoning_config` enables vLLM's `thinking_token_budget` for Gemma 4 (thinking budgets should now work).
- **LoRA KV-cache squeeze fixed:** `max_loras` defaults to 1 and `max_lora_rank` is inferred from the adapter.
- **Scheduler stall:** `scheduler_reserve_full_isl=False` so Gemma 4 requests don't wait forever when the KV cache
  is smaller than max_model_len (likely our 2026-09-30 vLLM freeze).
- **Unchanged:** `agent_runner.py` (overflow still skips the fallback diff) and `config.py` (compaction).

**Scoring broken since the update** (thread 744807, 14+ teams): "Notebook Threw Exception" within minutes, including
byte-identical bundles that scored before. Our v8-best-2k (56740087, 2026-10-01 01:06 UTC) failed this way.

**Consequences.** Findings measured on the old wheels may change: the 45% edit-failure rate (escaping), "thinking
hurts" (dropped thoughts, no budget), LoRA viability. Next: rebuild the pod on the new wheelhouse and rerun v8-best-2k
vs v8-best-2k + thinking with a budget, before more tuning. Don't submit until scoring is fixed.

---

## E15. Baseline and capped thinking on the fixed wheelhouse (running, 2026-10-01)

**Why.** Every earlier experiment ran on the Sep 27 wheels. The Sep 30 update fixed tool-result escaping, dropped
thoughts and the missing thinking budget (F2), so the edit-failure rate and the thinking verdict need re-measuring
before more tuning.
**Setup.** New pod (RTX 6000 Ada) with the **full current wheelhouse** installed the organizers' way (base env, then
every wheelhouse wheel `--no-deps --reinstall`, including their patched vLLM 0.19.1 and adk_submission 0.2.12), pytest
from the competition wheels. Compaction 32,768. 12 rich tasks, 2 samples per arm, 4 streams.
**Arms.**
- `v8-best-2k`: unchanged (T=1.0, top_k 64, 2048 output tokens, thinking off).
- `v8-best-2k-think1k`: thinking on, `thinking_budget: 1024` (now sent as vLLM `thinking_token_budget`).
**Measures.** Failed-edit rate vs the old 45%; whether reasoning tokens per turn stay ≤ ~1,024; time caps and
overflows; solves (vs each other and vs the old v8-best-2k's 4/24).
Automation: `/workspace/queue_e15.sh` starts when vLLM is ready; `runs/sync_e15.sh` syncs every 4 min, rebuilds
`experiments/metrics_e15.txt` / `diag_e15.md`, and deletes the pod after E15 finishes (not if vLLM fails to start).

---

## F3. Forum, 2026-10-01: the scorer's compaction settings, and new harness issues

1. **Scorer compaction = `token_threshold 14,336`, `compaction_interval 5`** (current HARNESS_README, updated
   ~Sep 25; our Sep 24 copy said 32,768 / 15; saved the current one as `data/HARNESS_README.current.md`). Pointed out
   in a reply on our thread 744692 (organizer: "Thanks, I will look into it."). **E8–E14 ran at 32,768, harsher than
   Kaggle**, so they overstated overflow; E2–E6 (14,336, ~12% overflow) are closer to the scorer. The runner now
   defaults to 14,336 / 5, and **E15 was restarted at 10:32 EDT with these values**. The patch-discarded-on-overflow
   bug still applies, just less often.
2. **Compaction drops tool results** (744794): ADK's summary keeps only text parts, and Gemma rarely writes text, so
   after compaction it loses context and tends to repeat its last call. Another post-compaction failure mode.
3. **Test-file reset is a no-op** (744825): `git checkout HEAD -- <files>` aborts on missing pathspecs, so an agent's
   edits to existing test files survive, `test_patch` fails to apply, and a correct source fix scores 0. Our prompt
   already forbids test edits; worth measuring how often Gemma edits test files.
4. **Updated Gemma 4 chat template** (744844): Google's 2026-07-15 template reportedly improves tool calling; the
   competition model ships the June one. Organizer-side only.
5. **src-layout packages** (744889): the subprocess sandbox may import the installed `requests` instead of
   `/workspace/src/requests`, which could explain our requests gold failures. Organizer question pending.
6. **Scoring outage** (744807): still no organizer reply; our failed 56740087 is listed in the thread.

---

## A6. At the scorer's compaction setting, every overflow is in the analyzer sub-agent

All 16 overflows in runs at threshold 14,336 (E2, E3, E4, E6) had the **analyzer** (`code_analyzer`, an
`agent_tool`) making the last request, at a peak prompt of 21–24.6k tokens; the coder's peak was ~3–18k, usually ~4k.
The analyzer runs in its own runner/session, which the App-level compaction config does not reach, and its prompt
("find exactly where it must be fixed… confirm by reading the actual code") drives it to keep reading until the window
is full. Its crash ends the whole task with an empty patch (see the fallback-diff bug). It also has
`search_similar_code`, which can return a huge result (forum 744577).

## E16. Fixing analyzer overflow (queued after E15)

Both at v8-best-2k settings, compaction 14,336 / 5, 2 samples, 12 rich tasks; control = E15's v8-best-2k arm:
- `v8-best-2k-solo`: no analyzer; the coder explores itself (its history is compacted).
- `v8-best-2k-alean`: analyzer keeps only `run_command` + `get_code_neighbors` (no `read_file`,
  `search_similar_code`, `get_code_subgraph`), its prompt asks for `| head -30` / `sed -n` ≤ 40 lines and never whole
  files, and it gets 1024 output tokens (`configs/analyzer_sampling.yaml`).
Measures: overflows (where they happen), solves, time caps, repeats after compaction.

---

## E15 / E16 final (new wheelhouse, scorer compaction 14,336 / 5; E16 deduplicated)

E16 was accidentally launched twice (two runners per sample, sharing results files, double GPU load); duplicates
removed (first result per task kept) and the remainder rerun with one runner per sample.

| arm | solved | overflow | 10/12-min cap |
|---|---|---|---|
| v8-best-2k | 5/24 | 1 | 1 |
| v8-best-2k-alean (lean analyzer) | 5/24 | 1 | 4 |
| v8-best-2k-solo (no analyzer) | 4/24 | **0** | 4 |
| v8-best-2k-think1k (thinking, budget 1024) | **7/24** | 6 | 8 |

**Findings.** (1) Removing or slimming the analyzer cuts overflow to 0–1 (vs 6 for thinking, which grows the
uncompacted analyzer context), but neither raised solves: all 4–5/24, noise. (2) Capped thinking solved most (7/24)
despite 6 overflows and 8 timeouts; it solves different tasks (rich_3063 ×2, rich_3506, rich_3676). Next: thinking on
the overflow-free bases (E17; needs a RunPod top-up, balance $0.99). (3) **12-hour risk:** median task time on the
pod is ~7 min (shared GPU); fewer crashes mean longer tasks.

**Tonight's submission (scheduled 00:02 UTC, `runs/submit_tonight3.sh`, retries 30× every 2 min):**
`v8-alean-10m` = lean analyzer + per-task cap lowered to 10 min. Thinking is not submitted: at ~11 min median per
task it would very likely exceed 12 hours.

## K2. Kaggle timing notebook (2026-10-01, finished in ~50 min)

Private notebook `averywright/gemma-timing-l4x4` on Kaggle's 4× L4 (machine_shape NvidiaL4, internet off, official
wheelhouse + competition model, built from the organizers' install and server cells). Runs one task at a time with the
scorer's compaction (14,336 / 5): v8-alean-10m on rich_3905/3777/3676/3506/3063/3052, then v8-best-2k-think1k on
rich_3905/3676/3063. Records seconds per model call, tokens per call, agent and wall minutes per task (incl. setup and
grading) → `timing_tasks.csv`, `timing_steps.csv`; downloaded to `runs/kaggle-timing/` by `runs/watch_kaggle_timing.sh`.
Purpose: calibrate max_time_minutes against the 12-hour total and decide whether thinking is affordable on L4 speed.

**Results (Kaggle 4× L4, one task at a time, scorer compaction):**

| config | solved | median agent min/task | median s/step | median completion tokens/step | per task |
|---|---|---|---|---|---|
| v8-alean-10m | 2/6 | **3.0** | **1.5** | 36 | 1.1–5.0 min; rich_3676 and rich_3063 used 99 and 98 of 100 tool calls in 4–5 min |
| v8-best-2k-think1k | 1/3 | 6.0 | 4.6 | 126 | 1.8, 6.0, 12.1 min (rich_3063 hit the 12-min cap) |

Install ~105 s, vLLM startup ~581 s (once per run). Whole notebook 2,975 s.

**Findings.** (1) Kaggle runs ~2× faster than our shared pod (3 vs ~7 min/task), so pod time-cap counts were
inflated by GPU sharing. (2) Tonight's v8-alean-10m fits easily: ~120 × ~3.5 min ≈ 7 h. (3) Thinking at budget 1024
is ~6 min/task → ~12 h for 120 tasks: not safe; needs budget ~256 or analyzer-only thinking. (4) **On Kaggle the call
cap, not time, ends looping runs:** two of six tasks burned ~100 tool calls in ~5 minutes.

## LB result 2026-10-02: v8-alean-10m = 0.06
Submission 56761248 (v8-alean-10m, T=1.0, lean analyzer, 10-min cap) scored 0.06 public, below v8-best (0.08) and below the
unmodified public 0.12 notebook (T=0.2, which ~107 teams reproduce deterministically). Fixing analyzer overflow did not
raise the LB score. Every T=1.0 variant we've submitted (0.05, 0.05, 0.08, 0.06) scored below the plain notebook copy.
Pod-local solve rates (~20%) do not predict the hidden-repo LB. Next: re-anchor on the exact 0.12 config and change one
variable per submission.

## Public evidence 2026-10-02 (matterhorn3838/gemma-4-superagent, SHA-verified bundles)
- The top public bundle (romanrozen) is byte-identical to our variants/v8-control (now also saved as variants/top-bundle):
  T=0.2, top_k 40, 8,192 output, thinking off, NO eval_config (scorer defaults 100 calls / 60 min / 300 s).
  It scored 0.12 (romanrozen) and 0.15 (byte-identical copy). 13 teams now sit at 0.15. LB noise is about ±0.03.
- Bundles with hard time caps scored 0.08 twice; long rule lists 0.05-0.06; temperature 1.0 0.06.
- Our v8-best = top bundle + T=1.0/top_k 64 + 12-min eval_config. Both changes are on the "don't" list.
- Forum #745028: calling an undeclared tool ends the task (ValueError) and discards the patch. The task message
  advertises read_file limits and the graph tools whatever agent.yaml declares, so removing a tool carries a risk.

## K3: thinking test on Kaggle 4xL4 (2026-10-02, notebook averywright/gemma-think-test, runs/kaggle-think/)
Same 8 rich tasks, one at a time, official harness, compaction 14,336/5. Base = top bundle (scorer-default limits).
| config | solved | overflows | 5-min timeouts | agent min/task (mean) | sec/step (median) |
|---|---|---|---|---|---|
| base (no thinking, no cap) | 2/8 (3905, 3777) | 2 (3506, 3063) | n/a | 4.3 (max 7.5) | 1.1 |
| think512 + 5-min cap | 2/8 (3905, 3676) | 0 | 6 | 4.5 | 2.9 |
| think1024 + 5-min cap | 2/8 (3905, 3676) | 0 | 3 | 4.4 | 1.75 |
- Thinking solved rich_3676 (base loops to 100 calls) but lost rich_3777 to the 5-min cap (base solved it in 2.4 min).
- Thinking removed the overflows (0 vs 2). Timeouts keep the fallback diff, overflows lose it.
- 8 tasks cannot separate the configs on solves. All three fit the 12-h limit at ~120 tasks.
- Decision: submit tb-think1024 tonight (fewer timeouts than 512, faster steps). LB is the real test.

## LB 2026-10-03: tb-think1024 = 0.10
Top bundle + coder thinking 1024 + 5-min cap scored 0.10 (top bundle: 0.12 / 0.15 public). Within the ±0.03 noise, but not
better. Likely cost: the 5-min cap (K3: 3/8 tasks hit it, and rich_3777 was lost to it). Both K3 base overflows were the
analyzer looping on repro scripts in its uncompacted context (prompt 24k + 8,192 output). Next: tb-solo (no analyzer).

## 2026-10-04: submitted the top bundle unchanged (56823408)
- Why: our three experiments (0.06, 0.10 and earlier) were judged against other teams' scores (0.12 / 0.15). We need our own
  measurement of the anchor under the current harness before more one-change tests mean anything.
- Harness check: max_turns / max_llm_calls cannot stop analyzer loops. AgentTool (google-adk 1.36.1, tools/agent_tool.py)
  runs the sub-agent in a new Runner with a default RunConfig (500 calls) and no compaction, so an analyzer loop ends only
  by overflow (patch lost) or by the task time limit (patch kept).
- tb-solo (analyzer removed) is ready (variants/tb-solo) as the next experiment. The Kaggle test gemma-solo-test was
  still queued after 12 h.

## LB 2026-10-04: top bundle unchanged = 0.12 (our own anchor).

## 2026-10-05: submitted seq-default
Base: public 0.13 bundle (hsiaosuan/gemma-developer-agent-0-13-submission, saved as variants/pub-013, zip sha256 6b3ebb33...):
SequentialAgent explore (read-only, output_key repair_plan) -> coder, thinking off (its thinking_budget values are dead
because include_thoughts:false disables thinking), limits 5 min / 60 calls / 60 s / 100 turns.
One change: eval_config.yaml removed (scorer defaults 100 calls / 60 min / 300 s). Hypothesis: tight limits cost tasks.
Harness note: submit_patch + a final text message ends the run, so any review stage must run before submit_patch.

## 2026-10-05: prepared v10-mech (not yet submitted)
Base: top-bundle (0.12 anchor), same agent tree, sampling and scorer-default limits; only the two prompts change.
Coder: grep exact issue names first and call the analyzer only if 2-3 searches find no clear location; first edit by
call 10; identical repeat calls forbidden; no bare pytest; edit only with edit_file (it returns the diff, so no re-read);
no git checkout/reset/stash/clean; no narration except a NOTES line every ~5 calls (compaction drops old tool output);
one closing line after submit_patch. Analyzer: grep first, graph tools only with exact symbol names, no repeats, ~12 calls.
Harness checks (wheelhouse 0.2.12 / swegemma 0.2.7): the scorer passes no callback registry to compile_submission, so
any callback fails compilation; enable_sandbox_testing defaults True (pytest allowed). Compiles with the real compiler.
Untested on the pod (no pod running). NOTES must share a message with a tool call: a text-only turn triggers a harness nudge (3 in a row end the session).
