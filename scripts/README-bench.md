# Benchmark scripts (merged 2026-09-25T14:47Z)

Two sessions were editing these. This is the merged version:
- harness-like task prompt, nudges, eval_config budgets, thinking (pod session)
- per-task venv + install_deps.py, baseline commit, harness-like grading,
  tolerant edit_file, gold/empty modes (laptop session)

Check the benchmark itself before trusting agent scores:
  python scripts/bench.py --name gold --mode gold --concurrency 16   # expect ~100%
  python scripts/bench.py --name empty --mode empty --concurrency 16 # expect ~0%
Agent run:
  python scripts/bench.py --name <run> --concurrency 3 [--tasks $(cat scripts/dev40.txt)]

Runs before this merge (smoke, baseline-dev40-nothink, think-v1, check-gold-all)
used a broken environment: fastapi/requests deps did not install. Ignore their scores.
Previous runner: run_one_task.py.pre-merge

Benchmark a variant without touching submission/ (each run freezes a copy in runs/bench/<run>/submission):
  python scripts/bench.py --name <run> --submission variants/<name> --concurrency 3 --tasks $(cat scripts/dev40.txt)

Code-intelligence tools (get_code_neighbors, search_similar_code, get_code_subgraph) are off by default
so older runs stay comparable. The contest harness offers them when the task has graph data (36/129
public tasks). Turn them on with CODE_TOOLS=1:
  CODE_TOOLS=1 python scripts/bench.py --name <run> ...
Failure breakdown for any run:  python3 scripts/analyze.py <run> [<run> ...]
