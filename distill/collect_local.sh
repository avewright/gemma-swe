#!/usr/bin/env bash
# Collect teacher traces on a laptop with Docker. collect.py runs here (it calls the teacher
# API); each rollout's commands run in an offline sandbox container (distill/sandbox.py).
# Arguments go to collect.py.
#
#   distill/collect_local.sh --teacher deepseek --samples 1 --limit 1
#   distill/collect_local.sh --teacher deepseek --samples 4 --concurrency 4
#
# DATA_DIR is the contest data (tasks.jsonl, snapshots/, wheels/, sandbox/, docker/);
# ENV_FILE holds the API keys. Traces land in traces/.
set -euo pipefail

GEMMA="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$(cd "${DATA_DIR:-$GEMMA/data}" && pwd)"
ENV_FILE="${ENV_FILE:-$GEMMA/.env}"
VENV="$GEMMA/distill/.venv"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q openai pyyaml numpy pyarrow
fi
docker build --platform linux/amd64 -q -t gemma-distill "$GEMMA/distill" >/dev/null

set -a; . "$ENV_FILE"; set +a
# A sleeping Mac freezes rollouts mid-request (and macOS pauses the per-rollout timeout
# while asleep), so keep it awake for as long as collection runs.
KEEP_AWAKE=(); command -v caffeinate >/dev/null && KEEP_AWAKE=(caffeinate -i)
GEMMA_ROOT="$GEMMA" GEMMA_DATA="$DATA_DIR" exec ${KEEP_AWAKE[@]+"${KEEP_AWAKE[@]}"} "$VENV/bin/python" "$GEMMA/distill/collect.py" --concurrency 4 "$@"
