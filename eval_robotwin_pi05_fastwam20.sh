#!/usr/bin/env bash
# Persistent per-GPU model servers and FastWAM-style simulator subprocesses.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=(--gpus 0 1 --workers-per-gpu 1 --episodes "${EPISODES:-20}" --video)
[[ -z "${RUN_DIR:-}" ]] || ARGS+=(--output-dir "$RUN_DIR")
[[ -z "${CHECKPOINT:-}" ]] || ARGS+=(--checkpoint "$CHECKPOINT")
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/eval_robotwin_pi05_parallel.py" "${ARGS[@]}" "$@"
