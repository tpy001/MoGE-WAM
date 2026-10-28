#!/usr/bin/env bash
# Official FastWAM 50-task order, one model server and two clients on physical GPU 0.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=(--gpus 0 --workers-per-gpu 2 --tasks all --phases clean random --episodes 20 --video)
[[ -z "${RUN_DIR:-}" ]] || ARGS+=(--output-dir "$RUN_DIR")
[[ -z "${CHECKPOINT:-}" ]] || ARGS+=(--checkpoint "$CHECKPOINT")
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/eval_robotwin_pi05_parallel.py" "${ARGS[@]}" "$@"
