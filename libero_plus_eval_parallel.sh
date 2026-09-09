#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${1:-127.0.0.1}"
PORT="${2:-8102}"
NUM_TRIALS="${3:-1}"
NUM_WORKERS="${4:-4}"
TASK_SUITE="${5:-${TASK_SUITE:-libero_goal}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/data5/tpy/code/LIBERO-plus}"
LIBERO_CONFIG_PATH="${LIBERO_PLUS_CONFIG_PATH:-$LIBERO_PLUS_ROOT/.libero_config}"
export LIBERO_PLUS_ROOT LIBERO_CONFIG_PATH
export PYTHONPATH="$LIBERO_PLUS_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]]; then
    echo "LIBERO-Plus config not found: $LIBERO_CONFIG_PATH/config.yaml" >&2
    exit 1
fi

SEED="${SEED:-7}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
ENV_RESOLUTION="${ENV_RESOLUTION:-256}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-./libero_plus_eval_parallel_${TIMESTAMP}}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Categories run sequentially. Tasks within each category run in parallel.
CATEGORIES=(
    camera
    robot
    # language
    # light
    # background
    # noise
    # layout
)

echo "============================================================"
echo "SimVLA LIBERO-Plus Parallel Evaluation"
echo "============================================================"
echo "  Server:          ws://${HOST}:${PORT}"
echo "  Task suite:      ${TASK_SUITE}"
echo "  Trials/task:     ${NUM_TRIALS}"
echo "  Workers:         ${NUM_WORKERS}"
echo "  Seed:            ${SEED}"
echo "  Replan steps:    ${REPLAN_STEPS}"
echo "  Image size:      ${IMAGE_SIZE}"
echo "  Env resolution:  ${ENV_RESOLUTION}"
echo "  Output:          ${OUTPUT_DIR}"
echo "============================================================"

TOTAL_START=$(date +%s)

for category in "${CATEGORIES[@]}"; do
    category_output="$OUTPUT_DIR/$TASK_SUITE/$category"
    log_file="$LOG_DIR/${TASK_SUITE}_${category}.log"
    result_json="$category_output/results.json"
    mkdir -p "$category_output"

    echo
    echo "============================================================"
    echo "Evaluating: $TASK_SUITE / $category"
    echo "Workers:    $NUM_WORKERS"
    echo "============================================================"

    CATEGORY_START=$(date +%s)

    client_args=(
        -u eval_libero_plus_parallel_fast.py
        --host "$HOST"
        --port "$PORT"
        --client_type websocket
        --task_suite "$TASK_SUITE"
        --category "$category"
        --num_trials "$NUM_TRIALS"
        --seed "$SEED"
        --replan_steps "$REPLAN_STEPS"
        --num_workers "$NUM_WORKERS"
        --image_size "$IMAGE_SIZE"
        --env_resolution "$ENV_RESOLUTION"
        --video_out "$OUTPUT_DIR/videos"
        --result_json "$result_json"
    )

    # Videos are disabled by default for maximum throughput. Add --save_video
    # to client_args above when sparse replan-frame videos are needed.
    "$PYTHON_BIN" "${client_args[@]}" 2>&1 | tee "$log_file"

    CATEGORY_END=$(date +%s)
    echo "$category finished in $((CATEGORY_END - CATEGORY_START))s"
done

TOTAL_END=$(date +%s)

echo
echo "============================================================"
echo "Evaluation Summary"
echo "============================================================"

for category in "${CATEGORIES[@]}"; do
    result_json="$OUTPUT_DIR/$TASK_SUITE/$category/results.json"
    printf '%-16s ' "$category"

    if [[ -f "$result_json" ]]; then
        "$PYTHON_BIN" - "$result_json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

print(
    f"{data['total_successes']}/{data['total_episodes']} "
    f"({data['success_rate'] * 100:.2f}%) | "
    f"{data['wall_elapsed_sec']:.1f}s | "
    f"{data['episodes_per_sec']:.4f} ep/s"
)
PY
    else
        echo "No result"
    fi
done

echo
echo "============================================================"
echo "Total evaluation time: $((TOTAL_END - TOTAL_START))s"
echo "Logs and results:      ${OUTPUT_DIR}"
echo "============================================================"
