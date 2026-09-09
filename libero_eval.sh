#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Arguments
# =============================================================================

HOST="${1:-127.0.1.1}"
PORT="${2:-8000}"
NUM_TRIALS="${3:-50}"

# 第 4 个参数：并行 LIBERO 环境数量
NUM_WORKERS="${4:-4}"

PYTHON_BIN="${PYTHON_BIN:-python}"

SEED=7
REPLAN_STEPS=10
IMAGE_SIZE=256
ENV_RESOLUTION=256

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-./paper_eval_parallel_${TIMESTAMP}}"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

# =============================================================================
# Evaluation suites
# =============================================================================

SUITES=(
    # libero_spatial
    # libero_object
    # libero_goal
    libero_10
)

echo "============================================================"
echo "SimVLA LIBERO Parallel Paper Evaluation"
echo "============================================================"
echo "  Server:          ws://${HOST}:${PORT}"
echo "  Trials/task:     ${NUM_TRIALS}"
echo "  Workers:         ${NUM_WORKERS}"
echo "  Seed:            ${SEED}"
echo "  Replan steps:    ${REPLAN_STEPS}"
echo "  Image size:      ${IMAGE_SIZE}"
echo "  Env resolution:  ${ENV_RESOLUTION}"
echo "  Output:          ${OUTPUT_DIR}"
echo "============================================================"
echo

# =============================================================================
# Run suites
#
# Suites 串行运行；
# 每个 suite 内部启动 NUM_WORKERS 个独立 LIBERO 环境。
# =============================================================================

TOTAL_START=$(date +%s)

for suite in "${SUITES[@]}"; do

    suite_output="$OUTPUT_DIR/$suite"
    log_file="$LOG_DIR/$suite.log"
    result_json="$suite_output/results.json"

    mkdir -p "$suite_output"

    echo
    echo "============================================================"
    echo "Evaluating: $suite"
    echo "Workers:    $NUM_WORKERS"
    echo "============================================================"

    SUITE_START=$(date +%s)

    client_args=(
        -u eval_libero_parallel_fast.py

        --host "$HOST"
        --port "$PORT"
        --client_type websocket

        --task_suite "$suite"
        --num_trials "$NUM_TRIALS"

        --seed "$SEED"
        --replan_steps "$REPLAN_STEPS"

        --num_workers "$NUM_WORKERS"

        --image_size "$IMAGE_SIZE"
        --env_resolution "$ENV_RESOLUTION"

        --video_out "$suite_output/videos"
        --result_json "$result_json"
    )

    # 默认不保存视频：
    # 并行评测 + replan-only rendering，速度最快。
    "$PYTHON_BIN" "${client_args[@]}" 2>&1 | tee "$log_file"

    SUITE_END=$(date +%s)
    SUITE_TIME=$((SUITE_END - SUITE_START))

    echo
    echo "$suite finished in ${SUITE_TIME}s"

done

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

# =============================================================================
# Summary
# =============================================================================

echo
echo "============================================================"
echo "Evaluation Summary"
echo "============================================================"

for suite in "${SUITES[@]}"; do

    result_json="$OUTPUT_DIR/$suite/results.json"

    printf '%-16s ' "$suite"

    if [[ -f "$result_json" ]]; then

        "$PYTHON_BIN" - "$result_json" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

success = data["total_successes"]
total = data["total_episodes"]
rate = data["success_rate"] * 100
wall = data["wall_elapsed_sec"]
throughput = data["episodes_per_sec"]

print(
    f"{success}/{total} "
    f"({rate:.2f}%) | "
    f"{wall:.1f}s | "
    f"{throughput:.4f} ep/s"
)
PY

    else
        echo "No result"
    fi

done

echo
echo "============================================================"
echo "Total evaluation time: ${TOTAL_TIME}s"
echo "Logs and results:      ${OUTPUT_DIR}"
echo "============================================================"