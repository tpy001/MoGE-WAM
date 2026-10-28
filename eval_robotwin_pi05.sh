#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/pi05_robotwin_rgb/robotwin_rgb_restored_20261019_201221/29999}"
CHECKPOINT="$(realpath "$CHECKPOINT")"
PORT="${PORT:-18765}"
for GPU in "${POLICY_GPU:-0}" "${SIM_GPU:-1}"; do
    case "$GPU" in
        0|1|2|3) ;;
        *) echo "Only physical GPUs 0-3 are allowed; GPUs 4-7 are reserved." >&2; exit 2 ;;
    esac
done
DENOISE_STEPS="${DENOISE_STEPS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/evaluation_results/robotwin_pi05_$(date +%Y%m%d_%H%M%S)}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/data1/user10/robotwin/RoboTwin}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/data1/user10/robotwin/env/bin/python}"
export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export RAYON_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"
SERVER=(taskset -c "${POLICY_CPU_AFFINITY:-48-55}" "$ROOT/.venv/bin/python"
    "$ROOT/scripts/eval_robotwin_pi05.py" --serve --checkpoint "$CHECKPOINT"
    --port "$PORT" --denoise-steps "$DENOISE_STEPS" --seed "${SEED:-42}")
CLIENT=(taskset -c "${SIM_CPU_AFFINITY:-56-63}" "$ROBOTWIN_PYTHON"
    "$ROOT/scripts/eval_robotwin_pi05.py" --checkpoint "$CHECKPOINT"
    --port "$PORT" --denoise-steps "$DENOISE_STEPS" --seed "${SEED:-42}"
    --robotwin-root "$ROBOTWIN_ROOT" --output-dir "$OUTPUT_DIR")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'POLICY_GPU=%s Server:' "${POLICY_GPU:-0}"; printf ' %q' "${SERVER[@]}"; printf '\n'
    printf 'SIM_GPU=%s Client:' "${SIM_GPU:-1}"; printf ' %q' "${CLIENT[@]}" "$@"; printf '\n'
    exit 0
fi
mkdir -p "$OUTPUT_DIR"
# Refuse to attach to an unrelated process already using this port.
"$ROOT/.venv/bin/python" -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "$PORT"
CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}" "${SERVER[@]}" > "$OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
READY=0
for ((attempt=0; attempt<600; attempt++)); do
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -60 "$OUTPUT_DIR/server.log" >&2; exit 1; }
    if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 1
done
[[ "$READY" == 1 ]] || { echo "Policy startup timed out: $OUTPUT_DIR/server.log" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${SIM_GPU:-1}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TMPDIR=/data1/user10/robotwin/tmp
export TORCH_EXTENSIONS_DIR=/data1/user10/robotwin/cache/torch_extensions
"${CLIENT[@]}" "$@" 2>&1 | tee "$OUTPUT_DIR/eval.log"
echo "Results: $OUTPUT_DIR"
