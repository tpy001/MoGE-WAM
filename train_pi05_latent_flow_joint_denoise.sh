#!/usr/bin/env bash
set -euo pipefail

LFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$LFF_ROOT"

PYTHON="$LFF_ROOT/.venv/bin/python"
DATASET_ROOT="${DATASET_ROOT:-/home/user10/.cache/huggingface/lerobot/physical-intelligence/libero_gt_depth_flow_global_v30}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-$LFF_ROOT/checkpoints/pi05_base/params}"

[[ -x "$PYTHON" ]] || { echo "Missing repository Python: $PYTHON" >&2; exit 1; }
[[ -f "$DATASET_ROOT/meta/info.json" ]] || { echo "Missing LeRobot dataset: $DATASET_ROOT" >&2; exit 1; }
[[ -d "$PRETRAIN_CKPT" ]] || { echo "Missing pretrain checkpoint: $PRETRAIN_CKPT" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-PHB}"
export PYTHONPATH="$LFF_ROOT/src:$LFF_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

EXP_NAME="${EXP_NAME:-pi05_latent_flow_joint_denoise_$(date -u +%Y%m%dT%H%M%SZ)}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
CPU_AFFINITY="${CPU_AFFINITY:-48-95}"

COMMAND=(
    taskset -c "$CPU_AFFINITY"
    "$PYTHON" scripts/train.py
    pi05_latent_flow_joint_denoise_libero
    --exp-name="$EXP_NAME"
    --fsdp-devices="$FSDP_DEVICES"
    --num-train-steps="$NUM_TRAIN_STEPS"
    --weight-loader.params-path="$PRETRAIN_CKPT"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'Dataset: %q\n' "$DATASET_ROOT"
    printf 'Python: %q\n' "$PYTHON"
    printf 'Command:'
    printf ' %q' "${COMMAND[@]}" "$@"
    printf '\n'
    exit 0
fi

exec "${COMMAND[@]}" "$@"
