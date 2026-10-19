#!/usr/bin/env bash
set -euo pipefail

LFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$LFF_ROOT"

PYTHON="$LFF_ROOT/.venv/bin/python"
DATASET_ROOT="${DATASET_ROOT:-/home/tpy/.cache/huggingface/lerobot/physical-intelligence/libero_gt_depth_flow_global_v30}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-$LFF_ROOT/checkpoints/pi05_base/params}"

[[ -x "$PYTHON" ]] || { echo "Missing repository Python: $PYTHON" >&2; exit 1; }
[[ -f "$DATASET_ROOT/meta/info.json" ]] || { echo "Missing dataset: $DATASET_ROOT" >&2; exit 1; }
[[ -d "$PRETRAIN_CKPT" ]] || { echo "Missing pretrain checkpoint: $PRETRAIN_CKPT" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
# Avoid the first-step cuDNN execution failure observed on this 5090 runtime.
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$LFF_ROOT/src:$LFF_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
# Reuse the installed NPP and FFmpeg 7 libraries for TorchCodec.
NPP_LIB="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/npp/lib")')"
FFMPEG_LIB="${FFMPEG_LIB:-/home/tpy/miniconda3/envs/latentforceflow/lib}"
[[ -f "$FFMPEG_LIB/libavutil.so.59" ]] || { echo "Missing FFmpeg 7 libraries: $FFMPEG_LIB" >&2; exit 1; }
export LD_LIBRARY_PATH="$NPP_LIB:$FFMPEG_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

EXP_NAME="${EXP_NAME:-pi05_latent_flow_joint_denoise_multinoise_5090_$(date +%Y%m%d_%H%M%S)}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"

# The interrupted A100 run has no committed checkpoint; initialize from pi05_base.
COMMAND=(
    "$PYTHON" scripts/train.py
    pi05_latent_flow_joint_denoise_multinoise_libero
    --exp-name="$EXP_NAME"
    --fsdp-devices="$FSDP_DEVICES"
    --batch-size=32
    --num-train-steps="$NUM_TRAIN_STEPS"
    --data.dataset-root="$DATASET_ROOT"
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
