#!/usr/bin/env bash
set -euo pipefail

LFF_ROOT=/data3/tangpeiyuan/latentforceflow_0908
cd "$LFF_ROOT"

export PRETRAIN_CKPT=checkpoints/pi05_base/params
: "${PRETRAIN_CKPT:?Set PRETRAIN_CKPT to the pretrain checkpoint params directory}"
if [[ ! -d "$PRETRAIN_CKPT" ]]; then
    echo "PRETRAIN_CKPT does not exist or is not a directory: $PRETRAIN_CKPT" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export HF_ENDPOINT=https://hf-mirror.com

# TorchCodec's CUDA wheel loads NPP at runtime, but the pip-provided NPP
# directory is not on the dynamic linker's default search path.

EXP_NAME="${EXP_NAME:-pi05_latent_flow_depth_simvla_posttrain_libero40_reprodeced}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-$NUM_TRAIN_STEPS}"


export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"

export NCCL_P2P_LEVEL=PHB

CPU_AFFINITY="${CPU_AFFINITY:-48-95}"

taskset -c "$CPU_AFFINITY"  uv run scripts/train.py \
    pi05_latent_flow_depth_simvla_posttrain_libero40 \
    --exp-name="$EXP_NAME" \
    --fsdp-devices="$FSDP_DEVICES" \
    --num-train-steps="$NUM_TRAIN_STEPS" \
    --lr-schedule.decay-steps="$LR_DECAY_STEPS" \
    --weight-loader.params-path="$PRETRAIN_CKPT" \
    --overwrite \
    "$@"
