#!/usr/bin/env bash
set -euo pipefail

# Compute future optical-flow videos for physical-intelligence/libero.
# Only compute flow for the external RGB camera: image.

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
REPO_ID="${REPO_ID:-physical-intelligence/libero_depth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/libero_depth_flow_step10_wrist}"
FUTURE_STEP="${FUTURE_STEP:-10}"
DEVICE="${DEVICE:-cuda}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
CPU_THREADS="${CPU_THREADS:-12}"
TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"
OPENCV_THREADS="${OPENCV_THREADS:-2}"
FLOW_BATCH_SIZE="${FLOW_BATCH_SIZE:-16}"

OMP_NUM_THREADS="${CPU_THREADS}" \
MKL_NUM_THREADS="${CPU_THREADS}" \
OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${PYTHON_BIN}" scripts/compute_lerobot_future_flow_video.py \
  --repo-id "${REPO_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --image-keys wrist_image \
  --stream-names wrist_image \
  --future-step "${FUTURE_STEP}" \
  --device "${DEVICE}" \
  --video-backend "${VIDEO_BACKEND}" \
  --torch-threads "${CPU_THREADS}" \
  --torch-interop-threads "${TORCH_INTEROP_THREADS}" \
  --opencv-threads "${OPENCV_THREADS}" \
  --batch-size "${FLOW_BATCH_SIZE}" \
  --overwrite \
  "$@"
