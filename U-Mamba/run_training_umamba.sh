#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
RESULTS_DIR="${REPO_ROOT}/results/target_umamba"
TARGET_SPLIT="${REPO_ROOT}/PPREMOPREBO_split.json"

mkdir -p "${RESULTS_DIR}"

torchrun --nproc_per_node="${NUM_GPUS}" \
    "${SCRIPT_DIR}/train_umamba.py" \
    --split_json "${TARGET_SPLIT}" \
    --results_dir "${RESULTS_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --roi_x 128 --roi_y 128 --roi_z 128 \
    --out_channels 87
