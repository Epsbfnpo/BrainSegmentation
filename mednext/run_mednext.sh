#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-1000}
RESULTS_DIR="${REPO_ROOT}/results/target_mednext_baseline"
SPLIT_JSON="${REPO_ROOT}/PPREMOPREBO_split.json"

mkdir -p "${RESULTS_DIR}"

echo "🚀 Starting MedNeXt Target-Only Baseline Training"
echo "   Output: ${RESULTS_DIR}"

torchrun --nproc_per_node="${NUM_GPUS}" \
    "${SCRIPT_DIR}/train_mednext.py" \
    --split_json "${SPLIT_JSON}" \
    --results_dir "${RESULTS_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --out_channels 87 \
    --roi_x 128 --roi_y 128 --roi_z 128 \
    --lr 1e-4
