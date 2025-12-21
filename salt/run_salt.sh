#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Inherit settings from env or defaults
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-2000}
RESULTS_DIR=${RESULTS_DIR:-${SCRIPT_DIR}/results/target_salt}
TARGET_SPLIT_JSON=${TARGET_SPLIT_JSON:-${REPO_ROOT}/PPREMOPREBO_split.json}
# Point to your Source model here
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/results_fixed/dHCP_registered_fixed/best_model.pth}
VOLUME_STATS=${VOLUME_STATS:-${REPO_ROOT}/new/priors/target/volume_stats.json}

# SALT Params
SALT_RANK=${SALT_RANK:-128}
SALT_LORA_RANK=${SALT_LORA_RANK:-32}
SALT_REG_WEIGHT=${SALT_REG_WEIGHT:-0.00001}
TIME_LIMIT_SECONDS=${TIME_LIMIT_SECONDS:-85800}

mkdir -p "${RESULTS_DIR}"

export CUDA_LAUNCH_BLOCKING=1

CMD=(
    torchrun --nproc_per_node="${NUM_GPUS}"
    "${SCRIPT_DIR}/train_salt.py"
    --split_json "${TARGET_SPLIT_JSON}"
    --results_dir "${RESULTS_DIR}"
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
    --volume_stats "${VOLUME_STATS}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --salt_rank "${SALT_RANK}"
    --salt_lora_rank "${SALT_LORA_RANK}"
    --salt_reg_weight "${SALT_REG_WEIGHT}"
    --time_limit_seconds "${TIME_LIMIT_SECONDS}"
    --roi_x 128 --roi_y 128 --roi_z 128
    --feature_size 48
    --out_channels 87
    --lr 1e-3
)

echo "Running SALT: ${CMD[*]}"
"${CMD[@]}"
