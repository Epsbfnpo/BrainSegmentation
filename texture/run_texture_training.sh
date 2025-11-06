#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ===== User configuration =====
RESULTS_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/texture/results"
SOURCE_SPLIT_JSON="${REPO_ROOT}/dHCP_split.json"
TARGET_SPLIT_JSON="${REPO_ROOT}/PPREMOPREBO_split.json"
PRETRAINED_CHECKPOINT="${REPO_ROOT}/old/checkpoints/source_best.pth"

NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
EPOCHS=${EPOCHS:-200}
JOB_TIME_LIMIT_MINUTES=${JOB_TIME_LIMIT_MINUTES:-115}
TIME_BUFFER_MINUTES=${TIME_BUFFER_MINUTES:-5}
DIST_TIMEOUT_MINUTES=${DIST_TIMEOUT_MINUTES:-180}

mkdir -p "${RESULTS_DIR}"

export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}

CMD=(
    "${SCRIPT_DIR}/train_texture.py"
    --source_split_json "${SOURCE_SPLIT_JSON}"
    --target_split_json "${TARGET_SPLIT_JSON}"
    --results_dir "${RESULTS_DIR}"
    --batch_size "${BATCH_SIZE}"
    --val_batch_size "${VAL_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --epochs "${EPOCHS}"
    --lr 5e-5
    --weight_decay 1e-4
    --roi_x 96 --roi_y 96 --roi_z 96
    --amp
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
    --domain_loss_weight 0.7
    --embed_align_weight 0.2
    --stats_align_weight 0.2
    --grl_lambda 1.0
    --auto_resume
    --job_time_limit "${JOB_TIME_LIMIT_MINUTES}"
    --time_buffer_minutes "${TIME_BUFFER_MINUTES}"
    --dist_timeout "${DIST_TIMEOUT_MINUTES}"
)

if [[ ${NUM_GPUS} -gt 1 ]]; then
    torchrun --nproc_per_node="${NUM_GPUS}" "${CMD[@]}"
else
    python "${CMD[@]}"
fi
