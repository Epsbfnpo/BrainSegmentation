#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULTS_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/texture/results"
SOURCE_SPLIT_JSON="${REPO_ROOT}/dHCP_split.json"
TARGET_SPLIT_JSON="${REPO_ROOT}/PPREMOPREBO_split.json"
PRETRAINED_CHECKPOINT="${REPO_ROOT}/old/checkpoints/source_best.pth"

mkdir -p "${RESULTS_DIR}"

python "${SCRIPT_DIR}/train_texture.py" \
    --source_split_json "${SOURCE_SPLIT_JSON}" \
    --target_split_json "${TARGET_SPLIT_JSON}" \
    --results_dir "${RESULTS_DIR}" \
    --batch_size 2 \
    --val_batch_size 1 \
    --num_workers 4 \
    --epochs 200 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --roi_x 96 --roi_y 96 --roi_z 96 \
    --amp \
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
    --domain_loss_weight 0.7 \
    --embed_align_weight 0.2 \
    --stats_align_weight 0.2 \
    --grl_lambda 1.0
