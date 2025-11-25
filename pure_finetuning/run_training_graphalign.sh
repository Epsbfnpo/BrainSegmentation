#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-2000}
RESULTS_DIR=${RESULTS_DIR:-${SCRIPT_DIR}/results/pure_finetuning}
TARGET_SPLIT_JSON=${TARGET_SPLIT_JSON:-${REPO_ROOT}/PPREMOPREBO_split.json}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/results_fixed/dHCP_registered_fixed/best_model.pth}

ROI_X=${ROI_X:-128}
ROI_Y=${ROI_Y:-128}
ROI_Z=${ROI_Z:-128}
FEATURE_SIZE=${FEATURE_SIZE:-48}
LR=${LR:-5e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
EMA_DECAY=${EMA_DECAY:-0.0}
USE_AMP=${USE_AMP:-1}

mkdir -p "${RESULTS_DIR}"

echo "=========================================================="
echo "Running pure fine-tuning baseline"
echo "Split JSON: ${TARGET_SPLIT_JSON}"
echo "Results   : ${RESULTS_DIR}"
echo "=========================================================="

torchrun --nproc_per_node="${NUM_GPUS}" \
    "${SCRIPT_DIR}/train_graphalign_age.py" \
    --split_json "${TARGET_SPLIT_JSON}" \
    --results_dir "${RESULTS_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --out_channels 87 \
    --roi_x "${ROI_X}" --roi_y "${ROI_Y}" --roi_z "${ROI_Z}" \
    --feature_size "${FEATURE_SIZE}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
    --loss_config "dice_ce" \
    --no_swin_checkpoint \
    $( [ "${USE_AMP}" -eq 0 ] && echo "--no_amp" || echo "--use_amp" ) \
    --log_interval 20 \
    --eval_interval 50
