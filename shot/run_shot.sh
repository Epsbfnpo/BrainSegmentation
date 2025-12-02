#!/bin/bash
set -euo pipefail

# 固定绝对路径，拒绝通过命令查询路径
SCRIPT_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/shot"
REPO_ROOT="/datasets/work/hb-nhmrc-dhcp/work/liu275"

# ------------ Configurable defaults ------------
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-2}          # Increase if memory allows to improve SHOT diversity term
EPOCHS=${EPOCHS:-100}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results/target_shot}
TARGET_SPLIT_JSON=${TARGET_SPLIT_JSON:-${REPO_ROOT}/PPREMOPREBO_split.json}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/results_fixed/dHCP_registered_fixed/best_model.pth}
ROI_X=${ROI_X:-128}
ROI_Y=${ROI_Y:-128}
ROI_Z=${ROI_Z:-128}
FEATURE_SIZE=${FEATURE_SIZE:-48}
LR=${LR:-1e-4}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
DIVERSITY_WEIGHT=${DIVERSITY_WEIGHT:-1.0}
MASTER_PORT=${MASTER_PORT:-29500}

mkdir -p "${RESULTS_DIR}"

CMD=(
    torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}"
    "${SCRIPT_DIR}/train_shot.py"
    --split_json "${TARGET_SPLIT_JSON}"
    --results_dir "${RESULTS_DIR}"
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --roi_x "${ROI_X}" --roi_y "${ROI_Y}" --roi_z "${ROI_Z}"
    --feature_size "${FEATURE_SIZE}"
    --lr "${LR}"
    --eval_interval "${EVAL_INTERVAL}"
    --save_interval "${SAVE_INTERVAL}"
    --diversity_weight "${DIVERSITY_WEIGHT}"
)

RESUME_FILE="${RESULTS_DIR}/latest_model.pt"
if [ -f "${RESUME_FILE}" ]; then
    echo "🔄 Detected latest checkpoint, resuming..."
    CMD+=(--resume "${RESUME_FILE}")
fi

echo "➡️  Launching SHOT (Source Hypothesis Transfer) training"
echo "    Results directory: ${RESULTS_DIR}"

"${CMD[@]}"
