#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
EPOCHS=${EPOCHS:-50}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results/target_tent}
TARGET_SPLIT_JSON=${TARGET_SPLIT_JSON:-${REPO_ROOT}/PPREMOPREBO_split.json}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/results_fixed/dHCP_registered_fixed/best_model.pth}

ROI_X=${ROI_X:-128}
ROI_Y=${ROI_Y:-128}
ROI_Z=${ROI_Z:-128}
FEATURE_SIZE=${FEATURE_SIZE:-48}
LR=${LR:-1e-4}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}
SAVE_INTERVAL=${SAVE_INTERVAL:-5}

mkdir -p "${RESULTS_DIR}"

CMD=(
    torchrun --nproc_per_node="${NUM_GPUS}"
    "${SCRIPT_DIR}/train_tent.py"
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
    --no_swin_checkpoint
)

RESUME_FILE="${RESULTS_DIR}/latest_model.pt"
if [ -f "${RESUME_FILE}" ]; then
    echo "🔄 Detected latest checkpoint, resuming..."
    CMD+=(--resume "${RESUME_FILE}")
fi

echo "➡️  Launching TENT (Entropy Minimization) Training..."
echo "    Results: ${RESULTS_DIR}"

"${CMD[@]}"
