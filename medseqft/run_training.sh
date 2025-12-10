#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Configuration (Matching GraphAlign) ---
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
EPOCHS=${EPOCHS:-2000}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results/target_medseqft}
TARGET_SPLIT_JSON=${TARGET_SPLIT_JSON:-${REPO_ROOT}/PPREMOPREBO_split.json}
# Ensure this points to your dHCP source model
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/results_fixed/dHCP_registered_fixed/best_model.pth}

# Model Params
ROI_X=${ROI_X:-128}
ROI_Y=${ROI_Y:-128}
ROI_Z=${ROI_Z:-128}
OUT_CHANNELS=${OUT_CHANNELS:-87}
FEATURE_SIZE=${FEATURE_SIZE:-48}

# Training Params
LR=${LR:-5e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}

mkdir -p "${RESULTS_DIR}"

echo "=========================================================="
echo "🚀 MedSeqFT (KD-based FFT) Training Start"
echo "Results Dir: ${RESULTS_DIR}"
echo "Pretrained: ${PRETRAINED_CHECKPOINT}"
echo "=========================================================="

CMD=(
    torchrun --nproc_per_node="${NUM_GPUS}"
    "${SCRIPT_DIR}/train_medseqft.py"
    --split_json "${TARGET_SPLIT_JSON}"
    --results_dir "${RESULTS_DIR}"
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --roi_x "${ROI_X}" --roi_y "${ROI_Y}" --roi_z "${ROI_Z}"
    --out_channels "${OUT_CHANNELS}"
    --feature_size "${FEATURE_SIZE}"
    --num_workers 4
    --lambda_kd 1.0
)

"${CMD[@]}"
