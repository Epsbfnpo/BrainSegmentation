#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${SCRIPT_DIR}/code_oa"
DEFAULT_SPLIT_JSON="${SCRIPT_DIR}/../PPREMOPREBO_split.json"
DEFAULT_LR_PAIRS="${SCRIPT_DIR}/../new/priors/target/dhcp_lr_swap.json"
DEFAULT_PRETRAINED="/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/results_fixed/dHCP_registered_fixed/best_model.pth"

SPLIT_JSON=${SPLIT_JSON:-${DEFAULT_SPLIT_JSON}}
EXP_NAME=${EXP_NAME:-UGTST_PPREMO}
PATCH_SIZE=${PATCH_SIZE:-"128 128"}
ROI_X=${ROI_X:-128}
ROI_Y=${ROI_Y:-128}
ROI_Z=${ROI_Z:-128}
TARGET_SPACING=${TARGET_SPACING:-"0.8 0.8 0.8"}
BATCH_SIZE=${BATCH_SIZE:-24}
LABELED_BS=${LABELED_BS:-12}
LABELED_SLICES=${LABELED_SLICES:-112}
BASE_LR=${BASE_LR:-1e-4}
MAX_ITER=${MAX_ITER:-20000}
NUM_CLASSES=${NUM_CLASSES:-87}
LATERALITY_PAIRS=${LATERALITY_PAIRS:-${DEFAULT_LR_PAIRS}}
PRETRAINED_PATH=${PRETRAINED_PATH:-${DEFAULT_PRETRAINED}}
EARLY_STOP=${EARLY_STOP:-5000}

if [ ! -f "${SPLIT_JSON}" ]; then
    echo "Split JSON not found: ${SPLIT_JSON}" >&2
    exit 1
fi

cd "${CODE_DIR}"

CMD=(
    python train_finetune.py
    --split_json "${SPLIT_JSON}"
    --exp "${EXP_NAME}"
    --patch_size ${PATCH_SIZE}
    --roi_x "${ROI_X}"
    --roi_y "${ROI_Y}"
    --roi_z "${ROI_Z}"
    --target_spacing ${TARGET_SPACING}
    --batch_size "${BATCH_SIZE}"
    --labeled_bs "${LABELED_BS}"
    --labeled_num "${LABELED_SLICES}"
    --base_lr "${BASE_LR}"
    --max_iterations "${MAX_ITER}"
    --num_classes "${NUM_CLASSES}"
    --early_stop_patient "${EARLY_STOP}"
)

if [ -n "${LATERALITY_PAIRS}" ] && [ -f "${LATERALITY_PAIRS}" ]; then
    CMD+=(--laterality_pairs_json "${LATERALITY_PAIRS}")
fi

if [ -z "${PRETRAINED_PATH}" ]; then
    echo "PRETRAINED_PATH must be specified." >&2
    exit 1
fi

if [ ! -f "${PRETRAINED_PATH}" ]; then
    echo "Pretrained checkpoint not found: ${PRETRAINED_PATH}" >&2
    exit 1
fi

CMD+=(--pretrained_path "${PRETRAINED_PATH}")

printf 'Running UGTST fine-tuning with command:\n  %s\n' "${CMD[*]}"
"${CMD[@]}"
