#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-2000}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results/target_l2sp}
TARGET_SPLIT_JSON=${TARGET_SPLIT_JSON:-${REPO_ROOT}/PPREMOPREBO_split.json}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/results_fixed/dHCP_registered_fixed/best_model.pth}
OUT_CHANNELS=${OUT_CHANNELS:-87}
USE_AMP=${USE_AMP:-1}
LAMBDA_L2SP=${LAMBDA_L2SP:-0.01}

ROI_X=${ROI_X:-128}
ROI_Y=${ROI_Y:-128}
ROI_Z=${ROI_Z:-128}
FEATURE_SIZE=${FEATURE_SIZE:-48}
LR=${LR:-5e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
LR_MIN=${LR_MIN:-5e-8}
LR_WARMUP_EPOCHS=${LR_WARMUP_EPOCHS:-120}
LR_WARMUP_START=${LR_WARMUP_START:-0.1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
EMA_DECAY=${EMA_DECAY:-0.0}
EVAL_TTA=${EVAL_TTA:-0}
TTA_AXES=${TTA_AXES:-"0"}
EVAL_MULTI_SCALE=${EVAL_MULTI_SCALE:-0}
EVAL_SCALES=${EVAL_SCALES:-"1.0"}
EPOCH_TIME_BUFFER=${EPOCH_TIME_BUFFER:-600}
SLURM_TIME_BUFFER=${SLURM_TIME_BUFFER:-300}
CLASS_MAP_JSON=${CLASS_MAP_JSON:-}

required_files=("${TARGET_SPLIT_JSON}")
for required in "${required_files[@]}"; do
    if [ ! -f "$required" ]; then
        echo "Missing required file: $required" >&2
        exit 1
    fi
done

mkdir -p "${RESULTS_DIR}"

RESUME_FILE="${RESULTS_DIR}/resume_from.txt"
RESUME_FROM=""
if [ -f "${RESUME_FILE}" ]; then
    RESUME_CANDIDATE="$(cat "${RESUME_FILE}")"
    if [ -n "${RESUME_CANDIDATE}" ] && [ -f "${RESUME_CANDIDATE}" ]; then
        RESUME_FROM="${RESUME_CANDIDATE}"
        echo "🔄 Resume checkpoint detected: ${RESUME_FROM}"
    else
        echo "⚠️  resume_from.txt present but candidate missing (${RESUME_CANDIDATE}); ignoring."
    fi
fi

CMD=(
    torchrun --nproc_per_node="${NUM_GPUS}"
    "${SCRIPT_DIR}/train_l2sp.py"
    --split_json "${TARGET_SPLIT_JSON}"
    --results_dir "${RESULTS_DIR}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --out_channels "${OUT_CHANNELS}"
    --roi_x "${ROI_X}" --roi_y "${ROI_Y}" --roi_z "${ROI_Z}"
    --feature_size "${FEATURE_SIZE}"
    --lr "${LR}" --weight_decay "${WEIGHT_DECAY}"
    --lr_min "${LR_MIN}"
    --lr_warmup_epochs "${LR_WARMUP_EPOCHS}"
    --lr_warmup_start_factor "${LR_WARMUP_START}"
    --grad_accum_steps "${GRAD_ACCUM_STEPS}"
    --lambda_l2sp "${LAMBDA_L2SP}"
    --epoch_time_buffer "${EPOCH_TIME_BUFFER}"
    --slurm_time_buffer "${SLURM_TIME_BUFFER}"
)

if [ -n "${RESUME_FROM}" ]; then
    CMD+=(--resume "${RESUME_FROM}")
elif [ -n "${PRETRAINED_CHECKPOINT}" ]; then
    if [ ! -f "${PRETRAINED_CHECKPOINT}" ]; then
        echo "⚠️  Pretrained checkpoint ${PRETRAINED_CHECKPOINT} not found; proceeding without it."
    else
        CMD+=(--pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
    fi
fi

if [ -n "${CLASS_MAP_JSON}" ]; then
    CMD+=(--class_map_json "${CLASS_MAP_JSON}")
fi

if python - <<PY >/dev/null 2>&1
import sys
try:
    v = float("${EMA_DECAY:-0}")
    sys.exit(0 if v > 0 else 1)
except Exception:
    sys.exit(1)
PY
then
    CMD+=(--ema_decay "${EMA_DECAY}")
fi

if [ "${EVAL_TTA}" -ne 0 ]; then
    CMD+=(--eval_tta --tta_flip_axes ${TTA_AXES})
fi

if [ "${EVAL_MULTI_SCALE}" -ne 0 ]; then
    CMD+=(--multi_scale_eval --eval_scales ${EVAL_SCALES})
fi

if [ "${USE_AMP}" -eq 0 ]; then
    CMD+=(--no_amp)
fi

printf 'Running command:\n  %s\n' "${CMD[*]}"

"${CMD[@]}"
