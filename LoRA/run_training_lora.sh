#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------- User configuration ----------
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-2000}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results/target_lora}
TARGET_SPLIT_JSON=${TARGET_SPLIT_JSON:-${REPO_ROOT}/PPREMOPREBO_split.json}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/results_fixed/dHCP_registered_fixed/best_model.pth}
OUT_CHANNELS=${OUT_CHANNELS:-87}
USE_AMP=${USE_AMP:-1}
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
LOG_INTERVAL=${LOG_INTERVAL:-20}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}
CACHE_RATE=${CACHE_RATE:-0.0}
CACHE_WORKERS=${CACHE_WORKERS:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
GRAD_CLIP=${GRAD_CLIP:-12.0}
USE_SLIDING_WINDOW=${USE_SLIDING_WINDOW:-1}
SW_BATCH_SIZE=${SW_BATCH_SIZE:-1}
SW_OVERLAP=${SW_OVERLAP:-0.25}
LOSS_CONFIG=${LOSS_CONFIG:-dice_focal}
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.1}

# ---------- Build command ----------
CMD=(
    torchrun --nproc_per_node="${NUM_GPUS}"
    "${SCRIPT_DIR}/train_lora.py"
    --split_json "${TARGET_SPLIT_JSON}"
    --results_dir "${RESULTS_DIR}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --cache_rate "${CACHE_RATE}"
    --cache_num_workers "${CACHE_WORKERS}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --out_channels "${OUT_CHANNELS}"
    --roi_x "${ROI_X}"
    --roi_y "${ROI_Y}"
    --roi_z "${ROI_Z}"
    --feature_size "${FEATURE_SIZE}"
    --lr_min "${LR_MIN}"
    --lr_warmup_epochs "${LR_WARMUP_EPOCHS}"
    --lr_warmup_start_factor "${LR_WARMUP_START}"
    --grad_accum_steps "${GRAD_ACCUM_STEPS}"
    --ema_decay "${EMA_DECAY}"
    --grad_clip "${GRAD_CLIP}"
    --eval_interval "${EVAL_INTERVAL}"
    --sw_batch_size "${SW_BATCH_SIZE}"
    --sw_overlap "${SW_OVERLAP}"
    --focal_gamma "${FOCAL_GAMMA}"
    --loss_config "${LOSS_CONFIG}"
    --log_interval "${LOG_INTERVAL}"
    --epoch_time_buffer "${EPOCH_TIME_BUFFER}"
    --slurm_time_buffer "${SLURM_TIME_BUFFER}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
)

if [ -n "${PRETRAINED_CHECKPOINT}" ]; then
    CMD+=(--pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
fi

if [ "${USE_AMP}" -eq 0 ]; then
    CMD+=(--no_amp)
fi

if [ "${USE_SLIDING_WINDOW}" -eq 0 ]; then
    CMD+=(--no_eval_sliding_window)
fi

if [ "${EVAL_TTA}" -ne 0 ]; then
    CMD+=(--eval_tta --tta_flip_axes ${TTA_AXES})
fi

if [ "${EVAL_MULTI_SCALE}" -ne 0 ]; then
    CMD+=(--multi_scale_eval --eval_scales ${EVAL_SCALES})
fi

echo "➡️  Launching training with ${NUM_GPUS} GPU(s)"
printf '    Results: %s\n' "${RESULTS_DIR}"

"${CMD[@]}"
