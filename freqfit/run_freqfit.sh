#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_GPUS=${NUM_GPUS:-4}
RESULTS_DIR=${RESULTS_DIR:-"${SCRIPT_DIR}/results/target_freqfit"}
SPLIT_JSON=${SPLIT_JSON:-"${REPO_ROOT}/PPREMOPREBO_split.json"}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-"/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/results_fixed/dHCP_registered_fixed/best_model.pth"}

EPOCHS=${EPOCHS:-2000}
LORA_RANK=${LORA_RANK:-8}

# 🔥 关键修改：降低 ROI 大小以节省显存
ROI_X=${ROI_X:-128}
ROI_Y=${ROI_Y:-128}
ROI_Z=${ROI_Z:-128}

mkdir -p "${RESULTS_DIR}"

CMD=(
    torchrun --nproc_per_node="${NUM_GPUS}"
    "${SCRIPT_DIR}/train_freqfit.py"
    --split_json "${SPLIT_JSON}"
    --results_dir "${RESULTS_DIR}"
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
    --epochs "${EPOCHS}"
    --lora_rank "${LORA_RANK}"

    # 传递修改后的 ROI
    --roi_x "${ROI_X}" --roi_y "${ROI_Y}" --roi_z "${ROI_Z}"

    --apply_spacing --apply_orientation --foreground_only

    # 🔥 确保不禁用 checkpoint (也就是要开启 checkpoint)
    # 不要加 --no_swin_checkpoint
)

RESUME_FILE="${RESULTS_DIR}/latest_model.pt"
if [ -f "${RESUME_FILE}" ]; then
    echo "🔄 Detected latest checkpoint, resuming training..."
    CMD+=(--resume "${RESUME_FILE}")
fi

echo "🚀 Launching FreqFiT (ROI 128x128x128) Training"
"${CMD[@]}"