#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Configuration ---
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS_STAGE1=${EPOCHS_STAGE1:-2000}
EPOCHS_STAGE2=${EPOCHS_STAGE2:-500}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results/target_medseqft}
TARGET_SPLIT_JSON=${TARGET_SPLIT_JSON:-${REPO_ROOT}/PPREMOPREBO_split.json}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/results_fixed/dHCP_registered_fixed/best_model.pth}

# Model Params
ROI_X=${ROI_X:-128}
ROI_Y=${ROI_Y:-128}
ROI_Z=${ROI_Z:-128}
OUT_CHANNELS=${OUT_CHANNELS:-87}
FEATURE_SIZE=${FEATURE_SIZE:-48}

# Pipeline artifacts
MDS_BUFFER_JSON="${RESULTS_DIR}/buffer_selected.json"
STAGE1_MODEL="${RESULTS_DIR}/stage1_final.pt"
STAGE2_LORA_DIR="${RESULTS_DIR}/lora_final"
FINAL_MERGED_MODEL="${RESULTS_DIR}/final_model.pt"

mkdir -p "${RESULTS_DIR}"

echo "=========================================================="
echo "🚀 MedSeqFT Automated Pipeline"
echo "Results Dir: ${RESULTS_DIR}"
echo "=========================================================="

# ---------------------------------------------------------
# Step 0: MDS Selection
# ---------------------------------------------------------
if [ ! -f "$MDS_BUFFER_JSON" ]; then
    echo "Processing Step 0: MDS Selection..."
    python "${SCRIPT_DIR}/calculate_mds.py" \
        --split_json "${TARGET_SPLIT_JSON}" \
        --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
        --output_json "${MDS_BUFFER_JSON}" \
        --top_k 50 \
        --roi_x "${ROI_X}" --roi_y "${ROI_Y}" --roi_z "${ROI_Z}" \
        --out_channels "${OUT_CHANNELS}" --feature_size "${FEATURE_SIZE}" \
        --batch_size 1 --num_workers 4 \
        --foreground_only
    echo "✅ Step 0: MDS Done."
else
    echo "⏭️ Step 0: MDS Buffer found, skipping."
fi

# ---------------------------------------------------------
# Step 1: KD-based FFT
# ---------------------------------------------------------
if [ ! -f "$STAGE1_MODEL" ]; then
    echo "Processing Step 1: KD-based Full Fine-Tuning..."
    torchrun --nproc_per_node="${NUM_GPUS}" \
        "${SCRIPT_DIR}/train_medseqft.py" \
        --split_json "${TARGET_SPLIT_JSON}" \
        --buffer_json "${MDS_BUFFER_JSON}" \
        --results_dir "${RESULTS_DIR}/stage1" \
        --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
        --epochs "${EPOCHS_STAGE1}" \
        --batch_size "${BATCH_SIZE}" \
        --lr 5e-5 --weight_decay 1e-5 \
        --roi_x "${ROI_X}" --roi_y "${ROI_Y}" --roi_z "${ROI_Z}" \
        --out_channels "${OUT_CHANNELS}" --feature_size "${FEATURE_SIZE}" \
        --lambda_kd 1.0 \
        --foreground_only

    mv "${RESULTS_DIR}/stage1/final_model.pt" "$STAGE1_MODEL"
    echo "✅ Step 1: FFT Done."
else
    echo "⏭️ Step 1: FFT Model found, skipping."
fi

# ---------------------------------------------------------
# Step 2: LoRA Refinement
# ---------------------------------------------------------
if [ ! -d "$STAGE2_LORA_DIR" ]; then
    echo "Processing Step 2: LoRA-based Knowledge Distillation..."
    python "${SCRIPT_DIR}/train_lora_refine.py" \
        --split_json "${TARGET_SPLIT_JSON}" \
        --results_dir "${RESULTS_DIR}/stage2" \
        --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
        --stage1_checkpoint "$STAGE1_MODEL" \
        --lora_rank 4 \
        --epochs "${EPOCHS_STAGE2}" \
        --lr 1e-4 \
        --roi_x "${ROI_X}" --roi_y "${ROI_Y}" --roi_z "${ROI_Z}" \
        --out_channels "${OUT_CHANNELS}" --feature_size "${FEATURE_SIZE}" \
        --foreground_only

    mv "${RESULTS_DIR}/stage2/lora_final" "$STAGE2_LORA_DIR"
    echo "✅ Step 2: LoRA Done."
else
    echo "⏭️ Step 2: LoRA Weights found, skipping."
fi

# ---------------------------------------------------------
# Step 3: Merge & Reparameterization
# ---------------------------------------------------------
if [ ! -f "$FINAL_MERGED_MODEL" ]; then
    echo "Processing Step 3: Merging Model..."
    python "${SCRIPT_DIR}/merge_model.py" \
        --base_model_path "${PRETRAINED_CHECKPOINT}" \
        --lora_path "$STAGE2_LORA_DIR" \
        --output_path "$FINAL_MERGED_MODEL" \
        --roi_x "${ROI_X}" --roi_y "${ROI_Y}" --roi_z "${ROI_Z}" \
        --out_channels "${OUT_CHANNELS}" --feature_size "${FEATURE_SIZE}"

    echo "🎉 All Steps Completed. Final model is at $FINAL_MERGED_MODEL"
else
    echo "✅ Pipeline fully finished."
fi
