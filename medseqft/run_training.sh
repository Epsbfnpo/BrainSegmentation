#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"  # 假设你的代码结构没变

# --- Configuration ---
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-2} # 3D SwinUNETR 128x128 显存占用大，2是安全值
EPOCHS_STAGE1=${EPOCHS_STAGE1:-300} # AMOS 规模较大，可以适当跑多点，或者按需调整
EPOCHS_STAGE2=${EPOCHS_STAGE2:-100}

# 结果保存路径
RESULTS_DIR=${RESULTS_DIR:-/datasets/work/hb-nhmrc-dhcp/work/liu275/medseqft/results/target_totalseg}

# [关键修改] 目标域 Split JSON (刚才生成的)
TARGET_SPLIT_JSON=${TARGET_SPLIT_JSON:-/datasets/work/hb-nhmrc-dhcp/work/liu275/TotalSegmentator_14cls_final_split.json}

# [关键修改] AMOS 预训练模型 (Source Model)
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/results_fixed/amos_supervised_finetune/AMOS_CT_Finetune_Fixed/best_model.pth}

# Model Params
ROI_X=${ROI_X:-128}
ROI_Y=${ROI_Y:-128}
ROI_Z=${ROI_Z:-128}
OUT_CHANNELS=${OUT_CHANNELS:-15}  # 0背景 + 14器官 = 15类
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
    if [ ! -f "$MDS_BUFFER_JSON" ]; then
        echo "⏳ Step 0 interrupted (Timeout). Requesting Resubmit..."
        exit 0
    fi
    echo "✅ Step 0: MDS Done."
else
    echo "⏭️ Step 0: MDS Buffer found, skipping."
fi

# ---------------------------------------------------------
# Step 1: KD-based FFT
# ---------------------------------------------------------
if [ ! -f "$STAGE1_MODEL" ]; then
    echo "Processing Step 1: KD-based Full Fine-Tuning..."
    TEMP_STAGE1_OUTPUT="${RESULTS_DIR}/stage1/final_model.pt"
    torchrun --nproc_per_node="${NUM_GPUS}" \
        "${SCRIPT_DIR}/train_medseqft.py" \
        --split_json "${TARGET_SPLIT_JSON}" \
        --buffer_json "${MDS_BUFFER_JSON}" \
        --results_dir "${RESULTS_DIR}/stage1" \
        --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
        --epochs "${EPOCHS_STAGE1}" \
        --batch_size "${BATCH_SIZE}" \
        --lr 1e-3 --weight_decay 1e-5 \
        --roi_x "${ROI_X}" --roi_y "${ROI_Y}" --roi_z "${ROI_Z}" \
        --out_channels "${OUT_CHANNELS}" --feature_size "${FEATURE_SIZE}" \
        --lambda_kd 0.0 \
        --foreground_only \
        --cache_rate 0.0

    if [ ! -f "$TEMP_STAGE1_OUTPUT" ]; then
        echo "⏳ Step 1 incomplete (Timeout/Interrupted). Stopping pipeline here."
        exit 0
    fi

    mv "$TEMP_STAGE1_OUTPUT" "$STAGE1_MODEL"
    echo "✅ Step 1: FFT Done."
else
    echo "⏭️ Step 1: FFT Model found, skipping."
fi

# ---------------------------------------------------------
# Step 2: LoRA Refinement
# ---------------------------------------------------------
if [ ! -d "$STAGE2_LORA_DIR" ]; then
    echo "Processing Step 2: LoRA-based Knowledge Distillation..."
    TEMP_LORA_OUTPUT="${RESULTS_DIR}/stage2/lora_final"
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

    if [ ! -d "$TEMP_LORA_OUTPUT" ]; then
        echo "⏳ Step 2 incomplete (Timeout/Interrupted). Stopping pipeline here."
        exit 0
    fi

    mv "$TEMP_LORA_OUTPUT" "$STAGE2_LORA_DIR"
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
    if [ ! -f "$FINAL_MERGED_MODEL" ]; then
        echo "⚠️ Step 3 failed to produce output. Check logs."
        exit 1
    fi

    echo "🎉 All Steps Completed. Final model is at $FINAL_MERGED_MODEL"
else
    echo "✅ Pipeline fully finished."
fi
