#!/bin/bash

# =================================================================
# MedSeqFT Testing Script
# Designed to be functionally identical to run_testing_salt.sh
# =================================================================

set -euo pipefail

# 1. 基础路径配置
REPO_ROOT="/datasets/work/hb-nhmrc-dhcp/work/liu275/medseqft" # 你的代码路径
TEST_SCRIPT="${REPO_ROOT}/test_medseqft.py"

# 模型路径 (训练完后生成的)
MODEL_PATH="/datasets/work/hb-nhmrc-dhcp/work/liu275/medseqft/results/target_totalseg/final_model.pt"

# [关键修改] 测试集 Split JSON (复用同一个 JSON，里面包含了 testing 字段)
TEST_SPLIT_JSON="/datasets/work/hb-nhmrc-dhcp/work/liu275/TotalSegmentator_14cls_final_split.json"

# 输出目录
OUTPUT_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/medseqft/results/target_totalseg/test_predictions"
METRICS_PATH="/datasets/work/hb-nhmrc-dhcp/work/liu275/medseqft/results/target_totalseg/test_metrics.json"

# [关键修改] 移除或更新先验知识文件 (Prior Knowledge)
# dHCP 有 adjacency_prior 和 structural_rules，TotalSegmentator 暂时没有
# 如果 test_medseqft.py 强依赖这些，你需要创建一个空的或者假的，或者修改 python 代码跳过这些检查
# 建议：暂时把这些变量留空，或者指向不存在的文件，并在 Python 中做容错
PRIOR_DIR=""
ADJ_PRIOR=""
STRUCT_RULES=""
LR_PAIRS=""
CLASS_MAP=""

echo "Running MedSeqFT evaluation..."
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"

# 3. 运行测试
# 注意：MedSeqFT 不需要 salt_rank 等参数，但推理参数(ROI/Overlap)应保持一致
python "${TEST_SCRIPT}" \
    --split_json "${TEST_SPLIT_JSON}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --metrics_path "${METRICS_PATH}" \
    --adjacency_prior "${ADJ_PRIOR}" \
    --structural_rules "${STRUCT_RULES}" \
    --laterality_pairs_json "${LR_PAIRS}" \
    --class_map_json "${CLASS_MAP}" \
    --roi_x 128 --roi_y 128 --roi_z 128 \
    --out_channels 15 \
    --feature_size 48 \
    --resample_to_native \
    --sw_batch_size 1 \
    --sw_overlap 0.7 \
    --use_amp

echo "Done. Comparison metrics saved to ${METRICS_PATH}"
