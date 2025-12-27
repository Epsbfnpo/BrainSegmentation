#!/bin/bash

# =================================================================
# MedSeqFT Testing Script
# Designed to be functionally identical to run_testing_salt.sh
# =================================================================

set -euo pipefail

# 1. 基础路径配置
# 请修改为您的 medseqft 代码所在目录
REPO_ROOT="/datasets/work/hb-nhmrc-dhcp/work/liu275/medseqft"
TEST_SCRIPT="${REPO_ROOT}/test_medseqft.py"

# MedSeqFT 最终融合模型路径 (由 merge_model.py 生成)
MODEL_PATH="${REPO_ROOT}/results/target_medseqft/stage1/best_model.pt"

# 测试集 Split 文件 (保持和 SALT 一致)
TEST_SPLIT_JSON="${REPO_ROOT}/../PPREMOPREBO_split_test.json"

# 输出目录
OUTPUT_DIR="${REPO_ROOT}/results/target_medseqft/test_predictions"
METRICS_PATH="${REPO_ROOT}/results/target_medseqft/analysis/test_metrics.json"

# 2. 先验知识文件 (必须与 SALT 测试使用完全相同的文件)
# 请检查这些路径是否正确
PRIOR_DIR="${REPO_ROOT}/../new/priors/target"
ADJ_PRIOR="${PRIOR_DIR}/adjacency_prior.npz"
STRUCT_RULES="${PRIOR_DIR}/structural_rules.json"
LR_PAIRS="${PRIOR_DIR}/dhcp_lr_swap.json"
CLASS_MAP="${PRIOR_DIR}/class_map.json"

echo "Running MedSeqFT evaluation..."
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"

# 3. 运行测试
# 注意：MedSeqFT 不需要 salt_rank 等参数，但推理参数(ROI/Overlap)应保持一致
python3 "${TEST_SCRIPT}" \
    --split_json "${TEST_SPLIT_JSON}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --metrics_path "${METRICS_PATH}" \
    --adjacency_prior "${ADJ_PRIOR}" \
    --structural_rules "${STRUCT_RULES}" \
    --laterality_pairs_json "${LR_PAIRS}" \
    --class_map_json "${CLASS_MAP}" \
    --out_channels 87 \
    --roi_x 128 --roi_y 128 --roi_z 128 \
    --resample_to_native \
    --sw_batch_size 4 \
    --sw_overlap 0.5 \
    --use_amp

echo "Done. Comparison metrics saved to ${METRICS_PATH}"