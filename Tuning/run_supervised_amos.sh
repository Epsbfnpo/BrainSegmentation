#!/bin/bash
set -euo pipefail

# 环境变量
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 项目路径
REPO_ROOT="/datasets/work/hb-nhmrc-dhcp/work/liu275"
CODE_DIR="${REPO_ROOT}/Supervised_Finetune" # 假设你把代码放在这
# 结果输出路径
RESULTS_DIR="${REPO_ROOT}/results/amos_supervised_finetune"
mkdir -p "${RESULTS_DIR}"

# 关键输入路径
# 1. 数据索引 (我们之前生成的含Label的JSON)
DATA_SPLIT_JSON="${REPO_ROOT}/AMOS_pretrain_split.json"
# 2. 预训练模型 (SSL阶段的产出)
# 请确保这个路径指向你上一阶段实际生成的 final_model.pth
PRETRAINED_MODEL="${REPO_ROOT}/results/amos_ssl_pretrain/final_model.pth"

# 训练配置
EXPERIMENT_NAME="AMOS_CT_Finetune_From_SSL"
EPOCHS=300
BATCH_SIZE=2
NUM_GPUS=4
LEARNING_RATE=1e-4

# 模型几何参数 (与 SSL 阶段保持一致)
ROI_X=128
ROI_Y=128
ROI_Z=128
NUM_CLASSES=15  # 0(背景) + 14(器官)
TARGET_SPACING="1.5 1.5 1.5"

# 启动训练
cd "${CODE_DIR}"

echo "🚀 Starting AMOS Supervised Fine-tuning"
echo "   Pretrained Weights: ${PRETRAINED_MODEL}"
echo "   Output Dir: ${RESULTS_DIR}"

torchrun --nproc_per_node=${NUM_GPUS} \
    main_supervised_dhcp.py \
    --exp_name "${EXPERIMENT_NAME}" \
    --results_dir "${RESULTS_DIR}" \
    --data_split_json "${DATA_SPLIT_JSON}" \
    --pretrained_model "${PRETRAINED_MODEL}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --num_classes ${NUM_CLASSES} \
    --roi_x ${ROI_X} --roi_y ${ROI_Y} --roi_z ${ROI_Z} \
    --target_spacing ${TARGET_SPACING} \
    --num_workers 8 \
    --cache_rate 0.1 \
    --use_amp
