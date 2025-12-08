#!/bin/bash
set -euo pipefail

# 环境变量设置
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
# 显存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

# 项目路径 (请根据实际情况调整)
REPO_ROOT="/datasets/work/hb-nhmrc-dhcp/work/liu275"
CODE_DIR="${REPO_ROOT}/LoRA"  # 假设 SSL 代码也在这里，或者修改为你存放 SSL 代码的路径
# 注意：你需要把第一阶段的代码放到服务器上，假设放在 ${REPO_ROOT}/SSL_Pretrain
SSL_CODE_DIR="${REPO_ROOT}/SSL_Pretrain" 

# 输出路径
RESULTS_DIR="${REPO_ROOT}/results/amos_ssl_pretrain"
mkdir -p "${RESULTS_DIR}"

# 数据配置
DATA_SPLIT_JSON="${REPO_ROOT}/AMOS_pretrain_split.json"

# 训练超参数
EXPERIMENT_NAME="AMOS_CT_SSL"
EPOCHS=100           # 预训练不需要太多 epoch，收敛即可
BATCH_SIZE=2         # 3D SwinUNETR 显存占用大，单卡2是安全值
NUM_GPUS=4           # 使用 4 张卡
LEARNING_RATE=1e-4

# 模型几何参数 (适配 CT)
ROI_X=96
ROI_Y=96
ROI_Z=96
TARGET_SPACING="1.5 1.5 1.5" # 关键修改：从 0.5 改为 1.5

# 启动训练
# 注意：这里假设您已经把第一阶段的所有 .py 文件放到了 $SSL_CODE_DIR
cd "${SSL_CODE_DIR}"

echo "🚀 Starting AMOS SSL Pre-training"
echo "   Split: ${DATA_SPLIT_JSON}"
echo "   Output: ${RESULTS_DIR}"
echo "   Spacing: ${TARGET_SPACING}"

torchrun --nproc_per_node=${NUM_GPUS} \
    main_ssl_dhcp.py \
    --experiment_name "${EXPERIMENT_NAME}" \
    --results_dir "${RESULTS_DIR}" \
    --data_split_json "${DATA_SPLIT_JSON}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --roi_x ${ROI_X} --roi_y ${ROI_Y} --roi_z ${ROI_Z} \
    --target_spacing ${TARGET_SPACING} \
    --num_workers 8 \
    --cache_rate 0.1 \
    --use_amp  # 开启混合精度以节省显存
