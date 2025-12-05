#!/bin/bash
set -euo pipefail

# 获取当前脚本所在目录 (trm/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 获取项目根目录 (上级目录)，用于定位 split json
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------- 1. 默认参数配置 ----------

# 硬件资源
NUM_GPUS=${NUM_GPUS:-1}

# 数据路径
SPLIT_JSON=${SPLIT_JSON:-"${REPO_ROOT}/PPREMOPREBO_split.json"}
RESULTS_DIR=${RESULTS_DIR:-"${REPO_ROOT}/results/target_trm"}

# 关键输入：源域模型权重
# 已更新为您指定的 dHCP 预训练模型路径
DEFAULT_PRETRAINED="/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/results_fixed/dHCP_registered_fixed/best_model.pth"
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-"${DEFAULT_PRETRAINED}"}

# 训练超参
EPOCHS=${EPOCHS:-100}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}   # TRM 特有：前 N 个 epoch 在线统计分布，之后冻结
BATCH_SIZE=${BATCH_SIZE:-1}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
ACCUM_STEPS=${ACCUM_STEPS:-1}
SEED=${SEED:-42}

# 模型与数据几何参数
ROI_X=${ROI_X:-128}
ROI_Y=${ROI_Y:-128}
ROI_Z=${ROI_Z:-128}
OUT_CHANNELS=${OUT_CHANNELS:-87}
FEATURE_SIZE=${FEATURE_SIZE:-48}
TARGET_SPACING=${TARGET_SPACING:-"0.8 0.8 0.8"}

# 性能参数
NUM_WORKERS=${NUM_WORKERS:-4}
CACHE_RATE=${CACHE_RATE:-0.0}

# TRM 算法特有参数 (动量更新系数)
TRM_MOMENTUM=${TRM_MOMENTUM:-0.9}

# 开关标记
FOREGROUND_ONLY=${FOREGROUND_ONLY:-1}
USE_SWIN_CKPT=${USE_SWIN_CKPT:-1}

# ---------- 2. 检查必要文件 ----------

if [ ! -f "${SPLIT_JSON}" ]; then
    echo "❌ Error: Split file not found at ${SPLIT_JSON}"
    exit 1
fi

# 检查预训练模型是否存在（仅警告，防止在某些节点上路径暂时不可达导致脚本直接退出）
if [ ! -f "${PRETRAINED_CHECKPOINT}" ]; then
    echo "⚠️  Warning: Pretrained checkpoint not found at:"
    echo "   ${PRETRAINED_CHECKPOINT}"
    echo "   Please verify the path or file permissions."
fi

mkdir -p "${RESULTS_DIR}"

# ---------- 3. 构建命令 ----------

CMD=(
    torchrun --nproc_per_node="${NUM_GPUS}"
    "${SCRIPT_DIR}/train_trm.py"
    
    # --- 路径 ---
    --split_json "${SPLIT_JSON}"
    --results_dir "${RESULTS_DIR}"
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
    
    # --- 训练参数 ---
    --epochs "${EPOCHS}"
    --warmup_epochs "${WARMUP_EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --val_batch_size "${VAL_BATCH_SIZE}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --accumulation_steps "${ACCUM_STEPS}"
    --seed "${SEED}"
    
    # --- 数据几何 ---
    --roi_x "${ROI_X}" --roi_y "${ROI_Y}" --roi_z "${ROI_Z}"
    --target_spacing ${TARGET_SPACING}
    --apply_spacing
    --apply_orientation
    
    # --- 模型结构 ---
    --out_channels "${OUT_CHANNELS}"
    --feature_size "${FEATURE_SIZE}"
    
    # --- 性能与算法细节 ---
    --num_workers "${NUM_WORKERS}"
    --cache_rate "${CACHE_RATE}"
    --trm_momentum "${TRM_MOMENTUM}"
)

# ---------- 4. 动态添加 Flag 参数 ----------

if [ "${FOREGROUND_ONLY}" -eq 1 ]; then
    CMD+=(--foreground_only)
fi

if [ "${USE_SWIN_CKPT}" -eq 0 ]; then
    CMD+=(--no_swin_checkpoint)
fi

# 注意：此处彻底移除了 Weighted Sampling 和 Volume Stats 的逻辑
# 保证本方法纯净、独立，无 Prior 依赖

# ---------- 5. 打印并执行 ----------

echo "=============================================================="
echo "🚀 Launching TRM (Transfer Risk Map) Training Baseline"
echo "   Time: $(date)"
echo "   GPUs: ${NUM_GPUS}"
echo "   Results Dir: ${RESULTS_DIR}"
echo "   Pretrained: ${PRETRAINED_CHECKPOINT}"
echo "=============================================================="

"${CMD[@]}" "$@"
