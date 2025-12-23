#!/bin/bash -l
#SBATCH --job-name=medseqft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=OD-210257
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=6:00:00
#SBATCH --signal=B:TERM@300

set -euo pipefail

echo "=============================================================="
echo "Job $SLURM_JOB_ID started at $(date)"
echo "=============================================================="

# Environment
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

cd /datasets/work/hb-nhmrc-dhcp/work/liu275/medseqft
source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

RESULTS_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/medseqft/results/target_medseqft"
SCRIPT_PATH="/datasets/work/hb-nhmrc-dhcp/work/liu275/medseqft/run_medseqft.sbatch"

FINAL_MODEL_PATH="${RESULTS_DIR}/final_model.pt"

# Stage 1 (FFT) 的断点
STAGE1_CKPT="${RESULTS_DIR}/stage1/latest_checkpoint.pt"
# Stage 2 (LoRA) 的断点
STAGE2_CKPT="${RESULTS_DIR}/stage2/lora_ckpt_latest.pt"
# Stage 0 (MDS) 的断点 (如果是 calculate_mds.py 保存的临时文件)
MDS_PROGRESS="${RESULTS_DIR}/mds_progress.json"

set +e
bash run_training.sh
STATUS=$?
set -e

echo "=============================================================="
echo "Job $SLURM_JOB_ID ended at $(date) with exit status $STATUS"
echo "=============================================================="

# --- 多阶段兼容的续训逻辑 ---

# 1. 如果最终模型存在，说明全部完成
if [ -f "$FINAL_MODEL_PATH" ]; then
    echo "🏁 Final model detected at $FINAL_MODEL_PATH; training complete."

# 2. 如果 Exit Code 是 0 (代表 Python 正常响应超时)
#    AND (Stage 1 断点存在 OR Stage 2 断点存在 OR MDS 进度存在)
elif [ "$STATUS" -eq 0 ] && \
     ( [ -f "$STAGE1_CKPT" ] || [ -f "$STAGE2_CKPT" ] || [ -f "$MDS_PROGRESS" ] ); then

    echo "🔁 Training incomplete (Timeout/Exit 0 detected). Found checkpoint. Requeuing job..."
    sbatch "$SCRIPT_PATH"

# 3. 其他情况视为失败
else
    echo "❌ Job failed or Crashed (Status $STATUS). Not resubmitting."
    # 打印一下当前状态方便调试
    echo "   Debug info:"
    echo "   - Final Model Exists? $([ -f "$FINAL_MODEL_PATH" ] && echo Yes || echo No)"
    echo "   - Stage 1 Ckpt Exists? $([ -f "$STAGE1_CKPT" ] && echo Yes || echo No)"
    echo "   - Stage 2 Ckpt Exists? $([ -f "$STAGE2_CKPT" ] && echo Yes || echo No)"
    exit "$STATUS"
fi