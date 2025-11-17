#!/bin/bash -l
#SBATCH --job-name=ugtst-ppremo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=OD-210257
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=2:00:00
#SBATCH --signal=B:TERM@300

set -euo pipefail

echo "=============================================================="
echo "UGTST TARGET-ONLY TRAINING"
echo "Job $SLURM_JOB_ID started at $(date)"
echo "=============================================================="

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_ALLOW_TF32=1
export CUBLAS_ALLOW_TF32=1

cd /datasets/work/hb-nhmrc-dhcp/work/liu275/UGTST-main
source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

REPO_ROOT="/datasets/work/hb-nhmrc-dhcp/work/liu275/UGTST-main"
RUN_SCRIPT="${REPO_ROOT}/run_training_ugtst.sh"
SCRIPT_PATH="${REPO_ROOT}/run_ugtst_flex.sbatch"
EXP_NAME=${EXP_NAME:-UGTST_PPREMO}
MODEL_DIR="${REPO_ROOT}/model/${EXP_NAME}"
FINAL_MODEL_PATH="${MODEL_DIR}/UNet_best_model.pth"
COMPLETE_FLAG="${MODEL_DIR}/training_complete.txt"

set +e
bash "${RUN_SCRIPT}"
STATUS=$?
set -e

echo "=============================================================="
echo "Job $SLURM_JOB_ID ended at $(date) with exit status $STATUS"
echo "=============================================================="

if [ -f "$COMPLETE_FLAG" ] && [ -f "$FINAL_MODEL_PATH" ]; then
    echo "🏁 Training complete. Final model: $FINAL_MODEL_PATH"
elif compgen -G "${MODEL_DIR}/iter_*.pth" > /dev/null; then
    echo "🔁 Latest checkpoint detected but no final model; resubmitting."
    sbatch --export=ALL "$SCRIPT_PATH"
else
    if [ "$STATUS" -ne 0 ]; then
        echo "❌ Job exited with status $STATUS and no checkpoints; manual check required."
    else
        echo "⚠️ Job finished without checkpoints; not resubmitting."
    fi
fi

exit $STATUS
