#!/bin/bash -l

#SBATCH --job-name=target-salt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=OD-210257
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=24:00:00

set -euo pipefail

echo "=============================================================="
echo "TARGET SALT TRAINING JOB"
echo "Job $SLURM_JOB_ID started at $(date)"
echo "=============================================================="

# Environment Setup
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_ALLOW_TF32=1
export CUBLAS_ALLOW_TF32=1

# Activate Conda (Adjust path to match your environment used in L2-SP)
source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

REPO_ROOT="/datasets/work/hb-nhmrc-dhcp/work/liu275/salt"
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results/target_salt}
export RESULTS_DIR

SCRIPT_PATH="${REPO_ROOT}/run_salt.sbatch"
FINAL_MODEL_PATH="${RESULTS_DIR}/final_model.pt"
LATEST_MODEL_PATH="${RESULTS_DIR}/latest_model.pt"

set +e
bash ./run_salt.sh
STATUS=$?
set -e

echo "Job finished with status $STATUS"

if [ "$STATUS" -eq 3 ]; then
    echo "⏳ Time limit hit. Resubmitting job."
    sbatch --export=ALL "$SCRIPT_PATH"
    exit 0
fi

if [ -f "$FINAL_MODEL_PATH" ]; then
    echo "🏁 Final model detected; training complete."
elif [ "$STATUS" -ne 0 ]; then
    echo "❌ Job failed with status $STATUS."
elif [ -f "$LATEST_MODEL_PATH" ]; then
    echo "⚠️ Job exited without final model; latest checkpoint present."
else
    echo "⚠️ Job finished but no checkpoint found."
fi

exit $STATUS
