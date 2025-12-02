#!/bin/bash -l
#SBATCH --job-name=shot-adapt
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
#SBATCH --signal=B:USR1@120

set -euo pipefail

echo "=============================================================="
echo "SHOT ADAPTATION JOB STARTED"
echo "Job $SLURM_JOB_ID started at $(date)"
echo "=============================================================="

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_ALLOW_TF32=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

# Activate environment if available
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    conda activate brainseg || true
fi

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/target_shot}"
FINAL_MODEL="${RESULTS_DIR}/final_model.pt"
LATEST_MODEL="${RESULTS_DIR}/latest_model.pt"

set +e
bash "${SCRIPT_DIR}/run_shot.sh"
STATUS=$?
set -e

echo "=============================================================="
echo "Job finished with status $STATUS"
echo "=============================================================="

if [ -f "$FINAL_MODEL" ]; then
    echo "🏁 Training appears complete (found final_model.pt)."
elif [ -f "$LATEST_MODEL" ]; then
    echo "🔁 Checkpoint found but training not explicitly marked done. Requeuing..."
    sbatch "$SCRIPT_DIR/run_shot.sbatch"
else
    echo "⚠️  No checkpoint found. Job failed or first run crashed."
    exit $STATUS
fi

exit $STATUS
