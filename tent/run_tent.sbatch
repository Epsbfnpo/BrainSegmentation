#!/bin/bash -l
#SBATCH --job-name=tent-adapt
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
echo "TENT ADAPTATION JOB STARTED"
echo "Job $SLURM_JOB_ID started at $(date)"
echo "=============================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_ALLOW_TF32=1

cd "${SCRIPT_DIR}"

if [ -f "${HOME}/miniconda3/bin/activate" ]; then
    source "${HOME}/miniconda3/bin/activate"
fi
if command -v conda >/dev/null 2>&1; then
    conda activate brainseg || true
fi

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/target_tent}"
FINAL_MODEL="${RESULTS_DIR}/best_model.pt"
LATEST_MODEL="${RESULTS_DIR}/latest_model.pt"
SCRIPT_PATH="${SCRIPT_DIR}/run_tent.sbatch"

set +e
bash ./run_tent.sh
STATUS=$?
set -e

echo "=============================================================="
echo "Job finished with status $STATUS"
echo "=============================================================="

if [ -f "$FINAL_MODEL" ]; then
    echo "🏁 Training appears complete (found best_model)."
elif [ -f "$LATEST_MODEL" ]; then
    echo "🔁 Checkpoint found but training not explicitly marked done. Requeuing..."
    sbatch "$SCRIPT_PATH"
else
    echo "⚠️  No checkpoint found. Job failed or first run crashed."
    exit $STATUS
fi

exit $STATUS
