#!/bin/bash -l
#SBATCH --job-name=umamba-target
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
echo "U-MAMBA TARGET-ONLY TRAINING"
echo "Job $SLURM_JOB_ID started at $(date)"
echo "=============================================================="

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate environment
source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

WORKDIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/U-Mamba"
cd "${WORKDIR}"

RESULTS_DIR="../results/target_umamba"
FINAL_MODEL="${RESULTS_DIR}/final_model.pt"
LATEST_MODEL="${RESULTS_DIR}/latest_model.pt"

set +e
bash ./run_training_umamba.sh
STATUS=$?
set -e

if [ -f "$FINAL_MODEL" ]; then
    echo "🏁 Training complete."
elif [ -f "$LATEST_MODEL" ]; then
    echo "🔁 Time limit reached, requeuing..."
    sbatch run_umamba.sbatch
else
    echo "❌ Job failed."
    exit $STATUS
fi
