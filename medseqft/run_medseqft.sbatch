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
#SBATCH --time=48:00:00
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

# Pipeline artifacts
FINAL_MODEL_PATH="${RESULTS_DIR}/final_model.pt"
LATEST_MODEL_PATH="${RESULTS_DIR}/latest_checkpoint.pt"

set +e
bash run_training.sh
STATUS=$?
set -e

echo "=============================================================="
echo "Job $SLURM_JOB_ID ended at $(date) with exit status $STATUS"
echo "=============================================================="

# Stateless, file-based resubmit logic
if [ -f "$FINAL_MODEL_PATH" ]; then
    echo "🏁 Final model detected at $FINAL_MODEL_PATH; training complete."
elif [ -f "$LATEST_MODEL_PATH" ]; then
    echo "🔁 Training incomplete (final model missing, latest checkpoint present); requeuing job..."
    sbatch "$SCRIPT_PATH"
else
    echo "❌ Job failed (no final model, no checkpoint). Exit status $STATUS"
    exit "$STATUS"
fi
