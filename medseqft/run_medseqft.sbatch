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

echo "Job $SLURM_JOB_ID started at $(date)"

# Environment
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

cd /datasets/work/hb-nhmrc-dhcp/work/liu275/medseqft
source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

RESULTS_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/medseqft/results/target_medseqft"
FINAL_MODEL="${RESULTS_DIR}/final_model.pt"
SCRIPT_PATH="/datasets/work/hb-nhmrc-dhcp/work/liu275/medseqft/run_medseqft.sbatch"

set +e
bash run_training.sh
STATUS=$?
set -e

echo "Job ended with status $STATUS at $(date)"

# Resubmit Logic
if [ -f "$FINAL_MODEL" ]; then
    echo "🏁 Training complete."
else
    # 143 usually means SIGTERM (timeout)
    if [ "$STATUS" -eq 0 ] || [ "$STATUS" -eq 143 ]; then
        echo "🔁 Job timed out or incomplete. Resubmitting..."
        sbatch "$SCRIPT_PATH"
    else
        echo "❌ Job failed. Not resubmitting."
    fi
fi

exit $STATUS
