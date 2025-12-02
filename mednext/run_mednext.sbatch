#!/bin/bash -l
#SBATCH --job-name=mednext-base
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
echo "MEDNEXT TARGET BASELINE JOB STARTED"
echo "Job $SLURM_JOB_ID started at $(date)"
echo "=============================================================="

WORKDIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/mednext"
cd "${WORKDIR}"

source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

RESULTS_DIR="../results/target_mednext_baseline"
FINAL_MODEL="${RESULTS_DIR}/final_model.pt"
LATEST_MODEL="${RESULTS_DIR}/latest_model.pt"

set +e
bash ./run_mednext.sh
STATUS=$?
set -e

echo "Job finished with status $STATUS"

if [ -f "$FINAL_MODEL" ]; then
    echo "🏁 Training complete."
elif [ -f "$LATEST_MODEL" ]; then
    echo "🔁 Training interrupted. Requeuing..."
    sbatch run_mednext.sbatch
else
    echo "❌ Training failed or didn't start properly."
    exit $STATUS
fi
