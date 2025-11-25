#!/bin/bash -l
#SBATCH --job-name=pure-ft
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

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

echo "=============================================================="
echo "PURE FINE-TUNING JOB"
echo "Job $SLURM_JOB_ID started at $(date)"
echo "=============================================================="

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_ALLOW_TF32=1
export CUBLAS_ALLOW_TF32=1

REPO_ROOT="/datasets/work/hb-nhmrc-dhcp/work/liu275/pure_finetuning"
cd "${REPO_ROOT}"
source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results/pure_finetuning}
export RESULTS_DIR
FINAL_MODEL_PATH="${RESULTS_DIR}/final_model.pt"
LATEST_MODEL_PATH="${RESULTS_DIR}/latest_model.pt"

set +e
bash ./run_training_graphalign.sh
STATUS=$?
set -e

echo "=============================================================="
echo "Job $SLURM_JOB_ID ended at $(date) with exit status $STATUS"
echo "Results directory: ${RESULTS_DIR}"
echo "=============================================================="

if [ -f "$FINAL_MODEL_PATH" ]; then
    echo "🏁 Final model detected at $FINAL_MODEL_PATH; training complete."
elif [ -f "$LATEST_MODEL_PATH" ]; then
    echo "🔁 Training incomplete (final model missing, latest checkpoint present); requeuing job."
    sbatch --export=ALL "$SCRIPT_PATH"
else
    if [ "$STATUS" -ne 0 ]; then
        echo "❌ Job exited with status $STATUS and no latest checkpoint; manual intervention required."
    else
        echo "⚠️ Job completed without producing final or latest checkpoints; not resubmitting."
    fi
fi

exit $STATUS
