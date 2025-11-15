#!/bin/bash -l
#SBATCH --job-name=target-priors
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
echo "TARGET-ONLY TRAINING JOB"
echo "Job $SLURM_JOB_ID started at $(date)"
echo "=============================================================="

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_ALLOW_TF32=1
export CUBLAS_ALLOW_TF32=1
export DISABLE_PRIOR=${DISABLE_PRIOR:-1}
export USE_AMP=${USE_AMP:-0}

cd /datasets/work/hb-nhmrc-dhcp/work/liu275/new
source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

REPO_ROOT="/datasets/work/hb-nhmrc-dhcp/work/liu275/new"
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results/target_only}
export RESULTS_DIR
RESUBMIT_SENTINEL="${RESULTS_DIR}/.resubmitted"
SCRIPT_PATH="${REPO_ROOT}/run_graph_align_flex.sbatch"

set +e
bash ./run_training_graphalign.sh
STATUS=$?
set -e

echo "=============================================================="
echo "Job $SLURM_JOB_ID ended at $(date) with exit status $STATUS"
echo "(DISABLE_PRIOR=${DISABLE_PRIOR} | USE_AMP=${USE_AMP})"
echo "=============================================================="

if [ "$STATUS" -eq 2 ]; then
    if [ ! -f "$RESUBMIT_SENTINEL" ]; then
        echo "⏱️  Time limit reached; scheduling a single resubmission"
        touch "$RESUBMIT_SENTINEL"
        sbatch --export=ALL,RESUBMITTED=1 "$SCRIPT_PATH"
    else
        echo "Resubmission already performed; not submitting again"
    fi
fi

exit $STATUS
