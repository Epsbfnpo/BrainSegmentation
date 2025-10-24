#!/bin/bash -l
#SBATCH --job-name=causal
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

echo "=============================================================="
echo "CAUSAL CROSS-DOMAIN GRAPH ALIGNMENT JOB"
echo "Job $SLURM_JOB_ID started at $(date)"
echo "Running on host $(hostname)"
echo "=============================================================="

# Threading and NCCL hygiene
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_ALLOW_TF32=1
export CUBLAS_ALLOW_TF32=1

# Project root for causal stack
cd /datasets/work/hb-nhmrc-dhcp/work/liu275/causal

# Activate Python environment
source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

# Launch causal training run
bash ./run_training_graphalign_causal.sh
EXIT_STATUS=$?

echo "=============================================================="
echo "Job $SLURM_JOB_ID completed with status $EXIT_STATUS at $(date)"
echo "=============================================================="

# Auto-resubmit logic to continue long trainings over multiple 2h slots
RESULTS_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/causal/results"
LATEST_CHECKPOINT="${RESULTS_DIR}/latest.pth"
FINAL_MODEL="${RESULTS_DIR}/final_model.pth"

if [ -f "$LATEST_CHECKPOINT" ] && [ ! -f "$FINAL_MODEL" ]; then
    echo "Training not yet converged. Preparing to resubmit job after cooldown."
    sbatch --dependency=singleton --job-name=causal "$0"
    RESUBMIT_STATUS=$?
    if [ $RESUBMIT_STATUS -ne 0 ]; then
        echo "Failed to queue follow-up job (status $RESUBMIT_STATUS). Please resubmit manually."
    else
        echo "Follow-up job queued successfully."
    fi
fi

exit $EXIT_STATUS