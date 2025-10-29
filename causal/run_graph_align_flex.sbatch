#!/bin/bash -l
#SBATCH --job-name=graph_align_flex_new
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=OD-210257
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=2:00:00         # 2 hours per job
#SBATCH --signal=B:TERM@300    # Send SIGTERM 5 minutes before time limit

echo "=============================================================="
echo "CROSS-DOMAIN GRAPH ALIGNMENT JOB"
echo "Job $SLURM_JOB_ID started at $(date)"
echo "=============================================================="

# Set environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_ALLOW_TF32=1
export CUBLAS_ALLOW_TF32=1

# Fixed path
cd /datasets/work/hb-nhmrc-dhcp/work/liu275/new

# Activate environment
source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

# Run training with cross-domain alignment
bash ./run_training_graphalign.sh

# Check exit status
EXIT_STATUS=$?

echo "=============================================================="
echo "Job $SLURM_JOB_ID ended at $(date) with exit status $EXIT_STATUS"
echo "=============================================================="

# Auto-resubmission is handled inside run_training_graphalign.sh to avoid
# double-submitting follow-up jobs that would otherwise clutter the queue.
