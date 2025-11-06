#!/bin/bash
#SBATCH --job-name=texture
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=OD-210257
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export TORCH_ALLOW_TF32=1
export CUBLAS_ALLOW_TF32=1

cd /datasets/work/hb-nhmrc-dhcp/work/liu275/texture

source /datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate
conda activate brainseg

NUM_GPUS_REQ=${SLURM_GPUS_PER_NODE:-4}
NUM_WORKERS_REQ=$(( SLURM_CPUS_PER_TASK * SLURM_NTASKS_PER_NODE / NUM_GPUS_REQ ))

NUM_GPUS=${NUM_GPUS_REQ} \
NUM_WORKERS=${NUM_WORKERS_REQ} \
JOB_TIME_LIMIT_MINUTES=115 \
TIME_BUFFER_MINUTES=5 \
DIST_TIMEOUT_MINUTES=180 \
bash ./run_texture_training.sh
