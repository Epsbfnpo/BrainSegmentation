#!/bin/bash
#SBATCH --job-name=texture
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=OD-210257
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=24:00:00

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

bash ./run_training_texture.sh
