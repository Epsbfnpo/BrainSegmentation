#!/bin/bash
#SBATCH --job-name=pure_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=pure_finetune_%j.out
#SBATCH --error=pure_finetune_%j.err

module load python/3.10 || true
module load cuda/12.1 || true

cd "$(dirname "$0")"

bash run_training_pure.sh
