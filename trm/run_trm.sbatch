#!/bin/bash
#SBATCH --job-name=trm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=trm-%j.out
#SBATCH --error=trm-%j.err

set -euo pipefail

module load cuda/12.1 || true
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate trm || true

export NUM_GPUS=${NUM_GPUS:-1}
export SPLIT_JSON=${SPLIT_JSON:-"${PWD}/PPREMOPREBO_split.json"}
export RESULTS_DIR=${RESULTS_DIR:-"${PWD}/trm_runs"}
export PRETRAINED=${PRETRAINED:-"/path/to/source_checkpoint.ckpt"}

bash $(dirname "$0")/run_trm.sh "$@"
