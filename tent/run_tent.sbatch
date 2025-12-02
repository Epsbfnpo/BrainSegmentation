#!/bin/bash -l
#SBATCH --job-name=tent-adapt
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
#SBATCH --signal=B:USR1@120

set -euo pipefail

echo "=============================================================="
echo "TENT ADAPTATION JOB STARTED"
echo "Job $SLURM_JOB_ID started at $(date)"
echo "=============================================================="

# 固定绝对路径，避免依赖未知环境变量或 Slurm 暂存目录
WORKDIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/tent"
RUN_SCRIPT="${WORKDIR}/run_tent.sh"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_ALLOW_TF32=1

cd "${WORKDIR}"

if [ ! -f "${RUN_SCRIPT}" ]; then
    echo "❌ Missing run script: ${RUN_SCRIPT}"
    echo "Make sure run_tent.sh is present next to run_tent.sbatch on the compute node."
    exit 127
fi

# 明确指定 Conda 路径，避免依赖 $HOME 等环境变量
CONDA_SETUP="/datasets/work/hb-nhmrc-dhcp/work/liu275/miniconda3/bin/activate"
if [ -f "${CONDA_SETUP}" ]; then
    source "${CONDA_SETUP}"
    conda activate brainseg || true
fi

RESULTS_DIR="${RESULTS_DIR:-${WORKDIR}/../results/target_tent}"
FINAL_MODEL="${RESULTS_DIR}/best_model.pt"
LATEST_MODEL="${RESULTS_DIR}/latest_model.pt"
SCRIPT_PATH="${WORKDIR}/run_tent.sbatch"

set +e
bash "${RUN_SCRIPT}"
STATUS=$?
set -e

echo "=============================================================="
echo "Job finished with status $STATUS"
echo "=============================================================="

if [ -f "$FINAL_MODEL" ]; then
    echo "🏁 Training appears complete (found best_model)."
elif [ -f "$LATEST_MODEL" ]; then
    echo "🔁 Checkpoint found but training not explicitly marked done. Requeuing..."
    sbatch "$SCRIPT_PATH"
else
    echo "⚠️  No checkpoint found. Job failed or first run crashed."
    exit $STATUS
fi

exit $STATUS
