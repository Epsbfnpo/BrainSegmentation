#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ===== User configuration =====
RESULTS_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/texture/results"
SOURCE_SPLIT_JSON="${REPO_ROOT}/dHCP_split.json"
TARGET_SPLIT_JSON="${REPO_ROOT}/PPREMOPREBO_split.json"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/datasets/work/hb-nhmrc-dhcp/work/liu275/model_final.pt}"

NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
EPOCHS=${EPOCHS:-200}
JOB_TIME_LIMIT_MINUTES=${JOB_TIME_LIMIT_MINUTES:-115}
TIME_BUFFER_MINUTES=${TIME_BUFFER_MINUTES:-5}
DIST_TIMEOUT_MINUTES=${DIST_TIMEOUT_MINUTES:-180}

mkdir -p "${RESULTS_DIR}"

export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}

CMD=(
    "${SCRIPT_DIR}/train_texture.py"
    --source_split_json "${SOURCE_SPLIT_JSON}"
    --target_split_json "${TARGET_SPLIT_JSON}"
    --results_dir "${RESULTS_DIR}"
    --batch_size "${BATCH_SIZE}"
    --val_batch_size "${VAL_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --epochs "${EPOCHS}"
    --lr 5e-5
    --weight_decay 1e-4
    --roi_x 96 --roi_y 96 --roi_z 96
    --amp
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
    --domain_loss_weight 0.7
    --embed_align_weight 0.2
    --stats_align_weight 0.2
    --grl_lambda 1.0
    --foreground_only
    --auto_resume
    --job_time_limit "${JOB_TIME_LIMIT_MINUTES}"
    --time_buffer_minutes "${TIME_BUFFER_MINUTES}"
    --dist_timeout "${DIST_TIMEOUT_MINUTES}"
)

set +e
if [[ ${NUM_GPUS} -gt 1 ]]; then
    torchrun --nproc_per_node="${NUM_GPUS}" "${CMD[@]}"
else
    python "${CMD[@]}"
fi
EXIT_STATUS=$?
set -e

echo "=============================================================="
echo "TEXTURE ADAPTATION TRAINING SUMMARY"
echo "Command exited with status ${EXIT_STATUS}"
echo "Results directory: ${RESULTS_DIR}"
echo "=============================================================="

TRAIN_LOG="${RESULTS_DIR}/training.log"
LATEST_CKPT="${RESULTS_DIR}/checkpoint_latest.pth"
FINAL_MODEL="${RESULTS_DIR}/last_model.pth"
BEST_MODEL="${RESULTS_DIR}/best_model.pth"
ELASTIC_ERROR="${RESULTS_DIR}/elastic_error.json"

if [[ -f "${TRAIN_LOG}" ]]; then
    echo "📄 Last training log entries:"
    tail -n 15 "${TRAIN_LOG}" || true
    echo "--------------------------------------------------------------"
fi

if [[ ${EXIT_STATUS} -eq 0 ]]; then
    if [[ -f "${FINAL_MODEL}" ]]; then
        echo "✅ Training finished successfully."
        echo "   Final model : ${FINAL_MODEL}"
        [[ -f "${BEST_MODEL}" ]] && echo "   Best model  : ${BEST_MODEL}"
    else
        echo "⏸ Training paused (time limit or manual stop)."
        if [[ -f "${LATEST_CKPT}" ]]; then
            echo "   Latest checkpoint: ${LATEST_CKPT}"
            stat -c "   Modified: %y" "${LATEST_CKPT}" 2>/dev/null || true
        fi
    fi
else
    echo "❌ Training command exited with failure (${EXIT_STATUS})."
    if [[ -f "${ELASTIC_ERROR}" ]]; then
        echo "   Elastic error details:"
        python -c "import json; import sys; from pathlib import Path; p=Path(sys.argv[1]);\nprint(json.dumps(json.loads(p.read_text()), indent=2)[:2000])" "${ELASTIC_ERROR}" 2>/dev/null || head -n 20 "${ELASTIC_ERROR}" || true
    fi
    if [[ -f "${LATEST_CKPT}" ]]; then
        echo "   A checkpoint is available for recovery: ${LATEST_CKPT}"
    fi
fi

NEEDS_RESUBMIT=0
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    if [[ ! -f "${FINAL_MODEL}" ]]; then
        if [[ ${EXIT_STATUS} -eq 0 ]]; then
            NEEDS_RESUBMIT=1
        elif [[ -f "${LATEST_CKPT}" ]]; then
            NEEDS_RESUBMIT=1
        fi
    fi
fi

if [[ ${NEEDS_RESUBMIT} -eq 1 ]]; then
    echo ""
    echo "🔄 Auto-resubmitting follow-up job..."
    RESUBMIT_JOB_NAME=${SLURM_JOB_NAME:-texture}
    SUBMIT_DIR=${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}
    if command -v sbatch >/dev/null 2>&1; then
        sbatch --dependency=singleton --job-name="${RESUBMIT_JOB_NAME}" "${SUBMIT_DIR}/run_texture_training.sbatch"
    else
        echo "⚠️ 'sbatch' command not found; please resubmit manually."
    fi
else
    echo ""
    echo "No auto-resubmission required."
fi

echo "=============================================================="
echo "Helpful commands:"
echo "  tail -f ${TRAIN_LOG}"
echo "  ls -lh ${RESULTS_DIR}"
echo "=============================================================="
