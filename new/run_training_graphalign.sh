#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

is_positive() {
    python - "$1" <<'PY'
import sys
try:
    value = float(sys.argv[1])
except Exception:
    sys.exit(1)
sys.exit(0 if value > 0 else 1)
PY
}

# ---------- User configuration ----------
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-400}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results/target_only}
TARGET_SPLIT_JSON=${TARGET_SPLIT_JSON:-${REPO_ROOT}/PPREMOPREBO_split.json}
TARGET_PRIOR_ROOT=${TARGET_PRIOR_ROOT:-${REPO_ROOT}/new/priors/target}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-}
OUT_CHANNELS=${OUT_CHANNELS:-87}
DISABLE_PRIOR=${DISABLE_PRIOR:-0}
USE_AMP=${USE_AMP:-1}
PRIOR_PROFILE=${PRIOR_PROFILE:-volume_shape_edge_sym}

ROI_X=${ROI_X:-128}
ROI_Y=${ROI_Y:-128}
ROI_Z=${ROI_Z:-128}
FEATURE_SIZE=${FEATURE_SIZE:-48}
LR=${LR:-5e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
LR_MIN=${LR_MIN:-1e-7}
LR_WARMUP_EPOCHS=${LR_WARMUP_EPOCHS:-20}
LR_WARMUP_START=${LR_WARMUP_START:-0.1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
EMA_DECAY=${EMA_DECAY:-0.0}
LAMBDA_VOLUME=${LAMBDA_VOLUME:-0.2}
LAMBDA_SHAPE=${LAMBDA_SHAPE:-0.2}
LAMBDA_EDGE=${LAMBDA_EDGE:-0.1}
LAMBDA_SPEC=${LAMBDA_SPEC:-0.05}
LAMBDA_REQUIRED=${LAMBDA_REQUIRED:-0.05}
LAMBDA_FORBIDDEN=${LAMBDA_FORBIDDEN:-0.05}
LAMBDA_SYM=${LAMBDA_SYM:-0.02}
LAMBDA_DYN=${LAMBDA_DYN:-0.2}

case "${PRIOR_PROFILE}" in
    volume_only)
        LAMBDA_VOLUME=${LAMBDA_VOLUME:-0.2}
        LAMBDA_SHAPE=0.0
        LAMBDA_EDGE=0.0
        LAMBDA_SPEC=0.0
        LAMBDA_REQUIRED=0.0
        LAMBDA_FORBIDDEN=0.0
        LAMBDA_SYM=0.0
        LAMBDA_DYN=0.0
        ;;
    volume_shape)
        LAMBDA_VOLUME=${LAMBDA_VOLUME:-0.2}
        LAMBDA_SHAPE=${LAMBDA_SHAPE:-0.2}
        LAMBDA_EDGE=0.0
        LAMBDA_SPEC=0.0
        LAMBDA_REQUIRED=0.0
        LAMBDA_FORBIDDEN=0.0
        LAMBDA_SYM=0.0
        LAMBDA_DYN=0.0
        ;;
    volume_shape_edge)
        LAMBDA_VOLUME=${LAMBDA_VOLUME:-0.2}
        LAMBDA_SHAPE=${LAMBDA_SHAPE:-0.2}
        LAMBDA_EDGE=${LAMBDA_EDGE:-0.1}
        LAMBDA_SPEC=0.0
        LAMBDA_REQUIRED=0.0
        LAMBDA_FORBIDDEN=0.0
        LAMBDA_SYM=0.0
        LAMBDA_DYN=0.0
        ;;
    volume_shape_edge_sym)
        LAMBDA_VOLUME=${LAMBDA_VOLUME:-0.2}
        LAMBDA_SHAPE=${LAMBDA_SHAPE:-0.2}
        LAMBDA_EDGE=${LAMBDA_EDGE:-0.1}
        LAMBDA_SPEC=0.0
        LAMBDA_REQUIRED=0.0
        LAMBDA_FORBIDDEN=0.0
        LAMBDA_SYM=${LAMBDA_SYM:-0.02}
        LAMBDA_DYN=0.0
        ;;
    full)
        :
        ;;
    custom)
        :
        ;;
    off)
        DISABLE_PRIOR=1
        ;;
    *)
        echo "Unknown PRIOR_PROFILE='${PRIOR_PROFILE}'. Supported profiles: volume_only, volume_shape, volume_shape_edge, volume_shape_edge_sym, full, custom, off" >&2
        exit 1
        ;;
esac

if [ "${DISABLE_PRIOR}" -ne 0 ]; then
    PRIOR_PROFILE="off"
fi

echo "➡️  Prior profile: ${PRIOR_PROFILE} (DISABLE_PRIOR=${DISABLE_PRIOR})"
printf '    lambdas: volume=%s shape=%s edge=%s spec=%s required=%s forbidden=%s symmetry=%s dyn=%s\n' \
    "${LAMBDA_VOLUME}" "${LAMBDA_SHAPE}" "${LAMBDA_EDGE}" "${LAMBDA_SPEC}" \
    "${LAMBDA_REQUIRED}" "${LAMBDA_FORBIDDEN}" "${LAMBDA_SYM}" "${LAMBDA_DYN}"

need_volume=0
if is_positive "${LAMBDA_VOLUME}"; then
    need_volume=1
fi

need_sdf=0
if is_positive "${LAMBDA_SHAPE}"; then
    need_sdf=1
fi

need_adj=0
for lambda_val in "${LAMBDA_EDGE}" "${LAMBDA_SPEC}" "${LAMBDA_REQUIRED}" "${LAMBDA_FORBIDDEN}" "${LAMBDA_SYM}" "${LAMBDA_DYN}"; do
    if is_positive "${lambda_val}"; then
        need_adj=1
        break
    fi
done

printf '    prior artifacts needed: volume=%s sdf=%s adjacency=%s\n' \
    "${need_volume}" "${need_sdf}" "${need_adj}"

DYN_START_EPOCH=${DYN_START_EPOCH:-60}
DYN_RAMP_EPOCHS=${DYN_RAMP_EPOCHS:-40}
DYN_MISMATCH_REF=${DYN_MISMATCH_REF:-0.08}
DYN_MAX_SCALE=${DYN_MAX_SCALE:-3.0}
AGE_RELIABILITY_MIN=${AGE_RELIABILITY_MIN:-0.3}
AGE_RELIABILITY_POW=${AGE_RELIABILITY_POW:-0.5}
EVAL_TTA=${EVAL_TTA:-0}
TTA_AXES=${TTA_AXES:-"0"}
EVAL_MULTI_SCALE=${EVAL_MULTI_SCALE:-0}
EVAL_SCALES=${EVAL_SCALES:-"1.0"}
PRECHECK_PRIORS=${PRECHECK_PRIORS:-1}
LATERALITY_PAIRS=${LATERALITY_PAIRS:-}
EPOCH_TIME_BUFFER=${EPOCH_TIME_BUFFER:-600}
SLURM_TIME_BUFFER=${SLURM_TIME_BUFFER:-300}

# ---------- Derived paths ----------
VOLUME_STATS="${TARGET_PRIOR_ROOT}/volume_stats.json"
SDF_TEMPLATES="${TARGET_PRIOR_ROOT}/sdf_templates.npz"
ADJACENCY_PRIOR="${TARGET_PRIOR_ROOT}/adjacency_prior.npz"
RESTRICTED_MASK="${TARGET_PRIOR_ROOT}/R_mask.npy"
STRUCTURAL_RULES="${TARGET_PRIOR_ROOT}/structural_rules.json"

required_files=("${TARGET_SPLIT_JSON}" "${VOLUME_STATS}")
if [ "${DISABLE_PRIOR}" -eq 0 ]; then
    if [ "${need_sdf}" -eq 1 ]; then
        required_files+=("${SDF_TEMPLATES}")
    fi
    if [ "${need_adj}" -eq 1 ]; then
        required_files+=("${ADJACENCY_PRIOR}")
    fi
fi

for required in "${required_files[@]}"; do
    if [ ! -f "$required" ]; then
        echo "Missing required file: $required" >&2
        exit 1
    fi
done

if [ "${DISABLE_PRIOR}" -eq 0 ] && [ "${PRECHECK_PRIORS}" -ne 0 ]; then
    if [ "${need_sdf}" -eq 0 ] && [ "${need_adj}" -eq 0 ]; then
        echo "Skipping prior precheck (volume-only profile)"
    else
        python "${SCRIPT_DIR}/prior_validator.py" \
            --dir "${TARGET_PRIOR_ROOT}" \
            --num-classes "${OUT_CHANNELS}"
    fi
fi

mkdir -p "${RESULTS_DIR}"

RESUME_FILE="${RESULTS_DIR}/resume_from.txt"
if [ -z "${PRETRAINED_CHECKPOINT}" ] && [ -f "${RESUME_FILE}" ]; then
    RESUME_CANDIDATE="$(cat "${RESUME_FILE}")"
    if [ -n "${RESUME_CANDIDATE}" ] && [ -f "${RESUME_CANDIDATE}" ]; then
        PRETRAINED_CHECKPOINT="${RESUME_CANDIDATE}"
        RESUME_FROM="${RESUME_CANDIDATE}"
    fi
fi

if [ -z "${RESUME_FROM:-}" ] && [ -n "${PRETRAINED_CHECKPOINT}" ] && [[ "${PRETRAINED_CHECKPOINT}" == *checkpoint*pt ]]; then
    RESUME_FROM="${PRETRAINED_CHECKPOINT}"
fi

CMD=(
    torchrun --nproc_per_node="${NUM_GPUS}"
    "${SCRIPT_DIR}/train_graphalign_age.py"
    --split_json "${TARGET_SPLIT_JSON}"
    --results_dir "${RESULTS_DIR}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --out_channels "${OUT_CHANNELS}"
    --roi_x "${ROI_X}" --roi_y "${ROI_Y}" --roi_z "${ROI_Z}"
    --feature_size "${FEATURE_SIZE}"
    --lr "${LR}" --weight_decay "${WEIGHT_DECAY}"
    --lr_min "${LR_MIN}"
    --lr_warmup_epochs "${LR_WARMUP_EPOCHS}"
    --lr_warmup_start_factor "${LR_WARMUP_START}"
    --grad_accum_steps "${GRAD_ACCUM_STEPS}"
    --volume_stats "${VOLUME_STATS}"
    --lambda_volume "${LAMBDA_VOLUME}"
    --lambda_shape "${LAMBDA_SHAPE}"
    --lambda_edge "${LAMBDA_EDGE}"
    --lambda_spec "${LAMBDA_SPEC}"
    --lambda_required "${LAMBDA_REQUIRED}"
    --lambda_forbidden "${LAMBDA_FORBIDDEN}"
    --lambda_symmetry "${LAMBDA_SYM}"
    --lambda_dyn "${LAMBDA_DYN}"
    --dyn_start_epoch "${DYN_START_EPOCH}"
    --dyn_ramp_epochs "${DYN_RAMP_EPOCHS}"
    --dyn_mismatch_ref "${DYN_MISMATCH_REF}"
    --dyn_max_scale "${DYN_MAX_SCALE}"
    --age_reliability_min "${AGE_RELIABILITY_MIN}"
    --age_reliability_pow "${AGE_RELIABILITY_POW}"
    --prior_dir "${TARGET_PRIOR_ROOT}"
    --epoch_time_buffer "${EPOCH_TIME_BUFFER}"
    --slurm_time_buffer "${SLURM_TIME_BUFFER}"
)

if [ "${DISABLE_PRIOR}" -eq 0 ] && [ "${need_sdf}" -eq 1 ]; then
    CMD+=(--sdf_templates "${SDF_TEMPLATES}")
fi

if [ "${DISABLE_PRIOR}" -eq 0 ] && [ "${need_adj}" -eq 1 ]; then
    CMD+=(--adjacency_prior "${ADJACENCY_PRIOR}")
    if [ -f "${RESTRICTED_MASK}" ]; then
        CMD+=(--restricted_mask "${RESTRICTED_MASK}")
    fi
    if [ -f "${STRUCTURAL_RULES}" ]; then
        CMD+=(--structural_rules "${STRUCTURAL_RULES}")
    fi
fi

if [ "${DISABLE_PRIOR}" -eq 0 ] && [ "${PRECHECK_PRIORS}" -ne 0 ]; then
    CMD+=(--precheck_priors)
elif [ "${DISABLE_PRIOR}" -eq 0 ]; then
    CMD+=(--skip_prior_check)
fi

if [ "${DISABLE_PRIOR}" -ne 0 ]; then
    CMD+=(--disable_prior)
fi

if [ -n "${RESUME_FROM:-}" ]; then
    CMD+=(--resume "${RESUME_FROM}")
elif [ -n "${PRETRAINED_CHECKPOINT}" ]; then
    CMD+=(--pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
fi

if [ -n "${LATERALITY_PAIRS}" ]; then
    CMD+=(--laterality_pairs_json "${LATERALITY_PAIRS}")
fi

if python - <<PY >/dev/null 2>&1
import sys
try:
    v = float("${EMA_DECAY:-0}")
    sys.exit(0 if v > 0 else 1)
except Exception:
    sys.exit(1)
PY
then
    CMD+=(--ema_decay "${EMA_DECAY}")
fi

if [ "${EVAL_TTA}" -ne 0 ]; then
    CMD+=(--eval_tta --tta_flip_axes ${TTA_AXES})
fi

if [ "${EVAL_MULTI_SCALE}" -ne 0 ]; then
    CMD+=(--multi_scale_eval --eval_scales ${EVAL_SCALES})
fi

if [ "${USE_AMP}" -eq 0 ]; then
    CMD+=(--no_amp)
fi

printf 'Running command:\n  %s\n' "${CMD[*]}"

"${CMD[@]}"
