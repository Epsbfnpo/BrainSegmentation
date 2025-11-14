#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------- User configuration ----------
NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-400}
RESULTS_DIR=${RESULTS_DIR:-${REPO_ROOT}/results/target_only}
TARGET_SPLIT_JSON=${TARGET_SPLIT_JSON:-${REPO_ROOT}/PPREMOPREBO_split.json}
TARGET_PRIOR_ROOT=${TARGET_PRIOR_ROOT:-${REPO_ROOT}/new/priors/target}
CLASS_PRIOR_JSON=${CLASS_PRIOR_JSON:-${REPO_ROOT}/PPREMOPREBO_class_prior_foreground.json}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-}

ROI_X=${ROI_X:-128}
ROI_Y=${ROI_Y:-128}
ROI_Z=${ROI_Z:-128}
FEATURE_SIZE=${FEATURE_SIZE:-48}
LR=${LR:-5e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
LAMBDA_VOLUME=${LAMBDA_VOLUME:-0.2}
LAMBDA_SHAPE=${LAMBDA_SHAPE:-0.2}
LAMBDA_EDGE=${LAMBDA_EDGE:-0.1}
LAMBDA_SPEC=${LAMBDA_SPEC:-0.05}

# ---------- Derived paths ----------
VOLUME_STATS="${TARGET_PRIOR_ROOT}/volume_stats.json"
SDF_TEMPLATES="${TARGET_PRIOR_ROOT}/sdf_templates.npz"
ADJACENCY_PRIOR="${TARGET_PRIOR_ROOT}/adjacency_prior.npz"
RESTRICTED_MASK="${TARGET_PRIOR_ROOT}/R_mask.npy"

for required in "${TARGET_SPLIT_JSON}" "${VOLUME_STATS}" "${SDF_TEMPLATES}" "${ADJACENCY_PRIOR}"; do
    if [ ! -f "$required" ]; then
        echo "Missing required file: $required" >&2
        exit 1
    fi
done

mkdir -p "${RESULTS_DIR}"

CMD=(
    torchrun --nproc_per_node="${NUM_GPUS}"
    "${SCRIPT_DIR}/train_graphalign_age.py"
    --split_json "${TARGET_SPLIT_JSON}"
    --results_dir "${RESULTS_DIR}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --roi_x "${ROI_X}" --roi_y "${ROI_Y}" --roi_z "${ROI_Z}"
    --feature_size "${FEATURE_SIZE}"
    --lr "${LR}" --weight_decay "${WEIGHT_DECAY}"
    --class_prior_json "${CLASS_PRIOR_JSON}"
    --volume_stats "${VOLUME_STATS}"
    --sdf_templates "${SDF_TEMPLATES}"
    --adjacency_prior "${ADJACENCY_PRIOR}"
    --lambda_volume "${LAMBDA_VOLUME}"
    --lambda_shape "${LAMBDA_SHAPE}"
    --lambda_edge "${LAMBDA_EDGE}"
    --lambda_spec "${LAMBDA_SPEC}"
)

if [ -f "${RESTRICTED_MASK}" ]; then
    CMD+=(--restricted_mask "${RESTRICTED_MASK}")
fi

if [ -n "${PRETRAINED_CHECKPOINT}" ]; then
    CMD+=(--pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
fi

printf 'Running command:\n  %s\n' "${CMD[*]}"

"${CMD[@]}"
