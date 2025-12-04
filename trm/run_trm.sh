#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_GPUS=${NUM_GPUS:-1}
SPLIT_JSON=${SPLIT_JSON:-"${SCRIPT_DIR}/../PPREMOPREBO_split.json"}
RESULTS_DIR=${RESULTS_DIR:-"${SCRIPT_DIR}/../trm_runs"}
PRETRAINED=${PRETRAINED:-"/path/to/source_checkpoint.ckpt"}

CMD=(
    torchrun --nproc_per_node="${NUM_GPUS}"
    "${SCRIPT_DIR}/train_trm.py"
    --split_json "${SPLIT_JSON}"
    --results_dir "${RESULTS_DIR}"
    --pretrained_checkpoint "${PRETRAINED}"
    --apply_spacing
    --apply_orientation
    --foreground_only
)

echo "Running: ${CMD[@]} $@"
"${CMD[@]}" "$@"
