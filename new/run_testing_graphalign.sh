#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Example command to run target-only evaluation on the held-out test split.
# Update GPU visibility as needed before invoking this script.

python -u "${SCRIPT_DIR}/test_graphalign_age.py" \
  --split_json "${SCRIPT_DIR}/../PPREMOPREBO_split_test.json" \
  --model_path /datasets/work/hb-nhmrc-dhcp/work/liu275/new/results/target_only/best_model.pt \
  --output_dir "${SCRIPT_DIR}/test_predictions" \
  --metrics_path "${SCRIPT_DIR}/analysis/test_metrics.json" \
  --in_channels 1 \
  --out_channels 87 \
  --feature_size 48 \
  --roi_x 96 --roi_y 96 --roi_z 96 \
  --target_spacing 0.8 0.8 0.8 \
  --foreground_only \
  --sw_batch_size 1 \
  --sw_overlap 0.25 \
  --eval_tta \
  --tta_flip_axes 0 1 2
