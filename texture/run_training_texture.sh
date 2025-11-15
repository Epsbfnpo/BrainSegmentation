#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=${1:-"${SCRIPT_DIR}/outputs/$(date -u +%Y%m%dT%H%M%SZ)"}

python "${SCRIPT_DIR}/train_texture.py" \
  --output-dir "${OUTPUT_DIR}" \
  --source-split "${SCRIPT_DIR}/../dHCP_split.json" \
  --target-split "${SCRIPT_DIR}/../PPREMOPREBO_split.json" \
  --epochs 200 \
  --batch-size 2 \
  --val-batch-size 2 \
  --num-workers 6 \
  --cache-rate 0.3 \
  --cache-workers 6 \
  --roi 96 96 96 \
  --lambda-domain 0.5 \
  --lambda-mmd 0.1 \
  --style-dim 128 \
  --style-base-ch 16 \
  --domain-hidden 128 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --max-grad-norm 1.0 \
  --scheduler cosine \
  --amp
