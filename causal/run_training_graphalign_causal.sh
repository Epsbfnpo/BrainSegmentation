#!/bin/bash
# DAUnet Causal Training Script - Age-Conditioned Cross-Domain Graph Alignment
# Mirrors the production runner in new/ with causal regularization toggles enabled.

# ================= USER CONFIGURATION =================
RESULTS_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/causal/results"
BATCH_SIZE=2
NUM_GPUS=4
EPOCHS=1000

# Learning rate schedule
LR=2e-4
LR_MIN=1e-7
LR_WARMUP=10
LR_RESTARTS=(300 700)

# Time management for chunked runs
JOB_TIME_LIMIT=115  # minutes before hard stop (matches 2h jobs)
TIME_BUFFER=10      # minutes before timeout to checkpoint and exit cleanly

# Laterality swap list
LATERALITY_PAIRS_JSON="/datasets/work/hb-nhmrc-dhcp/work/liu275/causal/dhcp_lr_swap.json"

# ========== TARGET DOMAIN GRAPH PRIORS ==========
TARGET_PRIORS_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/causal/priors/PPREMOPREBO"
TARGET_PRIOR_ADJ_NPY="${TARGET_PRIORS_DIR}/prior_adj.npy"
TARGET_PRIOR_REQUIRED_JSON="${TARGET_PRIORS_DIR}/prior_required.json"
TARGET_PRIOR_FORBIDDEN_JSON="${TARGET_PRIORS_DIR}/prior_forbidden.json"
TARGET_WEIGHTED_ADJ_NPY="${TARGET_PRIORS_DIR}/weighted_adj.npy"
TARGET_VOLUME_STATS_JSON="${TARGET_PRIORS_DIR}/volume_stats.json"
TARGET_AGE_WEIGHTS_JSON="${TARGET_PRIORS_DIR}/age_weights.json"
TARGET_RESTRICTED_MASK_NPY="${TARGET_PRIORS_DIR}/R_mask.npy"

# ========== SOURCE DOMAIN GRAPH PRIORS ==========
SOURCE_PRIORS_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/causal/priors/dHCP"
SOURCE_PRIOR_ADJ_NPY="${SOURCE_PRIORS_DIR}/prior_adj.npy"
SOURCE_PRIOR_REQUIRED_JSON="${SOURCE_PRIORS_DIR}/prior_required.json"
SOURCE_PRIOR_FORBIDDEN_JSON="${SOURCE_PRIORS_DIR}/prior_forbidden.json"
SOURCE_WEIGHTED_ADJ_NPY="${SOURCE_PRIORS_DIR}/weighted_adj.npy"
SOURCE_VOLUME_STATS_JSON="${SOURCE_PRIORS_DIR}/volume_stats.json"
SOURCE_AGE_WEIGHTS_JSON="${SOURCE_PRIORS_DIR}/age_weights.json"
SOURCE_RESTRICTED_MASK_NPY="${SOURCE_PRIORS_DIR}/R_mask.npy"

# ========== GRAPH ALIGNMENT HYPER-PARAMETERS ==========
GRAPH_ALIGN_MODE="joint"
LAMBDA_SPEC_SRC=0.10
LAMBDA_EDGE_SRC=0.10
LAMBDA_SPEC_TGT=0.03
LAMBDA_EDGE_TGT=0.03
LAMBDA_SYM=0.05
LAMBDA_STRUCT=0.05
LAMBDA_DYN=0.2
GRAPH_TOPR=20
GRAPH_WARMUP=10
GRAPH_TEMP=1.0
GRAPH_POOL_KERNEL=3
GRAPH_POOL_STRIDE=2
DYN_TOP_K=12
DYN_START_EPOCH=50
DYN_RAMP_EPOCHS=50
QAP_MISMATCH_G=1.5

# ========== CAUSAL REGULARIZATION SETTINGS ==========
AGE_BIN_MIN=32
AGE_BIN_MAX=46
AGE_BIN_SIZE=2
AGE_BIN_MIN_COUNT=3
IRM_PENALTY_WEIGHT=0.08
IRM_PENALTY_TYPE="vrex"
GRAPH_INVARIANCE_WEIGHT=0.05
GRAPH_INVARIANCE_USE_SPECTRAL=1   # 1 enables spectral component, 0 disables
CI_WEIGHT=0.04
CI_BANDWIDTHS="1.0,5.0,10.0,20.0"
CF_DOMAIN_WEIGHT=0.05
CF_NOISE_SCALE=0.08
CF_AGE_WEIGHT=0.03
CF_AGE_DELTA=0.75

# ========== DATA SPLITS & PRETRAINED MODEL ==========
SOURCE_SPLIT_JSON="/scratch3/liu275/Data/dHCP/dHCP_split.json"
TARGET_SPLIT_JSON="/datasets/work/hb-nhmrc-dhcp/work/liu275/PPREMOPREBO_split.json"
SOURCE_PRIOR_JSON="/datasets/work/hb-nhmrc-dhcp/work/liu275/dHCP_class_prior_standard.json"
TARGET_PRIOR_JSON="/datasets/work/hb-nhmrc-dhcp/work/liu275/PPREMOPREBO_class_prior_standard.json"
PRETRAINED_MODEL="/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/results_fixed/dHCP_registered_fixed/best_model.pth"

# Additional knobs
USE_TTA=1
USE_TARGET_LABELS=1
TARGET_LABEL_WEIGHT=0.8
ENHANCED_CLASS_WEIGHTS=1
NUM_SMALL_CLASSES_BOOST=20
SMALL_CLASS_BOOST_FACTOR=2.0
DICE_DROP_THRESHOLD=0.3
DICE_WINDOW_SIZE=5
EARLY_STOPPING=1
EARLY_STOPPING_PATIENCE=50
USE_TOPK_DICE=1
TOPK_RATIO=0.3
TOPK_WARMUP=30
USE_FOREGROUND_ONLY=1
USE_RESTRICTED_MASK=1

# Optional overrides via environment variables
RESULTS_DIR="${RESULTS_DIR_OVERRIDE:-$RESULTS_DIR}"
BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-$BATCH_SIZE}
NUM_GPUS=${NUM_GPUS_OVERRIDE:-$NUM_GPUS}
EPOCHS=${EPOCHS_OVERRIDE:-$EPOCHS}
LR=${LR_OVERRIDE:-$LR}

# Ensure results directories exist
mkdir -p "$RESULTS_DIR"
mkdir -p "${RESULTS_DIR}/graph_analysis"
mkdir -p "${RESULTS_DIR}/dice_monitoring"

# Helper to build priors if missing (re-uses production builder from new/)
build_priors_if_missing() {
    local PRIOR_ADJ=$1
    local PRIORS_DIR=$2
    local SPLIT_JSON=$3

    if [ -f "$PRIOR_ADJ" ]; then
        return 0
    fi

    echo "🧠 Building graph priors for ${PRIORS_DIR}"
    mkdir -p "$PRIORS_DIR"

    python /datasets/work/hb-nhmrc-dhcp/work/liu275/new/build_graph_priors.py \
        --split_json "$SPLIT_JSON" \
        --out_dir "$PRIORS_DIR" \
        --num_classes 87 \
        --foreground_only \
        --dilate_iter 1 \
        --th_required 0.01 \
        --th_forbidden 1e-5 \
        --lr_pairs_json "$LATERALITY_PAIRS_JSON" \
        --age_bin_width 1.0 \
        --out_volume_stats_json "${PRIORS_DIR}/volume_stats.json" \
        --out_age_weights_json "${PRIORS_DIR}/age_weights.json"

    if [ $? -ne 0 ]; then
        echo "❌ Failed to build priors for ${PRIORS_DIR}"
        exit 1
    fi
    echo "✅ Priors generated for ${PRIORS_DIR}"
}

# Ensure laterality JSON exists
if [ ! -f "$LATERALITY_PAIRS_JSON" ]; then
    echo "📄 Creating laterality pairs JSON at $LATERALITY_PAIRS_JSON"
    mkdir -p "$(dirname "$LATERALITY_PAIRS_JSON")"
    cat > "$LATERALITY_PAIRS_JSON" <<'JSON'
[
  [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16],
  [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
  [31, 32], [33, 34], [35, 36], [37, 38], [39, 40], [41, 42], [43, 44],
  [45, 46], [47, 48], [49, 50], [51, 52], [53, 54], [55, 56], [57, 58],
  [59, 60], [61, 62], [63, 64], [65, 66], [67, 68], [69, 70], [71, 72],
  [73, 74], [75, 76], [77, 78], [79, 80], [81, 82], [83, 84], [85, 86]
]
JSON
fi

# Build priors on demand
build_priors_if_missing "$SOURCE_PRIOR_ADJ_NPY" "$SOURCE_PRIORS_DIR" "$SOURCE_SPLIT_JSON"
build_priors_if_missing "$TARGET_PRIOR_ADJ_NPY" "$TARGET_PRIORS_DIR" "$TARGET_SPLIT_JSON"

# Compose target graph args
TARGET_GRAPH_ARGS=""
[ -f "$TARGET_PRIOR_ADJ_NPY" ] && TARGET_GRAPH_ARGS+=" --prior_adj_npy $TARGET_PRIOR_ADJ_NPY"
[ -f "$TARGET_PRIOR_REQUIRED_JSON" ] && TARGET_GRAPH_ARGS+=" --prior_required_json $TARGET_PRIOR_REQUIRED_JSON"
[ -f "$TARGET_PRIOR_FORBIDDEN_JSON" ] && TARGET_GRAPH_ARGS+=" --prior_forbidden_json $TARGET_PRIOR_FORBIDDEN_JSON"

# Compose source graph args
SOURCE_GRAPH_ARGS=""
[ -f "$SOURCE_PRIOR_ADJ_NPY" ] && SOURCE_GRAPH_ARGS+=" --src_prior_adj_npy $SOURCE_PRIOR_ADJ_NPY"
[ -f "$SOURCE_PRIOR_REQUIRED_JSON" ] && SOURCE_GRAPH_ARGS+=" --src_prior_required_json $SOURCE_PRIOR_REQUIRED_JSON"
[ -f "$SOURCE_PRIOR_FORBIDDEN_JSON" ] && SOURCE_GRAPH_ARGS+=" --src_prior_forbidden_json $SOURCE_PRIOR_FORBIDDEN_JSON"

# Compose target age priors
if [ -f "$TARGET_WEIGHTED_ADJ_NPY" ]; then
    TARGET_GRAPH_ARGS+=" --weighted_adj_npy $TARGET_WEIGHTED_ADJ_NPY"
fi
if [ -f "$TARGET_VOLUME_STATS_JSON" ]; then
    TARGET_GRAPH_ARGS+=" --volume_stats_json $TARGET_VOLUME_STATS_JSON"
fi
if [ -f "$TARGET_AGE_WEIGHTS_JSON" ]; then
    TARGET_GRAPH_ARGS+=" --age_weights_json $TARGET_AGE_WEIGHTS_JSON"
fi

# Additional args placeholder
EXTRA_ARGS="$EXTRA_ARGS"

# Training command
CMD=(
    torchrun --standalone --nproc_per_node=$NUM_GPUS train_graphalign_causal.py
    --debug_mode
    --epochs=$EPOCHS
    --batch_size=$BATCH_SIZE
    --lr=$LR
    --lr_min=$LR_MIN
    --lr_warmup_epochs=$LR_WARMUP
)

for restart in "${LR_RESTARTS[@]}"; do
    CMD+=(--lr_restart_epochs $restart)
done

CMD+=(
    --results_dir "$RESULTS_DIR"
    --in_channels 1
    --out_channels 87
    --feature_size 48
    --roi_x 128 --roi_y 128 --roi_z 128
    --use_label_crop
    --eval_num 2
    --save_interval 10
    --cache_rate 0
    --cache_num_workers 2
    --num_workers 16
    --pretrained_model "$PRETRAINED_MODEL"
    --source_split_json "$SOURCE_SPLIT_JSON"
    --split_json "$TARGET_SPLIT_JSON"
    --target_prior_json "$TARGET_PRIOR_JSON"
    --source_prior_json "$SOURCE_PRIOR_JSON"
    --target_spacing 0.5 0.5 0.5
    --apply_spacing
    --apply_orientation
    --laterality_pairs_json "$LATERALITY_PAIRS_JSON"
    --loss_config dice_focal
    --focal_gamma 2.0
    --dist_timeout 120
    --clip 2.0
    --job_time_limit $JOB_TIME_LIMIT
    --time_buffer_minutes $TIME_BUFFER
    --graph_align_mode $GRAPH_ALIGN_MODE
    --lambda_spec_src $LAMBDA_SPEC_SRC
    --lambda_edge_src $LAMBDA_EDGE_SRC
    --lambda_spec_tgt $LAMBDA_SPEC_TGT
    --lambda_edge_tgt $LAMBDA_EDGE_TGT
    --lambda_sym $LAMBDA_SYM
    --lambda_struct $LAMBDA_STRUCT
    --lambda_dyn $LAMBDA_DYN
    --graph_topr $GRAPH_TOPR
    --graph_warmup_epochs $GRAPH_WARMUP
    --graph_temperature $GRAPH_TEMP
    --graph_pool_kernel $GRAPH_POOL_KERNEL
    --graph_pool_stride $GRAPH_POOL_STRIDE
    --dyn_top_k $DYN_TOP_K
    --dyn_start_epoch $DYN_START_EPOCH
    --dyn_ramp_epochs $DYN_RAMP_EPOCHS
    --qap_mismatch_g $QAP_MISMATCH_G
    --age_bin_min $AGE_BIN_MIN
    --age_bin_max $AGE_BIN_MAX
    --age_bin_size $AGE_BIN_SIZE
    --age_bin_min_count $AGE_BIN_MIN_COUNT
    --irm_penalty_weight $IRM_PENALTY_WEIGHT
    --irm_penalty_type $IRM_PENALTY_TYPE
    --graph_invariance_weight $GRAPH_INVARIANCE_WEIGHT
    --ci_weight $CI_WEIGHT
    --ci_bandwidths $CI_BANDWIDTHS
    --cf_domain_weight $CF_DOMAIN_WEIGHT
    --cf_noise_scale $CF_NOISE_SCALE
    --cf_age_weight $CF_AGE_WEIGHT
    --cf_age_delta $CF_AGE_DELTA
)

if [ "$GRAPH_INVARIANCE_USE_SPECTRAL" -eq 1 ]; then
    CMD+=(--graph_invariance_use_spectral)
fi

if [ $USE_TTA -eq 1 ]; then
    CMD+=(--use_tta)
fi
if [ $USE_TARGET_LABELS -eq 1 ]; then
    CMD+=(--use_target_labels)
fi
if [ $USE_FOREGROUND_ONLY -eq 1 ]; then
    CMD+=(--foreground_only)
fi
if [ $ENHANCED_CLASS_WEIGHTS -eq 1 ]; then
    CMD+=(--enhanced_class_weights)
fi
if [ $USE_TOPK_DICE -eq 1 ]; then
    CMD+=(--use_topk_dice --topk_ratio $TOPK_RATIO --topk_warmup_epochs $TOPK_WARMUP)
fi
if [ $EARLY_STOPPING -eq 1 ]; then
    CMD+=(--early_stopping --early_stopping_patience $EARLY_STOPPING_PATIENCE)
fi
if [ $USE_RESTRICTED_MASK -eq 1 ] && [ -f "$TARGET_RESTRICTED_MASK_NPY" ]; then
    CMD+=(--use_restricted_mask)
fi

CMD+=(
    --target_label_weight $TARGET_LABEL_WEIGHT
    --num_small_classes_boost $NUM_SMALL_CLASSES_BOOST
    --small_class_boost_factor $SMALL_CLASS_BOOST_FACTOR
    --dice_drop_threshold $DICE_DROP_THRESHOLD
    --dice_window_size $DICE_WINDOW_SIZE
)

# Append prior args
CMD+=($TARGET_GRAPH_ARGS)
CMD+=($SOURCE_GRAPH_ARGS)

# Add any user-provided extras
if [ -n "$EXTRA_ARGS" ]; then
    CMD+=($EXTRA_ARGS)
fi

# Run training and tee logs
{
    printf '🚀 Launching causal graph alignment training\n'
    printf 'Command: %q ' "${CMD[@]}"
    printf '\n==============================================================\n'
} | tee -a "${RESULTS_DIR}/training.log"

"${CMD[@]}" 2>&1 | tee -a "${RESULTS_DIR}/training.log"
EXIT_STATUS=${PIPESTATUS[0]}

# Crash log summary
if ls ${RESULTS_DIR}/crash_rank*.log >/dev/null 2>&1; then
    echo "❌ Crash logs detected:" | tee -a "${RESULTS_DIR}/training.log"
    for log in ${RESULTS_DIR}/crash_rank*.log; do
        echo "------------------------------------------------------------" | tee -a "${RESULTS_DIR}/training.log"
        echo "$(basename "$log")" | tee -a "${RESULTS_DIR}/training.log"
        head -n 40 "$log" | tee -a "${RESULTS_DIR}/training.log"
    done
fi

if [ $EXIT_STATUS -eq 0 ] && [ -f "${RESULTS_DIR}/final_model.pth" ]; then
    echo "✅ Training completed successfully." | tee -a "${RESULTS_DIR}/training.log"
else
    echo "⚠️ Training exited with status $EXIT_STATUS" | tee -a "${RESULTS_DIR}/training.log"
fi

exit $EXIT_STATUS
