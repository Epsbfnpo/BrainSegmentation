#!/bin/bash
# DAUnet Training Script - WITH CROSS-DOMAIN GRAPH ALIGNMENT
# Aligns target domain predictions to source domain graph structure
# WITH AUTO-RESUME SUPPORT AND SMART TIME MANAGEMENT

# Paths relative to this repository
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
RESULTS_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/new/results"
BATCH_SIZE=2
NUM_GPUS=4
EPOCHS=1000

# Time management configuration
JOB_TIME_LIMIT=115  # 115 minutes for 2-hour jobs (leaving 5 min buffer)
TIME_BUFFER=10      # 10 minutes buffer before job ends (extra safety)

# Laterality pairs JSON file path
LATERALITY_PAIRS_JSON="${SCRIPT_DIR}/dhcp_lr_swap.json"

SOURCE_CLASS_PRIOR="${REPO_ROOT}/dHCP_class_prior_foreground.json"
TARGET_CLASS_PRIOR="${REPO_ROOT}/PPREMOPREBO_class_prior_foreground.json"

# Shape template configuration (auto-generated if missing)
DATA_ROOT="${BRAINSEG_DATA_ROOT:-/datasets/work/hb-nhmrc-dhcp/work/liu275}"
SHAPE_TEMPLATES_PT="${REPO_ROOT}/priors/shape_templates.pt"
DHCPSPLIT_JSON="/scratch3/liu275/Data/dHCP/dHCP_split.json"
if [ ! -f "$DHCPSPLIT_JSON" ]; then
    DHCPSPLIT_JSON="${DATA_ROOT}/dHCP_split.json"
fi
DHCPSPLIT_TEST_JSON="/scratch3/liu275/Data/dHCP/dHCP_split_test.json"
if [ ! -f "$DHCPSPLIT_TEST_JSON" ]; then
    DHCPSPLIT_TEST_JSON="${DATA_ROOT}/dHCP_split_test.json"
fi
PPREMOPREBO_SPLIT_JSON="${DATA_ROOT}/PPREMOPREBO_split.json"
PPREMOPREBO_SPLIT_TEST_JSON="${DATA_ROOT}/PPREMOPREBO_split_test.json"

# ========== TARGET DOMAIN GRAPH PRIORS ==========
TARGET_PRIORS_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/new/priors/PPREMOPREBO"
TARGET_PRIOR_ADJ_NPY="${TARGET_PRIORS_DIR}/prior_adj.npy"
TARGET_PRIOR_REQUIRED_JSON="${TARGET_PRIORS_DIR}/prior_required.json"
TARGET_PRIOR_FORBIDDEN_JSON="${TARGET_PRIORS_DIR}/prior_forbidden.json"
TARGET_WEIGHTED_ADJ_NPY="${TARGET_PRIORS_DIR}/weighted_adj.npy"
TARGET_VOLUME_STATS_JSON="${TARGET_PRIORS_DIR}/volume_stats.json"
TARGET_AGE_WEIGHTS_JSON="${TARGET_PRIORS_DIR}/age_weights.json"
SHAPE_TEMPLATES_PATH="/datasets/work/hb-nhmrc-dhcp/work/liu275/new/priors/shape_templates.pt"
SHAPE_TEMPLATE_ARGS="--shape_templates_pt ${SHAPE_TEMPLATES_PATH}"

SHAPE_TEMPLATES_PATH="/datasets/work/hb-nhmrc-dhcp/work/liu275/new/priors/shape_templates.pt"
SHAPE_TEMPLATE_ARGS="--shape_templates_pt ${SHAPE_TEMPLATES_PATH}"

SHAPE_TEMPLATES_PATH="/datasets/work/hb-nhmrc-dhcp/work/liu275/new/priors/shape_templates.pt"
SHAPE_TEMPLATE_ARGS="--shape_templates_pt ${SHAPE_TEMPLATES_PATH}"


# ========== SOURCE DOMAIN GRAPH PRIORS (NEW) ==========
SOURCE_PRIORS_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/new/priors/dHCP"
SOURCE_PRIOR_ADJ_NPY="${SOURCE_PRIORS_DIR}/prior_adj.npy"
SOURCE_PRIOR_REQUIRED_JSON="${SOURCE_PRIORS_DIR}/prior_required.json"
SOURCE_PRIOR_FORBIDDEN_JSON="${SOURCE_PRIORS_DIR}/prior_forbidden.json"
SOURCE_WEIGHTED_ADJ_NPY="${SOURCE_PRIORS_DIR}/weighted_adj.npy"
SOURCE_VOLUME_STATS_JSON="${SOURCE_PRIORS_DIR}/volume_stats.json"
SOURCE_AGE_WEIGHTS_JSON="${SOURCE_PRIORS_DIR}/age_weights.json"


# ========== CROSS-DOMAIN ALIGNMENT CONFIGURATION ==========
GRAPH_ALIGN_MODE="joint"    # 'src_only', 'tgt_only', or 'joint'
LAMBDA_SPEC_SRC=0.1         # Weight for source domain spectral alignment (primary)
LAMBDA_EDGE_SRC=0.1         # Weight for source domain edge consistency (primary)
LAMBDA_SPEC_TGT=0.03        # Weight for target domain spectral alignment (regularization)
LAMBDA_EDGE_TGT=0.03        # Weight for target domain edge consistency (regularization)
LAMBDA_SYM=0.05             # Weight for symmetry consistency
GRAPH_TOPR=20               # Number of eigenvalues for spectral alignment
GRAPH_WARMUP=10             # Warmup epochs for graph loss
GRAPH_TEMP=1.0              # Temperature for adjacency computation

# Create results directory
mkdir -p $RESULTS_DIR
mkdir -p "${RESULTS_DIR}/graph_analysis"
mkdir -p "$(dirname "$SHAPE_TEMPLATES_PT")"

# Build shape templates if missing
if [ ! -f "$SHAPE_TEMPLATES_PT" ]; then
    echo "🧩 Building age-aware shape templates..."
    missing_split=0
    for split_path in "$DHCPSPLIT_JSON" "$DHCPSPLIT_TEST_JSON" "$PPREMOPREBO_SPLIT_JSON" "$PPREMOPREBO_SPLIT_TEST_JSON"; do
        if [ ! -f "$split_path" ]; then
            echo "❌ Required split file not found: $split_path"
            missing_split=1
        fi
    done
    if [ $missing_split -ne 0 ]; then
        echo "Aborting shape template generation due to missing splits."
        exit 1
    fi
    python build_shape_templates.py \
        --split "$DHCPSPLIT_JSON" \
        --split "$DHCPSPLIT_TEST_JSON" \
        --split "$PPREMOPREBO_SPLIT_JSON" \
        --split "$PPREMOPREBO_SPLIT_TEST_JSON" \
        --data-root "$DATA_ROOT" \
        --num-classes 87 \
        --age-bin-width 2.0 \
        --workers 32 \
        --device cpu \
        --target-shape 239x290x290 \
        --output "$SHAPE_TEMPLATES_PT"

    if [ $? -ne 0 ]; then
        echo "❌ Failed to build shape templates"
        exit 1
    fi
    echo "✅ Shape templates saved to $SHAPE_TEMPLATES_PT"
fi

# Build SOURCE domain graph priors if they don't exist
if [ ! -f "$SOURCE_PRIOR_ADJ_NPY" ]; then
    echo "🧠 Building graph priors from SOURCE domain (dHCP)..."
    mkdir -p "${SOURCE_PRIORS_DIR}"

    python build_graph_priors.py \
        --split_json /datasets/work/hb-nhmrc-dhcp/work/liu275/dHCP_split.json \
        --out_dir "${SOURCE_PRIORS_DIR}" \
        --num_classes 87 \
        --foreground_only \
        --dilate_iter 1 \
        --th_required 0.01 \
        --th_forbidden 1e-5 \
        --lr_pairs_json "$LATERALITY_PAIRS_JSON" \
        --age_bin_width 1.0 \
        --out_volume_stats_json "${SOURCE_PRIORS_DIR}/volume_stats.json" \
        --out_age_weights_json  "${SOURCE_PRIORS_DIR}/age_weights.json"


    if [ $? -ne 0 ]; then
        echo "❌ Failed to build source domain graph priors"
        exit 1
    fi
    echo "✅ Source domain graph priors built successfully"
fi

# Build TARGET domain graph priors if they don't exist
if [ ! -f "$TARGET_PRIOR_ADJ_NPY" ]; then
    echo "🧠 Building graph priors from TARGET domain (PPREMO/PREBO)..."
    mkdir -p "${TARGET_PRIORS_DIR}"

    python build_graph_priors.py \
        --split_json /datasets/work/hb-nhmrc-dhcp/work/liu275/PPREMOPREBO_split.json \
        --out_dir "${TARGET_PRIORS_DIR}" \
        --num_classes 87 \
        --foreground_only \
        --dilate_iter 1 \
        --th_required 0.01 \
        --th_forbidden 1e-5 \
        --lr_pairs_json "$LATERALITY_PAIRS_JSON" \
        --age_bin_width 1.0 \
        --out_volume_stats_json "${TARGET_PRIORS_DIR}/volume_stats.json" \
        --out_age_weights_json  "${TARGET_PRIORS_DIR}/age_weights.json"


    if [ $? -ne 0 ]; then
        echo "❌ Failed to build target domain graph priors"
        exit 1
    fi
    echo "✅ Target domain graph priors built successfully"
fi

# Create laterality pairs JSON if it doesn't exist
if [ ! -f "$LATERALITY_PAIRS_JSON" ]; then
    echo "Creating laterality pairs JSON file..."
    cat > "$LATERALITY_PAIRS_JSON" << 'EOF'
[
  [1, 2],
  [3, 4],
  [5, 6],
  [7, 8],
  [9, 10],
  [11, 12],
  [13, 14],
  [15, 16],
  [17, 18],
  [21, 20],
  [23, 22],
  [25, 24],
  [27, 26],
  [29, 28],
  [31, 30],
  [33, 32],
  [35, 34],
  [37, 36],
  [39, 38],
  [41, 40],
  [43, 42],
  [45, 44],
  [47, 46],
  [49, 50],
  [51, 52],
  [53, 54],
  [55, 56],
  [57, 58],
  [59, 60],
  [61, 62],
  [64, 63],
  [66, 65],
  [68, 67],
  [70, 69],
  [72, 71],
  [74, 73],
  [76, 75],
  [78, 77],
  [80, 79],
  [82, 81],
  [87, 86]
]
EOF
    echo "Created example laterality pairs at: $LATERALITY_PAIRS_JSON"
fi

echo "=============================================================="
echo "DAUnet TRAINING WITH CROSS-DOMAIN GRAPH ALIGNMENT"
echo "=============================================================="
echo "Configuration:"
echo "  - Total Epochs: $EPOCHS"
echo "  - Batch Size: $BATCH_SIZE per GPU"
echo "  - Number of GPUs: $NUM_GPUS"
echo "  - Results Directory: $RESULTS_DIR"
echo "  - Job Time Limit: $JOB_TIME_LIMIT minutes"
echo "  - Time Buffer: $TIME_BUFFER minutes"
echo ""
echo "🎯 CROSS-DOMAIN ALIGNMENT CONFIGURATION:"
echo "  - Alignment Mode: $GRAPH_ALIGN_MODE"
echo ""
echo "  SOURCE DOMAIN (dHCP) - Primary Target:"
echo "    - Adjacency matrix: $SOURCE_PRIOR_ADJ_NPY"
echo "    - λ_spec (source): $LAMBDA_SPEC_SRC"
echo "    - λ_edge (source): $LAMBDA_EDGE_SRC"
echo ""
echo "  TARGET DOMAIN (PPREMO/PREBO) - Regularization:"
echo "    - Adjacency matrix: $TARGET_PRIOR_ADJ_NPY"
echo "    - λ_spec (target): $LAMBDA_SPEC_TGT"
echo "    - λ_edge (target): $LAMBDA_EDGE_TGT"
echo ""
echo "  COMMON PARAMETERS:"
echo "    - λ_sym: $LAMBDA_SYM"
echo "    - Top-K eigenvalues: $GRAPH_TOPR"
echo "    - Warmup epochs: $GRAPH_WARMUP"
echo "    - Temperature: $GRAPH_TEMP"
echo "    - Laterality Pairs: $LATERALITY_PAIRS_JSON"
echo "    - Shape templates: $SHAPE_TEMPLATES_PT"
echo ""

# Check if training is already complete
if [ -f "${RESULTS_DIR}/final_model.pth" ]; then
    echo "🎉 Training already complete! Found final_model.pth"
    echo "   No need to continue training."
    exit 0
fi

# Check for existing checkpoints to resume from
LATEST_CHECKPOINT=$(ls -1t "${RESULTS_DIR}"/checkpoint_epoch_*.pth 2>/dev/null | head -n1)
LATEST_SAVE=$(ls -1t "${RESULTS_DIR}"/latest.pth 2>/dev/null | head -n1)
EXTRA_ARGS=""

# Prefer latest.pth if it exists and is newer
if [ -n "$LATEST_SAVE" ] && [ -f "$LATEST_SAVE" ]; then
    if [ -n "$LATEST_CHECKPOINT" ]; then
        if [ "$LATEST_SAVE" -nt "$LATEST_CHECKPOINT" ]; then
            RESUME_FROM="$LATEST_SAVE"
        else
            RESUME_FROM="$LATEST_CHECKPOINT"
        fi
    else
        RESUME_FROM="$LATEST_SAVE"
    fi
elif [ -n "$LATEST_CHECKPOINT" ]; then
    RESUME_FROM="$LATEST_CHECKPOINT"
else
    RESUME_FROM=""
fi

if [ -n "$RESUME_FROM" ]; then
    echo "🔄 AUTO-RESUME ENABLED"
    echo "   Resuming from: $RESUME_FROM"
    if [[ "$RESUME_FROM" == *"checkpoint_epoch_"* ]]; then
        EPOCH_NUM=$(echo "$RESUME_FROM" | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1/')
        echo "   Starting from epoch: $((EPOCH_NUM + 1))"
    fi
    EXTRA_ARGS="--resume_training --checkpoint $RESUME_FROM"
else
    echo "🆕 Starting fresh training (no checkpoints found)"
fi

echo ""
echo "🔧 KEY FEATURES:"
echo "  ✔ CROSS-DOMAIN graph structure alignment (dHCP → PPREMO/PREBO)"
echo "  ✔ Dual graph priors: source (primary) + target (regularization)"
echo "  ✔ Class-weighted edge alignment for small structures"
echo "  ✔ Spectral alignment of relation graphs"
echo "  ✔ Edge-level consistency constraints"
echo "  ✔ Symmetry preservation for laterality pairs"
echo "  ✔ Structural violation tracking"
echo "  ✔ Smart time management"
echo "  ✔ Auto-resume from latest checkpoint"
echo "  ✔ Laterality-aware augmentations and TTA"
echo ""
echo "📊 TRAINING STRATEGY:"
echo "  ✔ Source domain graph as primary alignment target"
echo "  ✔ Target domain graph as weak regularization"
echo "  ✔ Graph loss warmup for stable training"
echo "  ✔ Top-K Dice with warm-up (30 epochs)"
echo "  ✔ Class weights in Focal Loss and edge alignment"
echo "  ✔ Weighted sampling for small classes"
echo "  ✔ Worst-10 average for model selection"
echo "=============================================================="

# Export environment variables for H100 optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export TORCH_ALLOW_TF32=1
export CUBLAS_ALLOW_TF32=1

# Enhanced error capture
export TORCHELASTIC_ERROR_FILE="${RESULTS_DIR}/elastic_error.json"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1

echo ""
echo "🔍 ENHANCED ERROR CAPTURE ENABLED"
echo "  - Elastic error file: $TORCHELASTIC_ERROR_FILE"
echo ""

# Clean up old error logs before starting
rm -f ${RESULTS_DIR}/crash_rank*.log
rm -f ${RESULTS_DIR}/elastic_error.json

echo "🚀 Starting DAUnet Training with Cross-Domain Graph Alignment"
if [ -n "$RESUME_FROM" ]; then
    echo "   Resuming from checkpoint..."
fi
echo ""

# Prepare target domain graph prior arguments
TARGET_GRAPH_ARGS=""
if [ -f "$TARGET_PRIOR_ADJ_NPY" ]; then
    TARGET_GRAPH_ARGS="$TARGET_GRAPH_ARGS --prior_adj_npy $TARGET_PRIOR_ADJ_NPY"
fi
if [ -f "$TARGET_PRIOR_REQUIRED_JSON" ]; then
    TARGET_GRAPH_ARGS="$TARGET_GRAPH_ARGS --prior_required_json $TARGET_PRIOR_REQUIRED_JSON"
fi
if [ -f "$TARGET_PRIOR_FORBIDDEN_JSON" ]; then
    TARGET_GRAPH_ARGS="$TARGET_GRAPH_ARGS --prior_forbidden_json $TARGET_PRIOR_FORBIDDEN_JSON"
fi

# Prepare source domain graph prior arguments (NEW)
SOURCE_GRAPH_ARGS=""
if [ -f "$SOURCE_PRIOR_ADJ_NPY" ]; then
    SOURCE_GRAPH_ARGS="$SOURCE_GRAPH_ARGS --src_prior_adj_npy $SOURCE_PRIOR_ADJ_NPY"
fi
if [ -f "$SOURCE_PRIOR_REQUIRED_JSON" ]; then
    SOURCE_GRAPH_ARGS="$SOURCE_GRAPH_ARGS --src_prior_required_json $SOURCE_PRIOR_REQUIRED_JSON"
fi
if [ -f "$SOURCE_PRIOR_FORBIDDEN_JSON" ]; then
    SOURCE_GRAPH_ARGS="$SOURCE_GRAPH_ARGS --src_prior_forbidden_json $SOURCE_PRIOR_FORBIDDEN_JSON"
fi

# Run training with cross-domain alignment
torchrun --standalone --nproc_per_node=$NUM_GPUS train_graphalign_age.py \
    --debug_mode \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --lr=2e-4 \
    --lr_min=1e-7 \
    --lr_warmup_epochs=10 \
    --lr_restart_epochs 300 700 \
    --results_dir=$RESULTS_DIR \
    --in_channels=1 \
    --out_channels=87 \
    --weighted_adj_npy=$TARGET_WEIGHTED_ADJ_NPY \
    --volume_stats_json=$TARGET_VOLUME_STATS_JSON \
    --shape_templates_pt=$SHAPE_TEMPLATES_PT \
    --age_weights_json=$TARGET_AGE_WEIGHTS_JSON \
    --feature_size=48 \
    --roi_x=128 --roi_y=128 --roi_z=128 \
    --use_label_crop \
    --eval_num=2 \
    --save_interval=10 \
    --cache_rate=0 \
    --cache_num_workers=2 \
    --num_workers=16 \
    --pretrained_model=/datasets/work/hb-nhmrc-dhcp/work/liu275/model_final.pt \
    --source_split_json=/datasets/work/hb-nhmrc-dhcp/work/liu275/dHCP_split.json \
    --split_json=/datasets/work/hb-nhmrc-dhcp/work/liu275/PPREMOPREBO_split.json \
    --target_prior_json="$TARGET_CLASS_PRIOR" \
    --source_prior_json="$SOURCE_CLASS_PRIOR" \
    --target_spacing 0.5 0.5 0.5 \
    --apply_spacing \
    --apply_orientation \
    --laterality_pairs_json=$LATERALITY_PAIRS_JSON \
    --foreground_only \
    --loss_config dice_focal \
    --focal_gamma 2.0 \
    --use_topk_dice \
    --topk_ratio 0.3 \
    --topk_warmup_epochs 30 \
    --use_tta \
    --infer_overlap 0.7 \
    --use_amp \
    --amp_dtype bfloat16 \
    --use_target_labels \
    --target_label_weight 0.8 \
    --enhanced_class_weights \
    --num_small_classes_boost 20 \
    --small_class_boost_factor 2.0 \
    --dice_drop_threshold 0.3 \
    --dice_window_size 5 \
    --early_stopping \
    --early_stopping_patience 50 \
    --dist_timeout 120 \
    --clip 2.0 \
    --job_time_limit=$JOB_TIME_LIMIT \
    --time_buffer_minutes=$TIME_BUFFER \
    $SHAPE_TEMPLATE_ARGS \
    $TARGET_GRAPH_ARGS \
    $SOURCE_GRAPH_ARGS \
    --graph_align_mode=$GRAPH_ALIGN_MODE \
    --lambda_spec_src=$LAMBDA_SPEC_SRC \
    --lambda_edge_src=$LAMBDA_EDGE_SRC \
    --lambda_spec_tgt=$LAMBDA_SPEC_TGT \
    --lambda_edge_tgt=$LAMBDA_EDGE_TGT \
    --lambda_sym=$LAMBDA_SYM \
    --graph_topr=$GRAPH_TOPR \
    --graph_warmup_epochs=$GRAPH_WARMUP \
    --graph_temperature=$GRAPH_TEMP \
    --graph_pool_kernel=3 \
    --graph_pool_stride=2 \
    $EXTRA_ARGS \
    2>&1 | tee -a ${RESULTS_DIR}/training.log

# Check exit status
EXIT_STATUS=$?

echo ""
echo "=============================================================="

# Check for crash logs
echo ""
echo "🔍 Checking for crash logs..."
CRASH_LOGS=$(ls -1 ${RESULTS_DIR}/crash_rank*.log 2>/dev/null)
if [ -n "$CRASH_LOGS" ]; then
    echo "❌ CRASH LOGS FOUND!"
    echo "=============================================================="
    for log in $CRASH_LOGS; do
        echo ""
        echo "📄 Content of $log:"
        echo "------------------------------------------------------------"
        head -50 "$log"
        echo "------------------------------------------------------------"
        echo "(Showing first 50 lines, see full log at: $log)"
    done
    echo "=============================================================="
fi

if [ -f "${RESULTS_DIR}/elastic_error.json" ]; then
    echo ""
    echo "📄 Elastic error details found:"
    echo "=============================================================="
    python -c "
import json
try:
    with open('${RESULTS_DIR}/elastic_error.json', 'r') as f:
        error = json.load(f)
    print('Root cause:', error.get('message', 'N/A'))
    if 'failures' in error:
        for failure in error['failures'][:2]:
            print(f\"\\nRank {failure.get('rank', 'N/A')}: {failure.get('message', 'N/A')}\")
            if 'traceback' in failure:
                print('Traceback preview:', failure['traceback'][:500])
except:
    import traceback
    traceback.print_exc()
" 2>/dev/null || cat "${RESULTS_DIR}/elastic_error.json" | head -20
    echo "=============================================================="
fi

if [ $EXIT_STATUS -eq 0 ]; then
    # Check if training is complete
    if [ -f "${RESULTS_DIR}/final_model.pth" ]; then
        echo "✅ TRAINING COMPLETE!"
        echo "=============================================================="
        echo ""
        echo "📊 Results:"
        echo "  - Best model: ${RESULTS_DIR}/best_model.pth"
        echo "  - Final model: ${RESULTS_DIR}/final_model.pth"
        echo "  - Training log: ${RESULTS_DIR}/training.log"
        echo "  - Dice monitoring: ${RESULTS_DIR}/dice_monitoring/"
        echo "  - Graph analysis: ${RESULTS_DIR}/graph_analysis/"

        # Generate cross-domain alignment report
        if [ -f "${RESULTS_DIR}/dice_monitoring/dice_history.json" ]; then
            python -c "
import json
import numpy as np

try:
    with open('${RESULTS_DIR}/dice_monitoring/dice_history.json', 'r') as f:
        history = json.load(f)

    print('\\n🎯 CROSS-DOMAIN ALIGNMENT SUMMARY:')
    print('=' * 60)

    # Check for source/target alignment metrics
    if 'graph_spec_src' in history:
        src_spec = history['graph_spec_src']
        if src_spec:
            print(f'Source Spectral Loss: {np.mean(src_spec[-10:]):.4f} (last 10 epochs)')

    if 'graph_edge_src' in history:
        src_edge = history['graph_edge_src']
        if src_edge:
            print(f'Source Edge Loss: {np.mean(src_edge[-10:]):.4f} (last 10 epochs)')

    if 'spectral_distance_src' in history:
        spec_dist = history['spectral_distance_src']
        if spec_dist:
            print(f'Spectral Distance to Source: {spec_dist[-1]:.4f} (final)')

    print('=' * 60)
except:
    pass
" 2>/dev/null
        fi
    else
        echo "⏸ TRAINING PAUSED (checkpoint saved)"
        echo "=============================================================="
        echo ""
        echo "Training was gracefully paused (time limit or signal)."
        echo "Next run will automatically resume from the latest checkpoint."

        if [ -f "${RESULTS_DIR}/latest.pth" ]; then
            echo "Latest checkpoint: ${RESULTS_DIR}/latest.pth"
            echo "Modification time: $(stat -c %y ${RESULTS_DIR}/latest.pth)"
        fi
    fi
else
    echo "❌ TRAINING FAILED (exit code: $EXIT_STATUS)"
    echo "=============================================================="
    echo ""
    echo "Check the following for errors:"
    echo "1. Training log: ${RESULTS_DIR}/training.log"
    echo "2. Crash logs: ${RESULTS_DIR}/crash_rank*.log"
    echo "3. Elastic error: ${RESULTS_DIR}/elastic_error.json"

    # Still try to resubmit if there's a valid checkpoint
fi

echo ""
echo "📋 Analysis Commands:"
echo ""
echo "1. Check cross-domain alignment progress:"
echo "   grep 'Source\\|Target\\|graph_spec_src\\|graph_edge_src' ${RESULTS_DIR}/training.log | tail -20"
echo ""
echo "2. Compare source vs target alignment losses:"
echo "   python -c \"import json; h=json.load(open('${RESULTS_DIR}/dice_monitoring/dice_history.json')); import numpy as np; print(f'Src Spec: {np.mean(h.get(\\\"graph_spec_src\\\",[-1])[-10:]):.4f}'); print(f'Tgt Spec: {np.mean(h.get(\\\"graph_spec_tgt\\\",[-1])[-10:]):.4f}')\" 2>/dev/null"
echo ""
echo "3. Check structural consistency:"
echo "   grep 'Structural\\|symmetry\\|adjacency' ${RESULTS_DIR}/training.log | tail -10"
echo ""
echo "=============================================================="

exit $EXIT_STATUS
