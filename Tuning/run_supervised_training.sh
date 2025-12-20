#!/bin/bash
# Supervised fine-tuning on registered dHCP dataset with all fixes
# Fixed: Proper foreground mask, LR swap path, rotation angle, sliding window

# Configuration
EXPERIMENT_NAME="dHCP_registered_fixed"
EPOCHS=150
BATCH_SIZE=4
ACCUMULATION_STEPS=4
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4

# Enhanced differential learning rates
ENCODER_LR_RATIO=0.3
FREEZE_ENCODER_EPOCHS=5
DYNAMIC_ENCODER_LR=true
ENCODER_LR_BOOST_EPOCH=50
ENCODER_LR_BOOST_RATIO=0.5

# Model parameters
ROI_SIZE=128
FEATURE_SIZE=48
NUM_CLASSES=87  # Foreground only: 87 brain regions
DROPOUT_RATE=0.1

# Enhanced Loss function
LOSS_TYPE="tversky_focal"
DICE_WEIGHT=0.5
CE_WEIGHT=0.5
FOCAL_GAMMA=2.5

# Balanced Tversky parameters
TVERSKY_ALPHA=0.5
TVERSKY_BETA=0.5

# Enhanced class weight parameters
WEIGHT_POWER=0.75
MAX_WEIGHT=20.0
MIN_WEIGHT=0.1

# Data parameters
DATA_SPLIT_JSON="/scratch3/liu275/Data/dHCP_registered/dHCP_split.json"
CLASS_PRIOR_JSON="/datasets/work/hb-nhmrc-dhcp/work/liu275/dHCP_registered_class_prior_standard.json"
LATERALITY_PAIRS_JSON="/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/dhcp_lr_swap.json"
PRETRAINED_MODEL="/datasets/work/hb-nhmrc-dhcp/work/liu275/SSL/results_ssl/dHCP_distributed/best_model.pth"

# Enhanced sampling
CLASS_AWARE_SAMPLING=true
USE_PERSISTENT_DATASET=false
CACHE_RATE=1.0
CACHE_DIR="./cache_registered"
CLEAN_CACHE=false

# Target spacing for registered data
TARGET_SPACING="0.5 0.5 0.5"

# Augmentation parameters
MAX_ROTATION_ANGLE=0.1  # Radians (~6 degrees)

# System settings
NUM_WORKERS=16

# Learning rate scheduler
LR_SCHEDULER="cosine_no_restart"
LR_MIN=3e-6
LR_PATIENCE=8
LR_FACTOR=0.5

# Training settings
CLIP=1.0
VAL_INTERVAL=10
SAVE_INTERVAL=10
EARLY_STOPPING_PATIENCE=40

# Advanced features
USE_AMP=true
AUTO_BATCH_SIZE=false
VISUALIZE_WEIGHTS=true
USE_POST_PROCESSING=false
USE_EMA=false
EMA_DECAY=0.999
USE_TTA=false  # Test-time augmentation

# Enhanced sliding window inference
SLIDING_WINDOW_BATCH_SIZE=4
OVERLAP=0.7
ALWAYS_USE_SLIDING_WINDOW=true  # New: Force sliding window in validation

# Debugging
DEBUG_SAMPLES=0

# Results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/Tuning/results_fixed"

# Create results directory
mkdir -p $RESULTS_DIR

# Check if LR swap file exists
if [ ! -f "$LATERALITY_PAIRS_JSON" ]; then
    echo "⚠️  WARNING: LR swap file not found: $LATERALITY_PAIRS_JSON"
    echo "   LR flip augmentation will be disabled"
fi

# Save configuration
cat > $RESULTS_DIR/config.txt << EOF
Supervised Fine-tuning Configuration (Registered dHCP - FIXED)
=====================================================
Experiment: $EXPERIMENT_NAME
Timestamp: $TIMESTAMP

Dataset: dHCP Registered (T2w with labels)
Classes: $NUM_CLASSES (brain regions 1-87, background=-1)
Target spacing: $TARGET_SPACING mm

FIXES APPLIED:
- Foreground mask: label >= 0 (includes class 0)
- LR swap file: $LATERALITY_PAIRS_JSON
- Max rotation: $MAX_ROTATION_ANGLE rad (~$(echo "$MAX_ROTATION_ANGLE * 180 / 3.14159" | bc -l | cut -c1-5)°)
- Always use sliding window: $ALWAYS_USE_SLIDING_WINDOW
- Validation: No center crop (enables sliding window)

Key Features:
- Foreground-aware normalization (fixed)
- LR symmetric flipping with label swapping
- Class-aware patch sampling
- Balanced Tversky loss (α=$TVERSKY_ALPHA, β=$TVERSKY_BETA)
- Higher class weights (max=$MAX_WEIGHT)
- Dynamic encoder LR adjustment
- Increased sliding window overlap ($OVERLAP)
- Cosine LR without restarts

Training Parameters:
- Epochs: $EPOCHS
- Batch Size: $BATCH_SIZE (×$ACCUMULATION_STEPS = $((BATCH_SIZE * ACCUMULATION_STEPS)) effective)
- Learning Rate: $LEARNING_RATE
- Encoder LR Ratio: $ENCODER_LR_RATIO (boost to $ENCODER_LR_BOOST_RATIO at epoch $ENCODER_LR_BOOST_EPOCH)
- Weight Decay: $WEIGHT_DECAY
- LR Scheduler: $LR_SCHEDULER (min=$LR_MIN)
- Gradient Clipping: $CLIP
- ROI Size: ${ROI_SIZE}×${ROI_SIZE}×${ROI_SIZE}
- Max Rotation: $MAX_ROTATION_ANGLE rad
- Dropout Rate: $DROPOUT_RATE
- Mixed Precision: $USE_AMP

Loss Function:
- Type: $LOSS_TYPE
- Tversky Weight: $DICE_WEIGHT
- Focal Weight: $CE_WEIGHT
- Tversky Alpha: $TVERSKY_ALPHA
- Tversky Beta: $TVERSKY_BETA
- Focal Gamma: $FOCAL_GAMMA
- Class Weight Power: $WEIGHT_POWER
- Max Weight: $MAX_WEIGHT

Data:
- Split JSON: $DATA_SPLIT_JSON
- Class Prior: $CLASS_PRIOR_JSON
- LR Pairs: $LATERALITY_PAIRS_JSON
- SSL Pretrained: $PRETRAINED_MODEL
- Cache Rate: $CACHE_RATE
- Workers: $NUM_WORKERS
- Class-Aware Sampling: $CLASS_AWARE_SAMPLING

Validation:
- Interval: $VAL_INTERVAL epoch
- Sliding Window Batch: $SLIDING_WINDOW_BATCH_SIZE
- Overlap: $OVERLAP
- Always Use Sliding Window: $ALWAYS_USE_SLIDING_WINDOW
- Early Stopping Patience: $EARLY_STOPPING_PATIENCE
EOF

echo "=================================================="
echo "SUPERVISED FINE-TUNING (Registered dHCP - FIXED)"
echo "=================================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Results: $RESULTS_DIR"
echo ""
echo "✅ FIXES APPLIED:"
echo "   1. Foreground mask: label >= 0 (includes class 0)"
echo "   2. Validation: No center crop → sliding window enabled"
echo "   3. LR swap file: parameterized path"
echo "   4. Rotation angle: parameterized ($MAX_ROTATION_ANGLE rad)"
echo "   5. Always use sliding window: $ALWAYS_USE_SLIDING_WINDOW"
echo ""
echo "🔧 Key Enhancements:"
echo "   - Foreground-aware percentile normalization (fixed)"
echo "   - LR flip with anatomical label swapping"
echo "   - Class-aware patch-level sampling"
echo "   - Balanced Tversky loss (α=β=0.5)"
echo "   - Higher class weights (max=$MAX_WEIGHT)"
echo "   - Dynamic encoder LR scheduling"
echo "   - Increased sliding window overlap ($OVERLAP)"
echo ""
echo "📊 Configuration:"
echo "   - Loss: $LOSS_TYPE"
echo "   - Classes: $NUM_CLASSES (regions 1-87, background=-1)"
echo "   - ROI: ${ROI_SIZE}×${ROI_SIZE}×${ROI_SIZE}"
echo "   - Spacing: $TARGET_SPACING mm"
echo "   - Rotation: $MAX_ROTATION_ANGLE rad (~$(echo "$MAX_ROTATION_ANGLE * 180 / 3.14159" | bc -l | cut -c1-5)°)"
echo "   - Batch: $BATCH_SIZE × $ACCUMULATION_STEPS = $((BATCH_SIZE * ACCUMULATION_STEPS)) effective"
echo "   - Epochs: $EPOCHS (early stop at $EARLY_STOPPING_PATIENCE)"
echo ""
echo "🗿 Model:"
echo "   - SSL pretrained encoder"
echo "   - Encoder freeze: $FREEZE_ENCODER_EPOCHS epochs"
echo "   - Encoder LR: ${ENCODER_LR_RATIO}× base"
echo "   - Dynamic boost at epoch $ENCODER_LR_BOOST_EPOCH"
echo ""
echo "📈 Training:"
echo "   - LR: $LEARNING_RATE (min: $LR_MIN)"
echo "   - Scheduler: $LR_SCHEDULER"
echo "   - Class-aware sampling: $CLASS_AWARE_SAMPLING"
echo "   - Mixed precision: $USE_AMP"
echo "=================================================="

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Enable mixed precision training
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check GPU
echo ""
echo "🖥️ GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv
echo ""

# Build command
CMD="python main_supervised_dhcp.py \
    --exp_name $EXPERIMENT_NAME \
    --results_dir $RESULTS_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --accumulation_steps $ACCUMULATION_STEPS \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --encoder_lr_ratio $ENCODER_LR_RATIO \
    --freeze_encoder_epochs $FREEZE_ENCODER_EPOCHS \
    --encoder_lr_boost_epoch $ENCODER_LR_BOOST_EPOCH \
    --encoder_lr_boost_ratio $ENCODER_LR_BOOST_RATIO \
    --in_channels 1 \
    --out_channels $NUM_CLASSES \
    --feature_size $FEATURE_SIZE \
    --roi_x $ROI_SIZE \
    --roi_y $ROI_SIZE \
    --roi_z $ROI_SIZE \
    --dropout_rate $DROPOUT_RATE \
    --data_split_json $DATA_SPLIT_JSON \
    --class_prior_json $CLASS_PRIOR_JSON \
    --laterality_pairs_json $LATERALITY_PAIRS_JSON \
    --pretrained_model $PRETRAINED_MODEL \
    --num_workers $NUM_WORKERS \
    --cache_rate $CACHE_RATE \
    --cache_dir $CACHE_DIR \
    --foreground_only \
    --max_rotation_angle $MAX_ROTATION_ANGLE \
    --loss_type $LOSS_TYPE \
    --dice_smooth 1e-5 \
    --dice_weight $DICE_WEIGHT \
    --ce_weight $CE_WEIGHT \
    --focal_gamma $FOCAL_GAMMA \
    --tversky_alpha $TVERSKY_ALPHA \
    --tversky_beta $TVERSKY_BETA \
    --class_weights auto \
    --weight_power $WEIGHT_POWER \
    --max_weight $MAX_WEIGHT \
    --min_weight $MIN_WEIGHT \
    --target_spacing $TARGET_SPACING \
    --lr_scheduler $LR_SCHEDULER \
    --lr_patience $LR_PATIENCE \
    --lr_min $LR_MIN \
    --lr_factor $LR_FACTOR \
    --clip $CLIP \
    --val_interval $VAL_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --sliding_window_batch_size $SLIDING_WINDOW_BATCH_SIZE \
    --overlap $OVERLAP"

# Add optional flags
if [ "$USE_AMP" = true ]; then
    CMD="$CMD --use_amp"
fi

if [ "$AUTO_BATCH_SIZE" = true ]; then
    CMD="$CMD --auto_batch_size"
fi

if [ "$VISUALIZE_WEIGHTS" = true ]; then
    CMD="$CMD --visualize_weights"
fi

if [ "$CLASS_AWARE_SAMPLING" = true ]; then
    CMD="$CMD --class_aware_sampling"
fi

if [ "$DYNAMIC_ENCODER_LR" = true ]; then
    CMD="$CMD --dynamic_encoder_lr"
fi

if [ "$USE_PERSISTENT_DATASET" = true ]; then
    CMD="$CMD --use_persistent_dataset"
fi

if [ "$CLEAN_CACHE" = true ]; then
    CMD="$CMD --clean_cache"
fi

if [ "$USE_POST_PROCESSING" = true ]; then
    CMD="$CMD --use_post_processing"
fi

if [ "$USE_EMA" = true ]; then
    CMD="$CMD --use_ema --ema_decay $EMA_DECAY"
fi

if [ "$USE_TTA" = true ]; then
    CMD="$CMD --use_tta"
fi

if [ "$ALWAYS_USE_SLIDING_WINDOW" = true ]; then
    CMD="$CMD --always_use_sliding_window"
fi

if [ "$DEBUG_SAMPLES" -gt 0 ]; then
    CMD="$CMD --debug_samples $DEBUG_SAMPLES"
fi

# Run training
echo "🚀 Starting supervised fine-tuning with fixes..."
echo "Command: $CMD"
echo ""
eval "$CMD 2>&1 | tee $RESULTS_DIR/training.log"

# Check if training completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Supervised fine-tuning completed successfully!"
    echo "=================================================="
    echo ""
    echo "📊 Results saved to: $RESULTS_DIR"
    echo ""
    echo "📈 To monitor training:"
    echo "   tensorboard --logdir=$RESULTS_DIR/tensorboard"
    echo ""
    echo "📝 Analysis checklist:"
    echo "   1. Check if class 0 (region 1) Dice improved"
    echo "   2. Verify sliding window was used in validation"
    echo "   3. Review class weights distribution plot"
    echo "   4. Check per-region Dice scores"
    echo "   5. Analyze training dynamics (overfitting plot)"
    echo "   6. Compare train vs val macro Dice"
    echo ""
    echo "🔍 Key files:"
    echo "   - Best model: $RESULTS_DIR/best_model.pth"
    echo "   - Training log: $RESULTS_DIR/training.log"
    echo "   - Config: $RESULTS_DIR/config.json"
    echo "   - Monitoring: $RESULTS_DIR/monitoring/"
    echo "   - Class weights: $RESULTS_DIR/class_weights.png"
else
    echo ""
    echo "❌ Training failed! Check the log for errors."
    echo "   Log file: $RESULTS_DIR/training.log"
    echo ""
    echo "Common issues to check:"
    echo "   1. LR swap file path"
    echo "   2. Data split JSON path"
    echo "   3. GPU memory (try reducing batch size)"
    echo "   4. Class prior JSON format"
fi

# Generate summary report
echo ""
echo "📊 Generating final summary..."
python -c "
import json
import os
import pandas as pd
import numpy as np

results_dir = '$RESULTS_DIR'
config_path = os.path.join(results_dir, 'config.json')
monitor_path = os.path.join(results_dir, 'monitoring', 'training_history.csv')

print('\\n=== Configuration Summary ===')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f\"Loss type: {config.get('loss_type', 'unknown')}\")
    print(f\"Target spacing: {config.get('target_spacing', 'unknown')}\")
    print(f\"Max rotation: {config.get('max_rotation_angle', 0.1)} rad\")
    print(f\"Always sliding window: {config.get('always_use_sliding_window', False)}\")
    print(f\"Effective batch size: {config.get('batch_size', 4) * config.get('accumulation_steps', 4)}\")
    print(f\"LR scheduler: {config.get('lr_scheduler', 'unknown')}\")
    print(f\"Early stopping patience: {config.get('early_stopping_patience', 40)}\")

    if config.get('laterality_pairs_json'):
        if os.path.exists(config['laterality_pairs_json']):
            print(f\"✓ LR swap file found: {config['laterality_pairs_json']}\")
        else:
            print(f\"✗ LR swap file NOT found: {config['laterality_pairs_json']}\")

if os.path.exists(monitor_path):
    df = pd.read_csv(monitor_path)
    if not df.empty and 'val_dice' in df.columns:
        df = df.dropna(subset=['val_dice'])

        if len(df) > 0:
            best_idx = df['val_dice'].idxmax()
            best_epoch = df.loc[best_idx, 'epoch']
            best_dice = df.loc[best_idx, 'val_dice']

            print(f\"\\n=== Best Performance ===\")
            print(f\"Best epoch: {best_epoch}\")
            print(f\"Best val dice: {best_dice:.4f}\")

            if 'val_macro_dice' in df.columns:
                best_macro = df.loc[best_idx, 'val_macro_dice']
                print(f\"Best val macro dice: {best_macro:.4f}\")

            # Check if early stopped
            final_epoch = df['epoch'].max()
            if final_epoch < config.get('epochs', 150):
                print(f\"\\nEarly stopped at epoch: {final_epoch}\")
                print(f\"Stopped {final_epoch - best_epoch} epochs after best\")

            # Analyze training dynamics
            if 'train_dice' in df.columns:
                train_dice_at_best = df.loc[best_idx, 'train_dice']
                if not pd.isna(train_dice_at_best):
                    overfitting = train_dice_at_best - best_dice
                    print(f\"\\nTrain-Val gap at best: {overfitting:.4f}\")
                    if overfitting > 0.1:
                        print(\"⚠️ Significant overfitting detected\")
                    elif overfitting < 0.02:
                        print(\"✓ Good generalization (low train-val gap)\")
"