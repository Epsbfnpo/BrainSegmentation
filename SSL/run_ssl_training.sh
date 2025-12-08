#!/bin/bash
# Self-supervised pretraining on dHCP dataset - DISTRIBUTED VERSION
# Multi-GPU training with PyTorch DDP

# Configuration
EXPERIMENT_NAME="dHCP_distributed"
EPOCHS=150
BATCH_SIZE=8  # Per GPU batch size
LEARNING_RATE=1e-4
WEIGHT_DECAY=5e-4

# Number of GPUs to use
NUM_GPUS=2

# Model parameters
ROI_SIZE_X=96
ROI_SIZE_Y=96
ROI_SIZE_Z=96
FEATURE_SIZE=48

# SSL parameters for registered data
MASK_RATIO=0.25
MASK_PATCH_SIZE=16
MAX_ROTATION_ANGLE=0.1  # Maximum rotation in radians (~5.7 degrees)
PROJECTION_DIM=128

# Loss weights optimized for registered data
INPAINTING_WEIGHT=1.5
ROTATION_WEIGHT=0.1
CONTRASTIVE_WEIGHT=0.8

# Data parameters
DATA_SPLIT_JSON="/scratch3/liu275/Data/dHCP_registered/dHCP_split.json"
PRETRAINED_MODEL="/scratch3/liu275/Code/BrainSegFounder/model/64-gpu-model_bestValRMSE.pt"

# Target spacing (mm)
TARGET_SPACING="0.5 0.5 0.5"

# OPTIMIZED System settings
NUM_WORKERS=16 # Per GPU - will be adjusted based on available CPUs
CACHE_RATE=1.0  # Cache rate for distributed training
PREFETCH_FACTOR=4  # Number of batches to prefetch per worker

# Performance optimization flags
USE_PERSISTENT_CACHE=true  # Set to true to cache preprocessed data to disk
CACHE_DIR="/scratch3/liu275/ssl_cache"  # Directory for persistent cache
DETERMINISTIC=false  # Set to false for maximum speed (true for reproducibility)
CUDNN_BENCHMARK=true  # Enable cuDNN auto-tuning
USE_GRADIENT_CHECKPOINTING=true  # Disable for speed (enable if OOM)
COMPILE_MODEL=true  # Set to true if using PyTorch 2.0+
COMPILE_MODE="max-autotune"  # Options: default, reduce-overhead, max-autotune

# Learning rate scheduler
LR_SCHEDULER="cosine"
LR_COSINE_T0=50
LR_MIN=1e-6

# Validation settings
VAL_INTERVAL=2
SAVE_INTERVAL=10

# Distributed training timeout (minutes)
DIST_TIMEOUT=120

# Results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/datasets/work/hb-nhmrc-dhcp/work/liu275/SSL/${EXPERIMENT_NAME}"

# Create results directory
mkdir -p $RESULTS_DIR

# Save configuration
cat > $RESULTS_DIR/config.txt << EOF
SSL Pretraining Configuration - DISTRIBUTED VERSION
============================================================
Experiment: $EXPERIMENT_NAME
Timestamp: $TIMESTAMP

Dataset: dHCP_registered (T2w single modality, atlas-aligned)
Stage: 2 - Self-supervised pretraining with SimSiam (DISTRIBUTED)

Training Parameters:
- Number of GPUs: $NUM_GPUS
- Epochs: $EPOCHS
- Batch Size per GPU: $BATCH_SIZE
- Total Effective Batch Size: $((BATCH_SIZE * NUM_GPUS))
- Learning Rate: $LEARNING_RATE
- Weight Decay: $WEIGHT_DECAY
- LR Scheduler: $LR_SCHEDULER
- ROI Size: ${ROI_SIZE_X}x${ROI_SIZE_Y}x${ROI_SIZE_Z}
- Target Spacing: $TARGET_SPACING mm

SSL Components (Optimized for Registered Data):
1. Masked Volume Inpainting
   - Mask Ratio: $MASK_RATIO
   - Patch Size: $MASK_PATCH_SIZE
   - Weight: $INPAINTING_WEIGHT

2. Small-Angle Rotation Regression
   - Max Angle: $MAX_ROTATION_ANGLE radians (~5.7 degrees)
   - Weight: $ROTATION_WEIGHT
   - Using SmoothL1 regression loss

3. SimSiam Self-Distillation (No Negatives!)
   - Projection Dim: $PROJECTION_DIM
   - Weight: $CONTRASTIVE_WEIGHT

Data:
- Split JSON: $DATA_SPLIT_JSON
- Pretrained Model: $PRETRAINED_MODEL
- Cache Rate: $CACHE_RATE
- Workers per GPU: $NUM_WORKERS
- Prefetch Factor: $PREFETCH_FACTOR
- Persistent Cache: $USE_PERSISTENT_CACHE

Performance Optimizations:
- Distributed Training: $NUM_GPUS GPUs
- Deterministic: $DETERMINISTIC
- cuDNN Benchmark: $CUDNN_BENCHMARK
- Gradient Checkpointing: $USE_GRADIENT_CHECKPOINTING
- Model Compilation: $COMPILE_MODEL
- Compile Mode: $COMPILE_MODE
- AMP: Enabled (BF16)
- Non-blocking GPU transfers: Enabled
- Optimized data pipeline: Enabled
EOF

echo "=================================================="
echo "DISTRIBUTED SSL PRETRAINING WITH SimSiam"
echo "=================================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Results: $RESULTS_DIR"
echo ""
echo "🚀 Distributed Training Configuration:"
echo "   - Number of GPUs: $NUM_GPUS"
echo "   - Batch size per GPU: $BATCH_SIZE"
echo "   - Total effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "   - Distributed backend: NCCL"
echo ""
echo "📊 Model Configuration:"
echo "   - SimSiam self-distillation (no negatives)"
echo "   - Small-angle rotation regression"
echo "   - ROI: ${ROI_SIZE_X}x${ROI_SIZE_Y}x${ROI_SIZE_Z}"
echo "   - Epochs: $EPOCHS"
echo ""
echo "⚡ Performance Optimizations:"
echo "   - Workers per GPU: $NUM_WORKERS"
echo "   - Prefetch: $PREFETCH_FACTOR batches per worker"
echo "   - Cache rate: $CACHE_RATE"
echo "   - cuDNN auto-tuning: $CUDNN_BENCHMARK"
echo "   - Deterministic: $DETERMINISTIC"
echo "   - AMP: BF16 (H100 optimized)"
echo "   - Non-blocking transfers: Enabled"
echo "   - Gradient checkpointing: $USE_GRADIENT_CHECKPOINTING"
echo ""
echo "🔧 SSL Components:"
echo "   - Masked Inpainting (weight=$INPAINTING_WEIGHT)"
echo "   - Rotation Regression (weight=$ROTATION_WEIGHT)"
echo "   - SimSiam Learning (weight=$CONTRASTIVE_WEIGHT)"
echo "=================================================="

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=$NUM_GPUS

# Enable mixed precision training
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"  # Added 9.0 for H100

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Suppress checkpoint warnings
export TORCH_USE_REENTRANT=False

# Enable NCCL optimizations
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_TREE_THRESHOLD=0
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_TIMEOUT=$((DIST_TIMEOUT * 60))  # Convert to seconds
export NCCL_BLOCKING_WAIT=1

# CPU optimizations
export OMP_NUM_THREADS=$NUM_WORKERS
export MKL_NUM_THREADS=$NUM_WORKERS

# TF32 for Ampere/Hopper GPUs
export TORCH_ALLOW_TF32=1
export CUBLAS_ALLOW_TF32=1

# Check GPU availability
echo ""
echo "🖥️ GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv
echo ""

# Check CPU
echo "🖥️ CPU Status:"
echo "  Cores: $(nproc)"
echo "  Available memory: $(free -h | grep Mem | awk '{print $7}')"
echo ""

# Build command with optimizations
CMD="torchrun --standalone --nproc_per_node=$NUM_GPUS main_ssl_dhcp.py \
    --exp_name $EXPERIMENT_NAME \
    --results_dir ./results_ssl/ \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --in_channels 1 \
    --out_channels 88 \
    --feature_size $FEATURE_SIZE \
    --roi_x $ROI_SIZE_X \
    --roi_y $ROI_SIZE_Y \
    --roi_z $ROI_SIZE_Z \
    --data_split_json $DATA_SPLIT_JSON \
    --pretrained_model $PRETRAINED_MODEL \
    --num_workers $NUM_WORKERS \
    --cache_rate $CACHE_RATE \
    --prefetch_factor $PREFETCH_FACTOR \
    --mask_ratio $MASK_RATIO \
    --mask_patch_size $MASK_PATCH_SIZE \
    --max_rotation_angle $MAX_ROTATION_ANGLE \
    --projection_dim $PROJECTION_DIM \
    --inpainting_weight $INPAINTING_WEIGHT \
    --rotation_weight $ROTATION_WEIGHT \
    --contrastive_weight $CONTRASTIVE_WEIGHT \
    --target_spacing $TARGET_SPACING \
    --lr_scheduler $LR_SCHEDULER \
    --lr_cosine_t0 $LR_COSINE_T0 \
    --lr_min $LR_MIN \
    --val_interval $VAL_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --clip 1.0 \
    --use_small_rotation \
    --use_amp \
    --amp_dtype bfloat16 \
    --dist_timeout $DIST_TIMEOUT"

# Add performance optimization flags based on settings
if [ "$DETERMINISTIC" = true ]; then
    CMD="$CMD --deterministic"
fi

if [ "$CUDNN_BENCHMARK" = true ]; then
    CMD="$CMD --cudnn_benchmark"
fi

if [ "$USE_GRADIENT_CHECKPOINTING" = true ]; then
    CMD="$CMD --use_gradient_checkpointing"
fi

if [ "$COMPILE_MODEL" = true ]; then
    CMD="$CMD --compile_model --compile_mode $COMPILE_MODE"
fi

if [ "$USE_PERSISTENT_CACHE" = true ]; then
    CMD="$CMD --use_persistent_cache --cache_dir $CACHE_DIR"
fi

# Run training with timing
echo "🚀 Starting distributed SSL pretraining on $NUM_GPUS GPUs..."
echo ""
echo "Command: $CMD"
echo ""

START_TIME=$(date +%s)

# Run the distributed training
$CMD 2>&1 | tee $RESULTS_DIR/training.log

# Calculate training time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Check if training completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Distributed SSL pretraining completed successfully!"
    echo "=================================================="
    echo ""
    echo "⏱️  Training time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo "📊 Results saved to: $RESULTS_DIR"
    echo ""
    echo "📈 To monitor training:"
    echo "   tensorboard --logdir=$RESULTS_DIR/tensorboard"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Check monitoring plots in: $RESULTS_DIR/monitoring/"
    echo "   2. Use best model for Stage 3 supervised fine-tuning"
    echo "   3. Best model checkpoint: $RESULTS_DIR/best_model.pth"
    echo ""
    echo "⚡ Performance Summary:"
    echo "   - Training on $NUM_GPUS GPUs"
    echo "   - Total training time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "   - Average epoch time: $((DURATION / EPOCHS / 60)) minutes"
    echo "   - Check training.log for detailed metrics"
else
    echo ""
    echo "❌ Training failed! Check the log for errors."
    echo "   Log file: $RESULTS_DIR/training.log"
    echo ""
    echo "Common issues:"
    echo "   - OOM: Reduce batch_size or enable gradient_checkpointing"
    echo "   - NCCL errors: Check GPU availability and network settings"
    echo "   - Slow: Increase cache_rate, enable compile_model"
    echo "   - NaN: Check data normalization, reduce learning rate"
fi

# Alternative: Run with SLURM (if available)
cat > $RESULTS_DIR/slurm_submit.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=ssl_distributed
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Load modules (adjust based on your cluster)
module load cuda/11.8
module load python/3.9

# Activate conda environment
conda activate ssl_env

# Run the distributed training
srun torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:29500 main_ssl_dhcp_distributed.py [arguments...]
SLURM_EOF

echo ""
echo "📋 Alternative SLURM submission script saved to:"
echo "   $RESULTS_DIR/slurm_submit.sh"
echo ""
echo "To submit to SLURM cluster:"
echo "   sbatch $RESULTS_DIR/slurm_submit.sh"