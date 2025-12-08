#!/usr/bin/env python3
"""
Main script for SSL pretraining on dHCP dataset with Distributed Training Support
Multi-GPU version with DDP (DistributedDataParallel)
"""
import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import GradScaler
from tensorboardX import SummaryWriter
from monai.networks.nets import SwinUNETR
from monai.utils import set_determinism
import numpy as np
from datetime import datetime, timedelta
import traceback
import gc
import warnings

# Import SSL components
from ssl_components import SSL3WayLoss, MaskedVolumeInpainting, RotationPrediction, SimSiamLearning
from data_loader_ssl import get_ssl_dataloaders_distributed
from ssl_trainer import train_epoch_ssl_distributed, val_epoch_ssl_distributed, save_checkpoint_ssl
from monitoring import SSLMonitorDistributed


def is_dist():
    """Check if distributed training is enabled"""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_parser():
    """Get argument parser"""
    parser = argparse.ArgumentParser(description='SSL Pretraining on dHCP Dataset with SimSiam - Distributed Version')

    # Basic training parameters
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Mini-batch size per GPU')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay')

    # Model parameters
    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of input channels (T2w only)')
    parser.add_argument('--out_channels', default=88, type=int,
                        help='Number of output channels for reconstruction')
    parser.add_argument('--feature_size', default=48, type=int,
                        help='Feature size for transformer')
    parser.add_argument('--roi_x', default=96, type=int,
                        help='ROI size in x direction')
    parser.add_argument('--roi_y', default=96, type=int,
                        help='ROI size in y direction')
    parser.add_argument('--roi_z', default=96, type=int,
                        help='ROI size in z direction')

    # Data parameters
    parser.add_argument('--data_split_json',
                        default='/scratch3/liu275/Data/dHCP_registered/dHCP_split.json',
                        type=str, help='Path to dHCP data split JSON file')
    parser.add_argument('--class_prior_json', default=None, type=str,
                        help='Path to class prior JSON file')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of data loading workers per GPU')
    parser.add_argument('--cache_rate', default=0.1, type=float,
                        help='Cache rate for data loading')
    parser.add_argument('--use_persistent_cache', action='store_true',
                        help='Use persistent disk-based cache for preprocessed data')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory for persistent cache')
    parser.add_argument('--prefetch_factor', default=4, type=int,
                        help='Number of batches to prefetch per worker')

    # Pretrained model
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='Path to UKB pretrained model')
    parser.add_argument('--resume_training', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # SSL specific parameters
    parser.add_argument('--mask_ratio', default=0.20, type=float,
                        help='Ratio of volume to mask for inpainting')
    parser.add_argument('--mask_patch_size', default=16, type=int,
                        help='Size of masked patches')
    parser.add_argument('--max_rotation_angle', default=0.1, type=float,
                        help='Maximum rotation angle in radians for small rotations')
    parser.add_argument('--temperature', default=0.1, type=float,
                        help='Temperature (not used with SimSiam but kept for compatibility)')
    parser.add_argument('--projection_dim', default=128, type=int,
                        help='Dimension of projection head for SimSiam')
    parser.add_argument('--use_small_rotation', action='store_true', default=True,
                        help='Use small angle rotations for registered data')

    # Loss weights - optimized for registered data
    parser.add_argument('--inpainting_weight', default=1.5, type=float,
                        help='Weight for inpainting loss')
    parser.add_argument('--rotation_weight', default=0.1, type=float,
                        help='Weight for rotation prediction loss')
    parser.add_argument('--contrastive_weight', default=0.8, type=float,
                        help='Weight for SimSiam loss')

    # Training settings
    parser.add_argument('--clip', default=1.0, type=float,
                        help='Gradient clipping')
    parser.add_argument('--val_interval', default=5, type=int,
                        help='Validation interval')
    parser.add_argument('--save_interval', default=10, type=int,
                        help='Save checkpoint interval')

    # Resolution parameters
    parser.add_argument('--target_spacing', nargs=3, type=float,
                        default=[0.8, 0.8, 0.8],
                        help='Target voxel spacing in mm')

    # LR scheduler
    parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine', 'plateau'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_cosine_t0', default=25, type=int,
                        help='Number of iterations for the first restart')
    parser.add_argument('--lr_patience', default=10, type=int,
                        help='Patience for ReduceLROnPlateau')
    parser.add_argument('--lr_min', default=1e-6, type=float,
                        help='Minimum learning rate')
    parser.add_argument('--lr_factor', default=0.5, type=float,
                        help='Factor for ReduceLROnPlateau')

    # AMP parameters
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use Automatic Mixed Precision (AMP) training')
    parser.add_argument('--amp_dtype', default='bfloat16', choices=['float16', 'bfloat16'],
                        help='AMP data type (bfloat16 recommended for H100)')

    # Performance optimization parameters
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Use deterministic mode (slower but reproducible)')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=True,
                        help='Enable cudnn auto-tuner for better performance')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', default=False,
                        help='Use gradient checkpointing (saves memory but slower)')
    parser.add_argument('--compile_model', action='store_true', default=False,
                        help='Use torch.compile for potential speedups (requires PyTorch 2.0+)')
    parser.add_argument('--compile_mode', default='max-autotune',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')

    # Distributed training parameters
    parser.add_argument('--dist_timeout', default=120, type=int,
                        help='Timeout in minutes for distributed operations')

    # Output
    parser.add_argument('--results_dir', default='./results_ssl/', type=str,
                        help='Directory for results')
    parser.add_argument('--exp_name', default=None, type=str,
                        help='Experiment name')

    return parser


def load_pretrained_weights(model, checkpoint_path):
    """Load pretrained weights from UKB model"""
    print(f"\n📥 Loading pretrained weights from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Pretrained model not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Handle different prefixes and filter relevant weights
    model_dict = model.state_dict()
    pretrained_dict = {}

    for k, v in state_dict.items():
        # Remove module. prefix if present
        if k.startswith('module.'):
            k = k[7:]

        # Skip output layer weights (different number of classes)
        if 'out' in k or 'head' in k or 'final' in k:
            print(f"  Skipping layer: {k}")
            continue

        # Check if key exists and shape matches
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
            else:
                print(f"  Shape mismatch for {k}: {v.shape} vs {model_dict[k].shape}")

    # Update model dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print(f"✓ Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model")
    return model


def create_ssl_model(args):
    """Create model for SSL pretraining with optional optimizations"""
    # Base SwinUNETR model
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.in_channels,  # Reconstruct input for SSL
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_gradient_checkpointing,  # Controlled by args
    )

    # Load pretrained weights
    if args.pretrained_model:
        model = load_pretrained_weights(model, args.pretrained_model)

    return model


def setup_performance_optimizations(args):
    """Setup performance optimizations based on args"""

    print("\n⚡ Performance Optimizations:")

    # Determinism vs Performance
    if args.deterministic:
        set_determinism(seed=42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("  ✓ Deterministic mode enabled (reproducible but slower)")
    else:
        # For maximum performance
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)

        if args.cudnn_benchmark:
            print("  ✓ cuDNN auto-tuner enabled (faster but non-deterministic)")
        else:
            print("  ⚠️  cuDNN auto-tuner disabled")

        # Set seed for some reproducibility even in non-deterministic mode
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

    # Memory/Speed tradeoff
    if args.use_gradient_checkpointing:
        print("  ✓ Gradient checkpointing enabled (saves memory, slower)")
    else:
        print("  ✓ Gradient checkpointing disabled (faster, uses more memory)")

    # Additional optimizations
    if torch.cuda.is_available():
        # Enable TF32 for Ampere GPUs (A100, H100)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  ✓ TF32 enabled for matrix operations")

        # Set memory fraction to allow better caching
        torch.cuda.set_per_process_memory_fraction(0.95)
        print("  ✓ GPU memory fraction set to 95%")


def main():
    """Main training function with distributed support"""
    try:
        # Parse arguments
        parser = get_parser()
        args = parser.parse_args()

        # Initialize distributed training
        if is_dist():
            timeout = timedelta(minutes=args.dist_timeout)
            dist.init_process_group(backend="nccl", timeout=timeout)
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            world_size = dist.get_world_size()
            # Set environment variables for better NCCL performance
            os.environ['NCCL_TIMEOUT'] = str(args.dist_timeout * 60)
            os.environ['NCCL_BLOCKING_WAIT'] = '1'
        else:
            local_rank = 0
            world_size = 1

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        is_main = (not is_dist()) or dist.get_rank() == 0

        # Create experiment name if not provided
        if args.exp_name is None:
            args.exp_name = f"ssl_dHCP_distributed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create directories (only on main process)
        if is_main:
            args.results_dir = os.path.join(args.results_dir, args.exp_name)
            os.makedirs(args.results_dir, exist_ok=True)

        # Setup performance optimizations
        setup_performance_optimizations(args)

        # Save configuration (only on main process)
        if is_main:
            config_path = os.path.join(args.results_dir, 'config.json')
            config_dict = vars(args)
            config_dict['world_size'] = world_size
            config_dict['distributed'] = is_dist()
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)

            print("\n" + "=" * 80)
            print("SELF-SUPERVISED PRETRAINING ON dHCP DATASET - DISTRIBUTED VERSION")
            print("=" * 80)
            print(f"Experiment: {args.exp_name}")
            print(f"Results directory: {args.results_dir}")
            print(f"World Size: {world_size} GPUs")
            print(f"Training epochs: {args.epochs}")
            print(f"Batch size per GPU: {args.batch_size}")
            print(f"Total effective batch size: {args.batch_size * world_size}")
            print(f"Initial learning rate: {args.lr}")
            print(f"Weight decay: {args.weight_decay}")
            print(f"LR scheduler: {args.lr_scheduler}")
            print(f"ROI size: {args.roi_x}×{args.roi_y}×{args.roi_z}")
            print(f"Target spacing: {args.target_spacing}")
            print(f"AMP: {'Enabled (' + args.amp_dtype + ')' if args.use_amp else 'Disabled'}")
            print(f"Deterministic: {args.deterministic}")
            print(f"cuDNN Benchmark: {args.cudnn_benchmark}")
            print(f"Gradient Checkpointing: {args.use_gradient_checkpointing}")
            print(f"Model Compilation: {args.compile_model}")
            print("\nSSL Components (Optimized for Registered Data):")
            print(f"  - Masked Volume Inpainting (weight={args.inpainting_weight})")
            print(f"  - Small-Angle Rotation Regression (weight={args.rotation_weight})")
            print(f"    Max angle: {args.max_rotation_angle:.3f} rad (~{np.degrees(args.max_rotation_angle):.1f}°)")
            print(f"  - SimSiam Self-Distillation (weight={args.contrastive_weight})")
            print(f"    NO negative samples required!")
            print(f"    Projection dim: {args.projection_dim}")
            print("=" * 80)

        # Check GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This code requires GPU.")

        if is_main:
            print(f"\n🖥️  GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
            gpu_memory = torch.cuda.get_device_properties(local_rank).total_memory / 1024 ** 3
            print(f"Memory: {gpu_memory:.2f} GB")

        # Check for BF16 support
        if args.use_amp and args.amp_dtype == 'bfloat16':
            if torch.cuda.is_bf16_supported():
                if is_main:
                    print(f"✓ BF16 support detected on {torch.cuda.get_device_name(local_rank)}")
            else:
                if is_main:
                    print(f"⚠️  BF16 not supported on {torch.cuda.get_device_name(local_rank)}, falling back to FP16")
                args.amp_dtype = 'float16'

        # Create tensorboard writer (only on main process)
        writer = SummaryWriter(log_dir=os.path.join(args.results_dir, 'tensorboard')) if is_main else None

        # Create monitor (distributed version)
        monitor = SSLMonitorDistributed(args.results_dir) if is_main else None

        # Get data loaders with distributed sampling
        if is_main:
            print("\n📊 Creating distributed data loaders...")
        train_loader, val_loader = get_ssl_dataloaders_distributed(
            args,
            is_distributed=is_dist(),
            world_size=world_size,
            rank=local_rank
        )
        if is_main:
            print(f"✓ Training samples: {len(train_loader.dataset)}")
            print(f"✓ Validation samples: {len(val_loader.dataset)}")

        # Create model
        if is_main:
            print("\n🗿 Creating model...")
        model = create_ssl_model(args).to(device)

        # Create SSL components with SimSiam and small-angle rotation
        if is_main:
            print("\n🔧 Creating SSL components with SimSiam and small-angle rotation...")
        ssl_loss = SSL3WayLoss(
            mask_ratio=args.mask_ratio,
            mask_patch_size=args.mask_patch_size,
            rotation_angles=None,  # Not used with small-angle rotation
            temperature=args.temperature,  # Not used but kept for compatibility
            projection_dim=args.projection_dim,
            feature_dim=768,  # Adjust based on model architecture
            inpainting_weight=args.inpainting_weight,
            rotation_weight=args.rotation_weight,
            contrastive_weight=args.contrastive_weight,
            use_small_rotation=args.use_small_rotation,
            max_rotation_angle=args.max_rotation_angle,
        ).to(device)

        if is_main:
            print("✓ Initialized SSL3WayLoss with SimSiam (no negatives)")
            print("✓ Using small-angle rotation regression")
            print("✓ Using intensity augmentations for registered data")

        # Wrap model and ssl_loss with DDP
        if is_dist():
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                gradient_as_bucket_view=True,
                static_graph=False
            )

            ssl_loss = DDP(
                ssl_loss,
                device_ids=[local_rank],
                output_device=local_rank,
                gradient_as_bucket_view=True,
                static_graph=False
            )

            if is_main:
                print("\n✓ Model and SSL components wrapped with DistributedDataParallel")

        # Compile model if requested (PyTorch 2.0+)
        if args.compile_model:
            if hasattr(torch, "compile"):
                if is_main:
                    print(f"🔧 Compiling model with mode='{args.compile_mode}'...")
                try:
                    model = torch.compile(model, mode=args.compile_mode)
                    if is_main:
                        print("✓ Model compilation successful")
                except Exception as e:
                    if is_main:
                        print(f"⚠️  Model compilation failed: {e}")
                        print("  Continuing without compilation...")
            else:
                if is_main:
                    print("⚠️  torch.compile not available (requires PyTorch 2.0+)")

        # Create optimizer
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(ssl_loss.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            fused=True if torch.cuda.is_available() else False  # Use fused optimizer on CUDA
        )

        if torch.cuda.is_available() and optimizer.param_groups[0].get('fused', False):
            if is_main:
                print("✓ Using fused AdamW optimizer for better performance")

        # Create GradScaler for AMP
        scaler = None
        if args.use_amp:
            scaler = GradScaler()
            if is_main:
                print(f"✓ Initialized GradScaler for AMP ({args.amp_dtype})")

        # Learning rate scheduler
        if args.lr_scheduler == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=args.lr_cosine_t0,
                T_mult=2,
                eta_min=args.lr_min
            )
            scheduler_needs_metric = False
            if is_main:
                print(f"✓ Using CosineAnnealingWarmRestarts scheduler (T_0={args.lr_cosine_t0})")
        else:  # plateau
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=args.lr_factor,
                patience=args.lr_patience,
                min_lr=args.lr_min
            )
            scheduler_needs_metric = True
            if is_main:
                print(f"✓ Using ReduceLROnPlateau scheduler (patience={args.lr_patience}, factor={args.lr_factor})")

        # Resume training if needed
        start_epoch = 1
        best_val_loss = float('inf')

        if args.resume_training and args.checkpoint:
            if is_main:
                print(f"\n📂 Resuming from checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)

            # Load model and ssl_loss states
            if is_dist():
                model.module.load_state_dict(checkpoint['model_state_dict'])
                ssl_loss.module.load_state_dict(checkpoint['ssl_loss_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                ssl_loss.load_state_dict(checkpoint['ssl_loss_state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                if is_main:
                    print("✓ Loaded AMP scaler state")
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            if monitor and 'monitor_history' in checkpoint:
                monitor.history = checkpoint['monitor_history']
            if is_main:
                print(f"✓ Resuming from epoch {start_epoch} with best val loss = {best_val_loss:.4f}")

        # Synchronize before training
        if is_dist():
            if is_main:
                print("\n🔄 Synchronizing all processes before training...")
            dist.barrier()
            if is_main:
                print("✓ All processes synchronized, starting training!")

        # Print optimization summary
        if is_main:
            print("\n📊 Optimization Summary:")
            print(f"  • World Size: {world_size} GPUs")
            print(f"  • Data Loading: {args.num_workers} workers per GPU, prefetch={args.prefetch_factor}")
            print(f"  • GPU Transfer: Non-blocking enabled")
            print(f"  • Mixed Precision: {args.amp_dtype if args.use_amp else 'Disabled'}")
            print(f"  • cuDNN Tuning: {'Enabled' if args.cudnn_benchmark else 'Disabled'}")
            print(f"  • Model Compilation: {'Enabled' if args.compile_model else 'Disabled'}")
            print(f"  • Gradient Checkpointing: {'Enabled' if args.use_gradient_checkpointing else 'Disabled'}")
            print(f"  • Distributed Training: {'Enabled' if is_dist() else 'Disabled'}")

        # Training loop
        if is_main:
            print(f"\n🚀 Starting distributed training...")

        for epoch in range(start_epoch, args.epochs + 1):
            if is_main:
                print(f"\n{'=' * 60}")
                print(f"EPOCH {epoch}/{args.epochs}")
                print(f"{'=' * 60}")

            # Set epoch for distributed sampler
            if is_dist():
                if hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)

            # Train one epoch
            train_metrics = train_epoch_ssl_distributed(
                model=model,
                ssl_loss=ssl_loss,
                train_loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                device=device,
                writer=writer,
                args=args,
                is_distributed=is_dist(),
                world_size=world_size,
                rank=local_rank
            )

            # Validate
            if epoch % args.val_interval == 0:
                val_metrics = val_epoch_ssl_distributed(
                    model=model,
                    ssl_loss=ssl_loss,
                    val_loader=val_loader,
                    epoch=epoch,
                    device=device,
                    writer=writer,
                    args=args,
                    is_distributed=is_dist(),
                    world_size=world_size,
                    rank=local_rank
                )

                # Update monitor (only on main process)
                if monitor:
                    monitor.update(epoch, train_metrics, val_metrics)

                # Save best model (only on main process)
                if is_main and val_metrics['total_loss'] < best_val_loss:
                    if best_val_loss != float('inf'):
                        improvement = (best_val_loss - val_metrics['total_loss']) / best_val_loss * 100
                        print(
                            f"✨ New best model! Val loss: {val_metrics['total_loss']:.4f} (improved by {improvement:.1f}%)")
                    else:
                        print(f"✨ First best model! Val loss: {val_metrics['total_loss']:.4f}")

                    best_val_loss = val_metrics['total_loss']

                    # Extract model and ssl_loss from DDP if needed
                    model_to_save = model.module if is_dist() else model
                    ssl_loss_to_save = ssl_loss.module if is_dist() else ssl_loss

                    save_checkpoint_ssl(
                        model=model_to_save,
                        ssl_loss=ssl_loss_to_save,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        best_val_loss=best_val_loss,
                        args=args,
                        filepath=os.path.join(args.results_dir, 'best_model.pth'),
                        monitor_history=monitor.get_history() if monitor else None
                    )

                # Step scheduler
                if scheduler_needs_metric:
                    scheduler.step(val_metrics['total_loss'])
                    current_lr = optimizer.param_groups[0]['lr']
                    if is_main:
                        print(f"  Current LR: {current_lr:.2e}")
            else:
                # Update monitor with train metrics only (only on main process)
                if monitor:
                    monitor.update(epoch, train_metrics, None)

            # Step scheduler
            if not scheduler_needs_metric:
                scheduler.step()

            # Save periodic checkpoint (only on main process)
            if is_main and epoch % args.save_interval == 0:
                model_to_save = model.module if is_dist() else model
                ssl_loss_to_save = ssl_loss.module if is_dist() else ssl_loss

                save_checkpoint_ssl(
                    model=model_to_save,
                    ssl_loss=ssl_loss_to_save,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    args=args,
                    filepath=os.path.join(args.results_dir, f'checkpoint_epoch_{epoch}.pth'),
                    monitor_history=monitor.get_history() if monitor else None
                )

            # Clear cache
            if epoch % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            # Generate report (only on main process)
            if monitor and epoch % 20 == 0:
                monitor.generate_report()
                monitor.plot_metrics()

        # Save final model (only on main process)
        if is_main:
            model_to_save = model.module if is_dist() else model
            ssl_loss_to_save = ssl_loss.module if is_dist() else ssl_loss

            save_checkpoint_ssl(
                model=model_to_save,
                ssl_loss=ssl_loss_to_save,
                optimizer=optimizer,
                scaler=scaler,
                epoch=args.epochs,
                best_val_loss=best_val_loss,
                args=args,
                filepath=os.path.join(args.results_dir, 'final_model.pth'),
                monitor_history=monitor.get_history() if monitor else None
            )

        # Generate final report (only on main process)
        if monitor:
            monitor.generate_report()
            monitor.plot_metrics()

        if is_main:
            print("\n" + "=" * 80)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"Final model saved to: {os.path.join(args.results_dir, 'final_model.pth')}")
            print("\n🎯 Optimizations applied:")
            print("  - Distributed training across multiple GPUs")
            print("  - SimSiam self-distillation (no negatives)")
            print("  - Small-angle rotation regression")
            print("  - Intensity-based augmentations")
            if args.use_amp:
                print(f"  - AMP training with {args.amp_dtype}")
            if not args.deterministic:
                print("  - Non-deterministic mode for speed")
            if args.cudnn_benchmark:
                print("  - cuDNN auto-tuning")
            if not args.use_gradient_checkpointing:
                print("  - No gradient checkpointing (faster)")
            if args.compile_model:
                print(f"  - Model compiled with {args.compile_mode}")
            print("=" * 80)

        if writer:
            writer.close()

        # Clean up distributed training
        if is_dist():
            dist.destroy_process_group()

    except Exception as e:
        is_main = (not is_dist()) or dist.get_rank() == 0
        if is_main:
            print(f"\n❌ Training failed with error: {str(e)}")
            traceback.print_exc()
        if is_dist():
            dist.destroy_process_group()
        raise


if __name__ == "__main__":
    main()