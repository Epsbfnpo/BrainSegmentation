#!/usr/bin/env python3
"""
Main script for supervised fine-tuning on dHCP dataset
FIXED: Added parameters for rotation angle and LR swap file
"""
import argparse
import os
import sys
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tensorboardX import SummaryWriter
from monai.utils import set_determinism
import numpy as np
from datetime import datetime
import traceback
import gc

# Import supervised components
from model_supervised import create_supervised_model, freeze_encoder
from data_loader_supervised import get_supervised_dataloaders, get_optimal_batch_size
from loss_functions import get_loss_function
from trainer_supervised import train_epoch_supervised, val_epoch_supervised, save_checkpoint_supervised
from metrics import SegmentationMetrics
from monitoring_supervised import SupervisedMonitor


def get_parser():
    """Get argument parser with enhanced options"""
    parser = argparse.ArgumentParser(description='Supervised Fine-tuning on dHCP Dataset (Registered)')

    # Basic training parameters
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Mini-batch size (per GPU step)')
    parser.add_argument('--accumulation_steps', default=4, type=int,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--encoder_lr_ratio', default=0.3, type=float,
                        help='Learning rate ratio for encoder (0.3 = 30% of base lr)')
    parser.add_argument('--freeze_encoder_epochs', default=5, type=int,
                        help='Number of epochs to freeze encoder')

    # Model parameters
    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of input channels (T2w only)')
    parser.add_argument('--out_channels', default=87, type=int,
                        help='Number of output channels (87 brain regions)')
    parser.add_argument('--feature_size', default=48, type=int,
                        help='Feature size for transformer')
    parser.add_argument('--roi_x', default=96, type=int,
                        help='ROI size in x direction')
    parser.add_argument('--roi_y', default=96, type=int,
                        help='ROI size in y direction')
    parser.add_argument('--roi_z', default=96, type=int,
                        help='ROI size in z direction')
    parser.add_argument('--dropout_rate', default=0.1, type=float,
                        help='Dropout rate')

    # Data parameters
    parser.add_argument('--data_split_json', default='/scratch3/liu275/Data/dHCP_registered/dHCP_split.json',
                        type=str, help='Path to dHCP data split JSON file')
    parser.add_argument('--class_prior_json',
                        default='/datasets/work/hb-nhmrc-dhcp/work/liu275/dHCP_registered_class_prior_standard.json',
                        type=str, help='Path to class prior JSON file')
    parser.add_argument('--laterality_pairs_json', default='./dhcp_lr_swap.json',
                        type=str, help='Path to LR swap pairs JSON file')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--cache_rate', default=0.1, type=float,
                        help='Cache rate for data loading')
    parser.add_argument('--use_persistent_dataset', action='store_true', default=False,
                        help='Use PersistentDataset for caching')
    parser.add_argument('--cache_dir', default='./cache', type=str,
                        help='Directory for persistent dataset cache')
    parser.add_argument('--clean_cache', action='store_true',
                        help='Clean cache before training')
    parser.add_argument('--foreground_only', action='store_true', default=True,
                        help='Use foreground only mode')
    parser.add_argument('--class_aware_sampling', action='store_true', default=True,
                        help='Use class-aware sampling for rare classes')
    parser.add_argument('--debug_samples', default=0, type=int,
                        help='Use limited samples for debugging (0=use all)')

    # Augmentation parameters
    parser.add_argument('--max_rotation_angle', default=0.1, type=float,
                        help='Maximum rotation angle in radians (0.1 = ~6 degrees)')

    # Pretrained model
    parser.add_argument('--pretrained_model', type=str,
                        default='/datasets/work/hb-nhmrc-dhcp/work/liu275/SSL/results_ssl/dHCP_distributed/best_model.pth',
                        help='Path to SSL pretrained model')
    parser.add_argument('--resume_training', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Loss function
    parser.add_argument('--loss_type', default='tversky_focal',
                        choices=['dice', 'ce', 'dice_ce', 'focal', 'dice_focal', 'tversky_focal'],
                        help='Loss function type')
    parser.add_argument('--dice_smooth', default=1e-5, type=float,
                        help='Smoothing factor for Dice loss')
    parser.add_argument('--dice_weight', default=0.5, type=float,
                        help='Weight for Dice loss in combined loss')
    parser.add_argument('--ce_weight', default=0.5, type=float,
                        help='Weight for CE loss in combined loss')
    parser.add_argument('--focal_gamma', default=2.5, type=float,
                        help='Gamma for focal loss')
    parser.add_argument('--tversky_alpha', default=0.5, type=float,
                        help='Alpha for Tversky loss (FP weight)')
    parser.add_argument('--tversky_beta', default=0.5, type=float,
                        help='Beta for Tversky loss (FN weight)')
    parser.add_argument('--class_weights', default='auto', type=str,
                        help='Path to class weights file or "auto" for automatic calculation')
    parser.add_argument('--weight_power', default=0.75, type=float,
                        help='Power for class weight calculation')
    parser.add_argument('--max_weight', default=20.0, type=float,
                        help='Maximum class weight')
    parser.add_argument('--min_weight', default=0.1, type=float,
                        help='Minimum class weight')

    # Training settings
    parser.add_argument('--clip', default=1.0, type=float,
                        help='Gradient clipping')
    parser.add_argument('--val_interval', default=1, type=int,
                        help='Validation interval')
    parser.add_argument('--save_interval', default=5, type=int,
                        help='Save checkpoint interval')
    parser.add_argument('--early_stopping_patience', default=40, type=int,
                        help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--auto_batch_size', action='store_true', default=False,
                        help='Automatically find optimal batch size')

    # Resolution parameters
    parser.add_argument('--target_spacing', nargs=3, type=float,
                        default=[0.5, 0.5, 0.5],
                        help='Target voxel spacing in mm')

    # LR scheduler
    parser.add_argument('--lr_scheduler', default='cosine_no_restart',
                        choices=['cosine', 'cosine_no_restart', 'plateau'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_cosine_t0', default=25, type=int,
                        help='Number of iterations for the first restart')
    parser.add_argument('--lr_patience', default=8, type=int,
                        help='Patience for ReduceLROnPlateau')
    parser.add_argument('--lr_min', default=3e-6, type=float,
                        help='Minimum learning rate')
    parser.add_argument('--lr_factor', default=0.5, type=float,
                        help='Factor for ReduceLROnPlateau')

    # Dynamic encoder LR
    parser.add_argument('--dynamic_encoder_lr', action='store_true', default=False,
                        help='Dynamically adjust encoder LR during training')
    parser.add_argument('--encoder_lr_boost_epoch', default=50, type=int,
                        help='Epoch to boost encoder LR')
    parser.add_argument('--encoder_lr_boost_ratio', default=0.5, type=float,
                        help='Encoder LR ratio after boost')

    # Output
    parser.add_argument('--results_dir', default='./results_supervised/', type=str,
                        help='Directory for results')
    parser.add_argument('--exp_name', default=None, type=str,
                        help='Experiment name')
    parser.add_argument('--visualize_weights', action='store_true', default=False,
                        help='Visualize class weights')

    # Evaluation
    parser.add_argument('--sliding_window_batch_size', default=4, type=int,
                        help='Batch size for sliding window inference')
    parser.add_argument('--overlap', default=0.7, type=float,
                        help='Overlap for sliding window inference')
    parser.add_argument('--always_use_sliding_window', action='store_true', default=True,
                        help='Always use sliding window inference in validation')
    parser.add_argument('--use_post_processing', action='store_true', default=False,
                        help='Use post-processing (connected components, fill holes)')
    parser.add_argument('--use_tta', action='store_true', default=False,
                        help='Use test-time augmentation')

    # Advanced
    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='Use exponential moving average of model weights')
    parser.add_argument('--ema_decay', default=0.999, type=float,
                        help='EMA decay rate')

    return parser


def validate_configuration(args):
    """Validate configuration and print warnings"""

    # Check if LR swap file exists
    if args.laterality_pairs_json and not os.path.exists(args.laterality_pairs_json):
        print(f"⚠️ WARNING: LR swap file not found: {args.laterality_pairs_json}")
        print(f"   LR flip augmentation will be disabled")

    # Validate spacing
    if min(args.target_spacing) < 0.4:
        print(f"⚠️ WARNING: Very fine spacing {args.target_spacing} may require more GPU memory")

    # Check class prior file
    if args.class_aware_sampling and not os.path.exists(args.class_prior_json):
        print(f"⚠️ WARNING: Class prior file not found: {args.class_prior_json}")
        print(f"   Class-aware sampling will use uniform weights")

    # Validate rotation angle
    if args.max_rotation_angle > 0.2:  # ~11 degrees
        print(f"⚠️ WARNING: Large rotation angle {args.max_rotation_angle} rad for registered data")
        print(f"   Consider reducing for registered data (recommended: 0.05-0.1)")


def main():
    """Main training function with enhanced features"""
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # Validate configuration
    validate_configuration(args)

    # Create experiment name if not provided
    if args.exp_name is None:
        args.exp_name = f"supervised_dHCP_registered_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create directories
    args.results_dir = os.path.join(args.results_dir, args.exp_name)
    os.makedirs(args.results_dir, exist_ok=True)

    # Set random seed
    set_determinism(seed=42)

    # Save configuration
    config_path = os.path.join(args.results_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "=" * 80)
    print("SUPERVISED FINE-TUNING ON dHCP DATASET (REGISTERED - FIXED)")
    print("=" * 80)
    print(f"Experiment: {args.exp_name}")
    print(f"Results directory: {args.results_dir}")
    print(f"Training epochs: {args.epochs}")
    print(
        f"Batch size: {args.batch_size} (×{args.accumulation_steps} accumulation = {args.batch_size * args.accumulation_steps} effective)")
    print(f"Initial learning rate: {args.lr}")
    print(f"Loss function: {args.loss_type}")
    print(f"Target spacing: {args.target_spacing}")
    print(f"Max rotation angle: {args.max_rotation_angle} rad (~{args.max_rotation_angle * 180 / np.pi:.1f}°)")
    print(f"LR Scheduler: {args.lr_scheduler}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print(f"Class-aware sampling: {args.class_aware_sampling}")
    print(f"Encoder LR ratio: {args.encoder_lr_ratio}")
    print(f"Freeze encoder epochs: {args.freeze_encoder_epochs}")
    print(f"Sliding window overlap: {args.overlap}")
    print(f"Always use sliding window: {args.always_use_sliding_window}")
    print("=" * 80)

    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This code requires GPU.")

    device = torch.device("cuda:0")
    print(f"\n🖥️ GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.results_dir, 'tensorboard'))

    # Create monitor
    monitor = SupervisedMonitor(args.results_dir, num_classes=args.out_channels)
    print(f"\n📊 Created training monitor")

    # Create model first (needed for auto batch size)
    print("\n🏗️ Creating model...")
    model = create_supervised_model(args).to(device)

    # Auto find optimal batch size if requested
    if args.auto_batch_size:
        print("\n🔎 Finding optimal batch size...")
        optimal_batch_size = get_optimal_batch_size(
            model, device,
            roi_size=[args.roi_x, args.roi_y, args.roi_z],
            start_batch_size=args.batch_size,
            use_amp=args.use_amp
        )
        print(f"✓ Optimal batch size: {optimal_batch_size}")
        args.batch_size = optimal_batch_size

    # Get data loaders
    print("\n📊 Creating data loaders...")
    train_loader, val_loader = get_supervised_dataloaders(args)
    print(f"✓ Training samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")

    # Create loss function
    print("\n📉 Creating loss function...")
    criterion = get_loss_function(args, device)

    # Create optimizer with differential learning rates
    print("\n🔧 Creating optimizer with differential learning rates...")
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if 'swinViT' in name:  # Encoder parameters
            encoder_params.append(param)
        else:  # Decoder parameters
            decoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.lr * args.encoder_lr_ratio, 'name': 'encoder'},
        {'params': decoder_params, 'lr': args.lr, 'name': 'decoder'}
    ], weight_decay=args.weight_decay, betas=(0.9, 0.999))

    print(f"  Encoder parameters: {len(encoder_params)} with lr={args.lr * args.encoder_lr_ratio:.6f}")
    print(f"  Decoder parameters: {len(decoder_params)} with lr={args.lr:.6f}")

    # Create AMP scaler if needed
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # Learning rate scheduler
    if args.lr_scheduler == 'cosine_no_restart':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr_min
        )
        scheduler_needs_metric = False
        print(f"✓ Using CosineAnnealingLR (NO RESTART) with T_max={args.epochs}, eta_min={args.lr_min}")
    elif args.lr_scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # For Dice score
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.lr_min,
            verbose=True
        )
        scheduler_needs_metric = True
        print(f"✓ Using ReduceLROnPlateau (patience={args.lr_patience}, min_lr={args.lr_min})")
    else:  # Old cosine with restart
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.lr_cosine_t0,
            T_mult=2,
            eta_min=args.lr_min
        )
        scheduler_needs_metric = False
        print(f"✓ Using CosineAnnealingWarmRestarts (T_0={args.lr_cosine_t0})")

    # Create metrics evaluator
    metrics_evaluator = SegmentationMetrics(
        num_classes=args.out_channels,
        include_background=True,  # For consistency
        device=device,
        ignore_index=-1  # Background pixels are now -1
    )

    # Optional: Create EMA model
    ema_model = None
    if args.use_ema:
        print("\n📊 Creating EMA model...")
        from model_supervised import ModelEMA
        ema_model = ModelEMA(model, decay=args.ema_decay)

    # Resume training if needed
    start_epoch = 1
    best_val_dice = 0.0
    early_stopping_counter = 0

    if args.resume_training and args.checkpoint:
        print(f"\n📂 Resuming from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_dice = checkpoint.get('best_val_dice', 0.0)
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✓ Resuming from epoch {start_epoch} with best val Dice = {best_val_dice:.4f}")

    # Training loop
    print(f"\n🚀 Starting supervised fine-tuning...")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"EPOCH {epoch}/{args.epochs}")
            print(f"{'=' * 60}")

            # Dynamic encoder LR adjustment
            if args.dynamic_encoder_lr and epoch == args.encoder_lr_boost_epoch:
                print(f"📈 Boosting encoder LR ratio to {args.encoder_lr_boost_ratio}")
                for param_group in optimizer.param_groups:
                    if param_group.get('name') == 'encoder':
                        param_group['lr'] = args.lr * args.encoder_lr_boost_ratio

            # Optionally freeze encoder for initial epochs
            if epoch <= args.freeze_encoder_epochs:
                print(f"❄️ Encoder frozen for first {args.freeze_encoder_epochs} epochs")
                for param in encoder_params:
                    param.requires_grad = False
            else:
                for param in encoder_params:
                    param.requires_grad = True

            # Train one epoch
            train_metrics = train_epoch_supervised(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch,
                device=device,
                writer=writer,
                args=args,
                scaler=scaler,
                accumulation_steps=args.accumulation_steps,
                monitor=monitor
            )

            # Update EMA
            if ema_model is not None:
                ema_model.update()

            # Validate
            if epoch % args.val_interval == 0:
                # Use EMA model for validation if available
                eval_model = ema_model.model if ema_model is not None else model

                val_metrics = val_epoch_supervised(
                    model=eval_model,
                    val_loader=val_loader,
                    criterion=criterion,
                    metrics_evaluator=metrics_evaluator,
                    epoch=epoch,
                    device=device,
                    writer=writer,
                    args=args,
                    monitor=monitor
                )

                # Save best model
                if val_metrics['dice'] > best_val_dice:
                    improvement = val_metrics['dice'] - best_val_dice
                    best_val_dice = val_metrics['dice']
                    early_stopping_counter = 0

                    save_checkpoint_supervised(
                        model=eval_model if ema_model is not None else model,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_val_dice=best_val_dice,
                        args=args,
                        filepath=os.path.join(args.results_dir, 'best_model.pth'),
                        val_metrics=val_metrics,
                        scaler=scaler,
                        scheduler=scheduler
                    )
                    print(f"✨ New best model! Val Dice: {best_val_dice:.4f} (+{improvement:.4f})")
                else:
                    early_stopping_counter += 1
                    print(
                        f"⚠️ No improvement for {early_stopping_counter} epochs (patience: {args.early_stopping_patience})")

                # Step scheduler
                if scheduler_needs_metric:
                    scheduler.step(val_metrics['dice'])

                # Early stopping
                if early_stopping_counter >= args.early_stopping_patience:
                    print(f"\n⚠️ Early stopping triggered after {epoch} epochs")
                    print(f"Best validation Dice: {best_val_dice:.4f}")
                    break

            # Step scheduler
            if not scheduler_needs_metric:
                scheduler.step()

            # Save periodic checkpoint
            if epoch % args.save_interval == 0:
                save_checkpoint_supervised(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_dice=best_val_dice,
                    args=args,
                    filepath=os.path.join(args.results_dir, f'checkpoint_epoch_{epoch}.pth'),
                    val_metrics=val_metrics if epoch % args.val_interval == 0 else None,
                    scaler=scaler,
                    scheduler=scheduler
                )

            # Generate monitoring reports periodically
            if epoch % 10 == 0:
                monitor.generate_report()
                monitor.plot_metrics()

            # Clear cache
            if epoch % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        traceback.print_exc()

        # Save emergency checkpoint
        save_checkpoint_supervised(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_dice=best_val_dice,
            args=args,
            filepath=os.path.join(args.results_dir, f'emergency_epoch_{epoch}.pth'),
            scaler=scaler,
            scheduler=scheduler
        )
        raise

    # Save final model
    save_checkpoint_supervised(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs if epoch >= args.epochs else epoch,
        best_val_dice=best_val_dice,
        args=args,
        filepath=os.path.join(args.results_dir, 'final_model.pth'),
        scaler=scaler,
        scheduler=scheduler
    )

    # Final monitoring report and plots
    print("\n📊 Generating final reports and plots...")
    monitor.generate_report()
    monitor.plot_metrics()

    print("\n" + "=" * 80)
    print("✅ SUPERVISED FINE-TUNING COMPLETED!")
    print("=" * 80)
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Final model saved to: {os.path.join(args.results_dir, 'final_model.pth')}")
    print("=" * 80)

    writer.close()


if __name__ == "__main__":
    main()