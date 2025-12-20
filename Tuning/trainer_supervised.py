"""
Training and validation functions for supervised segmentation
FIXED: Always use sliding window option and proper ignore_index handling
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
from typing import Dict, Optional
import gc
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from metrics import SegmentationMetrics


def calculate_macro_dice(pred: torch.Tensor, target: torch.Tensor,
                         num_classes: int, ignore_index: int = -1) -> float:
    """Calculate macro-averaged Dice score

    ENHANCED: Handle ignore_index=-1 properly
    """
    # Create mask for valid pixels
    valid_mask = (target != ignore_index).float()

    # Get predictions
    if pred.dim() == 5:  # (B, C, H, W, D)
        pred_argmax = torch.argmax(pred, dim=1)
    else:
        pred_argmax = pred

    # Calculate per-class dice
    dice_scores = []

    for c in range(num_classes):
        # Get binary masks for this class
        pred_c = (pred_argmax == c).float() * valid_mask
        target_c = (target == c).float() * valid_mask

        # Calculate dice
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union > 0:
            dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
            dice_scores.append(dice.item())

    # Return macro average (only over classes that exist)
    if dice_scores:
        return np.mean(dice_scores)
    else:
        return 0.0


def train_epoch_supervised(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    writer,
    args,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    accumulation_steps: int = 1,
    monitor=None,
    is_main_process: bool = True
) -> Dict[str, float]:
    """Train one epoch with improved metrics calculation

    ENHANCED: Better handling of ignore_index=-1
    """

    model.train()

    # Metrics tracking
    metrics = {
        'loss': 0.0,
        'dice': 0.0,
        'macro_dice': 0.0,
        'grad_norm': 0.0,
        'lr': optimizer.param_groups[0]['lr']
    }

    # For foreground-only mode
    include_bg = True
    dice_metric = DiceMetric(include_background=include_bg, reduction="mean")

    num_steps = len(train_loader)
    use_amp = scaler is not None

    if is_main_process:
        print(f"\n🚀 Training - Epoch {epoch}")
        print(f"  Learning rate: {metrics['lr']:.6f}")
        print(f"  Gradient accumulation steps: {accumulation_steps}")
        print(f"  Mixed precision: {use_amp}")

    start_time = time.time()

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}") if is_main_process else train_loader

    # Zero gradients at start
    optimizer.zero_grad()
    accumulated_loss = 0.0

    # Track per-class performance
    class_dice_tracker = {i: [] for i in range(args.out_channels)}

    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Check for NaN
        if torch.isnan(images).any():
            print(f"\n⚠️ WARNING: NaN in input at step {i}")
            continue

        # Forward pass with optional mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        accumulated_loss += loss.item()

        if torch.isnan(loss):
            print(f"\n⚠️ WARNING: NaN loss at step {i}")
            continue

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == num_steps:
            # Gradient clipping
            if use_amp:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.clip
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.clip
                )
                optimizer.step()

            # Zero gradients for next accumulation
            optimizer.zero_grad()

            # Calculate online metrics with accumulated batch
            with torch.no_grad():
                # Calculate macro dice (same as validation)
                macro_dice = calculate_macro_dice(outputs, labels, args.out_channels, ignore_index=-1)

                # Original dice calculation
                pred_argmax = torch.argmax(outputs, dim=1)
                valid_mask = labels != -1

                if valid_mask.any():
                    # Convert predictions and labels to one-hot
                    pred_onehot = F.one_hot(
                        pred_argmax.clamp(0, args.out_channels-1),
                        num_classes=args.out_channels
                    ).permute(0, 4, 1, 2, 3).float()

                    labels_clamped = labels.clamp(0, args.out_channels-1)
                    labels_onehot = F.one_hot(
                        labels_clamped,
                        num_classes=args.out_channels
                    ).permute(0, 4, 1, 2, 3).float()

                    # Apply valid mask
                    pred_onehot = pred_onehot * valid_mask.unsqueeze(1)
                    labels_onehot = labels_onehot * valid_mask.unsqueeze(1)

                    dice_metric(y_pred=pred_onehot, y=labels_onehot)
                    batch_dice = dice_metric.aggregate().item()
                    dice_metric.reset()

                    # Track per-class performance
                    for c in range(args.out_channels):
                        pred_c = (pred_argmax == c).float() * valid_mask.float()
                        target_c = (labels == c).float() * valid_mask.float()

                        intersection = (pred_c * target_c).sum()
                        union = pred_c.sum() + target_c.sum()

                        if union > 0:
                            class_dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
                            class_dice_tracker[c].append(class_dice.item())
                else:
                    batch_dice = 0.0
                    macro_dice = 0.0

            # Update metrics
            metrics['loss'] += accumulated_loss * accumulation_steps
            metrics['dice'] += batch_dice
            metrics['macro_dice'] += macro_dice
            metrics['grad_norm'] += grad_norm.item()
            accumulated_loss = 0.0

            # Update progress bar
            if is_main_process:
                pbar.set_postfix({
                    'loss': f"{loss.item() * accumulation_steps:.4f}",
                    'dice': f"{batch_dice:.4f}",
                    'macro': f"{macro_dice:.4f}",
                    'grad': f"{grad_norm.item():.2f}",
                    'scale': f"{scaler.get_scale():.0f}" if use_amp else "N/A"
                })

            # Log to tensorboard
            if writer is not None and (i + 1) % (10 * accumulation_steps) == 0:
                global_step = (epoch - 1) * (num_steps // accumulation_steps) + (i + 1) // accumulation_steps
                writer.add_scalar('train/loss', loss.item() * accumulation_steps, global_step)
                writer.add_scalar('train/dice', batch_dice, global_step)
                writer.add_scalar('train/macro_dice', macro_dice, global_step)
                writer.add_scalar('train/grad_norm', grad_norm.item(), global_step)
                writer.add_scalar('train/lr', metrics['lr'], global_step)
                if use_amp:
                    writer.add_scalar('train/amp_scale', scaler.get_scale(), global_step)

    # Average metrics
    actual_steps = num_steps // accumulation_steps
    for key in metrics:
        if key != 'lr':
            metrics[key] /= actual_steps

    elapsed = time.time() - start_time

    # Report per-class statistics
    num_zero_dice = sum(1 for c, scores in class_dice_tracker.items()
                        if scores and np.mean(scores) < 0.1)

    if is_main_process:
        print(f"\n✓ Epoch {epoch} completed in {elapsed:.1f}s")
        print(f"  Average loss: {metrics['loss']:.4f}")
        print(f"  Average Dice: {metrics['dice']:.4f}")
        print(f"  Macro Dice (aligned): {metrics['macro_dice']:.4f}")
        print(f"  Classes with Dice < 0.1: {num_zero_dice}/{args.out_channels}")
        print(f"  Average gradient norm: {metrics['grad_norm']:.4f}")

    # Update monitor if provided
    if monitor is not None:
        monitor.update(epoch, metrics, val_metrics=None)

    return metrics


def val_epoch_supervised(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    metrics_evaluator: SegmentationMetrics,
    epoch: int,
    device: torch.device,
    writer,
    args,
    monitor=None,
    is_main_process: bool = True
) -> Dict[str, float]:
    """Validate one epoch with sliding window inference

    FIXED: Option to always use sliding window regardless of image size
    """

    model.eval()

    metrics = {
        'loss': 0.0,
        'dice': 0.0,
        'macro_dice': 0.0,
        'dice_per_class': {},
        'num_zero_dice': 0,
    }

    # For foreground-only mode
    include_bg = True
    dice_metric = DiceMetric(include_background=include_bg, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=include_bg, reduction="mean_batch")

    # Store all dice scores for per-class calculation
    all_dice_scores = []
    class_dice_tracker = {i: [] for i in range(args.out_channels)}

    num_steps = len(val_loader)

    if is_main_process:
        print(f"\n📊 Validation - Epoch {epoch}")
        if args.always_use_sliding_window:
            print(f"  Using sliding window inference (overlap={args.overlap})")

    start_time = time.time()

    with torch.no_grad():
        val_iter = tqdm(val_loader, desc="Validation") if is_main_process else val_loader
        for i, batch in enumerate(val_iter):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            if torch.isnan(images).any():
                print(f"\n⚠️ WARNING: NaN in validation input at step {i}")
                continue

            # FIXED: Option to always use sliding window or conditionally
            if args.always_use_sliding_window or images.shape[2:] != (args.roi_x, args.roi_y, args.roi_z):
                # Use sliding window inference
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=(args.roi_x, args.roi_y, args.roi_z),
                    sw_batch_size=args.sliding_window_batch_size,
                    predictor=model,
                    overlap=args.overlap,  # Higher overlap for better boundaries
                    mode="gaussian",  # Gaussian weighting for smoother fusion
                    padding_mode="constant",
                    cval=0,  # Pad with 0 for image
                )
            else:
                # Direct inference if image size matches ROI and not forcing sliding window
                outputs = model(images)

            # Optional: Test-time augmentation
            if hasattr(args, 'use_tta') and args.use_tta:
                # Simple TTA with flips
                outputs_flipped = []
                for axis in [2, 3, 4]:  # Spatial axes
                    images_flipped = torch.flip(images, dims=[axis])

                    if args.always_use_sliding_window:
                        outputs_flip = sliding_window_inference(
                            inputs=images_flipped,
                            roi_size=(args.roi_x, args.roi_y, args.roi_z),
                            sw_batch_size=args.sliding_window_batch_size,
                            predictor=model,
                            overlap=args.overlap,
                            mode="gaussian",
                            padding_mode="constant",
                            cval=0,
                        )
                    else:
                        outputs_flip = model(images_flipped)

                    outputs_flip = torch.flip(outputs_flip, dims=[axis])
                    outputs_flipped.append(outputs_flip)

                # Average predictions
                outputs = (outputs + sum(outputs_flipped)) / (len(outputs_flipped) + 1)

            # Calculate loss
            loss = criterion(outputs, labels)
            metrics['loss'] += loss.item()

            # Calculate macro dice
            macro_dice = calculate_macro_dice(outputs, labels, args.out_channels, ignore_index=-1)
            metrics['macro_dice'] += macro_dice

            # Convert predictions to one-hot
            pred_argmax = torch.argmax(outputs, dim=1)

            # Create valid mask
            valid_mask = (labels != -1).float()

            # Calculate per-class dice scores
            batch_dice_scores = []

            for c in range(args.out_channels):
                # For each class/region
                pred_c = (pred_argmax == c).float()
                label_c = (labels == c).float()

                # Apply valid mask
                pred_c = pred_c * valid_mask
                label_c = label_c * valid_mask

                # Calculate dice for this class
                intersection = (pred_c * label_c).sum(dim=(1, 2, 3))
                pred_sum = pred_c.sum(dim=(1, 2, 3))
                label_sum = label_c.sum(dim=(1, 2, 3))

                # Only calculate dice if the class exists
                dice_c = torch.where(
                    (pred_sum + label_sum) > 0,
                    (2.0 * intersection + 1e-5) / (pred_sum + label_sum + 1e-5),
                    torch.ones_like(intersection)
                )

                dice_value = dice_c.mean().item()
                batch_dice_scores.append(dice_value)

                # Track if class exists in this batch
                if (label_sum > 0).any():
                    class_dice_tracker[c].append(dice_value)

            # Store per-class dice scores
            all_dice_scores.append(batch_dice_scores)

            # Calculate overall dice
            pred_onehot = F.one_hot(
                pred_argmax.clamp(0, args.out_channels-1),
                num_classes=args.out_channels
            ).permute(0, 4, 1, 2, 3).float()

            labels_clamped = labels.clamp(0, args.out_channels-1)
            labels_onehot = F.one_hot(
                labels_clamped,
                num_classes=args.out_channels
            ).permute(0, 4, 1, 2, 3).float()

            # Apply valid mask
            pred_onehot = pred_onehot * valid_mask.unsqueeze(1)
            labels_onehot = labels_onehot * valid_mask.unsqueeze(1)

            dice_metric(y_pred=pred_onehot, y=labels_onehot)

    # Aggregate metrics
    metrics['loss'] /= num_steps
    metrics['macro_dice'] /= num_steps
    metrics['dice'] = dice_metric.aggregate().item()

    if dist.is_initialized():
        local_metrics = torch.tensor(
            [metrics['loss'], metrics['dice']],
            device=device,
            dtype=torch.float32
        )
        dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        metrics['loss'] = local_metrics[0].item() / world_size
        metrics['dice'] = local_metrics[1].item() / world_size

    # Calculate per-class dice scores
    if class_dice_tracker:
        for c, scores in class_dice_tracker.items():
            if scores:
                mean_dice = np.mean(scores)
                if args.out_channels == 87:
                    class_name = f'region_{c+1}'
                elif args.out_channels == 15:
                    amos_organs = [
                        "Background", "Spleen", "R_Kidney", "L_Kidney", "Gallbladder", "Esophagus",
                        "Liver", "Stomach", "Aorta", "IVC", "Pancreas", "R_Adrenal", "L_Adrenal",
                        "Duodenum", "Bladder"
                    ]
                    class_name = amos_organs[c] if c < len(amos_organs) else f'class_{c}'
                else:
                    class_name = f'class_{c}'
                metrics['dice_per_class'][class_name] = float(mean_dice)

                # Count zero dice classes
                if mean_dice < 0.1:
                    metrics['num_zero_dice'] += 1

    elapsed = time.time() - start_time

    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('val/loss', metrics['loss'], epoch)
        writer.add_scalar('val/dice', metrics['dice'], epoch)
        writer.add_scalar('val/macro_dice', metrics['macro_dice'], epoch)
        writer.add_scalar('val/num_zero_dice', metrics['num_zero_dice'], epoch)

    # Log per-class dice scores (only top and bottom 5 to avoid clutter)
    if metrics['dice_per_class'] and writer is not None:
        sorted_classes = sorted(metrics['dice_per_class'].items(), key=lambda x: x[1])

        # Log worst 5
        for class_name, dice_score in sorted_classes[:5]:
            writer.add_scalar(f'val/dice_worst/{class_name}', dice_score, epoch)

        # Log best 5
        for class_name, dice_score in sorted_classes[-5:]:
            writer.add_scalar(f'val/dice_best/{class_name}', dice_score, epoch)

    if is_main_process:
        print(f"\n✓ Validation completed in {elapsed:.1f}s")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Mean Dice: {metrics['dice']:.4f}")
        print(f"  Macro Dice: {metrics['macro_dice']:.4f}")
        print(f"  Classes with Dice < 0.1: {metrics['num_zero_dice']}/{args.out_channels}")

    # Show worst performing classes
    if metrics['dice_per_class'] and is_main_process:
        sorted_classes = sorted(metrics['dice_per_class'].items(), key=lambda x: x[1])
        print("\n  Worst performing regions:")
        for class_name, dice_score in sorted_classes[:5]:
            print(f"    {class_name}: {dice_score:.4f}")

        print("\n  Best performing regions:")
        for class_name, dice_score in sorted_classes[-5:]:
            print(f"    {class_name}: {dice_score:.4f}")

    # Reset metrics
    dice_metric.reset()
    dice_metric_batch.reset()

    # Update monitor if provided
    if monitor is not None:
        metrics['total_loss'] = metrics['loss']
        monitor.update(epoch, train_metrics={}, val_metrics=metrics)

    return metrics


def save_checkpoint_supervised(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_dice: float,
    args,
    filepath: str,
    val_metrics: Optional[Dict] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler: Optional[object] = None
):
    """Save training checkpoint with scheduler state"""

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_dice': best_val_dice,
        'args': args,
        'val_metrics': val_metrics
    }

    # Save scaler state if using AMP
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    # Save scheduler state
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")


def load_checkpoint_supervised(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler: Optional[object] = None
) -> Dict:
    """Load training checkpoint with scheduler state"""

    print(f"📂 Loading checkpoint: {filepath}")
    checkpoint = torch.load(filepath, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint
