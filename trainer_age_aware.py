import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
import time
from typing import Dict, Optional, List
from tqdm import tqdm
import psutil
import GPUtil
import json
import os
from collections import deque


def is_dist():
    return dist.is_initialized()


def dist_mean_scalar(x: torch.Tensor):
    if is_dist():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, dice_weight=0.0, ce_weight=0.0, focal_weight=0.0, include_background=True, focal_gamma=2.0,
                 class_weights=None, foreground_only=False, loss_config="dice_ce"):
        super().__init__()
        self.foreground_only = foreground_only
        self.num_classes = 87 if foreground_only else 88
        self.loss_config = loss_config

        if loss_config == "dice_ce":
            self.dice_weight = 0.6
            self.ce_weight = 0.4
            self.focal_weight = 0.0
        elif loss_config == "dice_focal":
            self.dice_weight = 0.5
            self.ce_weight = 0.1
            self.focal_weight = 0.4
        elif loss_config == "dice_ce_focal":
            self.dice_weight = 0.4
            self.ce_weight = 0.3
            self.focal_weight = 0.3
        else:
            self.dice_weight = dice_weight
            self.ce_weight = ce_weight
            self.focal_weight = focal_weight

        if self.dice_weight > 0:
            self.dice_loss = DiceLoss(to_onehot_y=False, softmax=True, include_background=True, squared_pred=True,
                                      reduction="mean")
        if self.ce_weight > 0:
            if class_weights is not None:
                self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
            else:
                self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if self.focal_weight > 0:
            self.focal_loss = FocalLoss(include_background=True, to_onehot_y=False, gamma=focal_gamma,
                                        weight=class_weights if class_weights is not None else None, reduction="mean")

    def _create_one_hot_with_ignore(self, labels, num_classes):
        valid_mask = labels != -1
        labels_for_onehot = labels.clone()
        labels_for_onehot[~valid_mask] = 0
        one_hot = F.one_hot(labels_for_onehot.long(), num_classes=num_classes)
        one_hot = one_hot.permute(0, 4, 1, 2, 3)
        one_hot = one_hot * valid_mask.unsqueeze(1)
        return one_hot, valid_mask

    def forward(self, pred, target):
        if len(target.shape) == 5 and target.shape[1] == 1:
            target = target.squeeze(1)
        total_loss = 0.0
        loss_dict = {}

        if self.ce_weight > 0:
            ce = self.ce_loss(pred, target.long())
            total_loss += self.ce_weight * ce
            loss_dict['ce'] = ce.item()
        else:
            loss_dict['ce'] = 0.0

        one_hot_target, valid_mask = self._create_one_hot_with_ignore(target, self.num_classes)
        one_hot_target = one_hot_target.to(dtype=pred.dtype)
        mask = valid_mask.unsqueeze(1).to(dtype=pred.dtype)
        pred_for_dice = pred * mask

        if self.dice_weight > 0:
            dice = self.dice_loss(pred_for_dice, one_hot_target)
            total_loss += self.dice_weight * dice
            loss_dict['dice'] = dice.item()
        else:
            loss_dict['dice'] = 0.0

        if self.focal_weight > 0:
            focal = self.focal_loss(pred * mask, one_hot_target)
            total_loss += self.focal_weight * focal
            loss_dict['focal'] = focal.item()
        else:
            loss_dict['focal'] = 0.0

        loss_dict['total'] = total_loss
        return total_loss, loss_dict


def train_epoch_age_aware(model, source_loader, target_loader, optimizer, epoch, total_epochs, writer, args,
                          device=None, is_distributed=False, world_size=1, rank=0, age_graph_loss=None):
    """Training epoch with age-aware losses"""

    is_main = (not is_distributed) or rank == 0

    # Update epoch for age graph loss if available
    if age_graph_loss is not None:
        age_graph_loss.set_epoch(epoch)

        if is_main:
            print(f"\n🧠 Using Age-Conditioned Graph Prior Loss")
            print(f"  Current epoch: {epoch}, Warmup factor: {age_graph_loss.get_warmup_factor():.3f}")

    model.train()
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
    if age_graph_loss is not None and hasattr(actual_model, "age_conditioner"):
        actual_model.age_conditioner.set_strength(age_graph_loss.get_warmup_factor())

    # Loss accumulators
    total_loss = 0
    seg_loss_total = 0
    target_seg_loss_total = 0
    dice_loss_total = 0
    ce_loss_total = 0
    focal_loss_total = 0

    # Age-aware loss accumulators
    age_loss_total = 0
    volume_loss_total = 0
    shape_loss_total = 0
    weighted_adj_loss_total = 0

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    num_steps = min(len(source_loader), len(target_loader))

    if is_main:
        print(f"\n🚀 Training - Epoch {epoch}/{total_epochs}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        if world_size > 1:
            print(f"  World size: {world_size} GPUs")

    use_target_labels = getattr(args, 'use_target_labels', True)
    target_label_start_epoch = getattr(args, 'target_label_start_epoch', 1)
    target_label_weight = getattr(args, 'target_label_weight', 0.5)

    if use_target_labels and epoch >= target_label_start_epoch:
        progress = (epoch - target_label_start_epoch) / (total_epochs - target_label_start_epoch)
        current_target_weight = target_label_weight * min(1.0, progress * 2)
        if is_main:
            print(f"  📊 Using target labels with weight: {current_target_weight:.3f}")
    else:
        current_target_weight = 0.0

    # Create segmentation loss
    loss_config = getattr(args, 'loss_config', 'dice_focal')
    focal_gamma = getattr(args, 'focal_gamma', 2.0)

    if hasattr(actual_model, 'class_weights') and actual_model.class_weights is not None:
        seg_criterion = CombinedSegmentationLoss(
            loss_config=loss_config, include_background=True,
            focal_gamma=focal_gamma, class_weights=actual_model.class_weights,
            foreground_only=args.foreground_only
        )
    else:
        seg_criterion = CombinedSegmentationLoss(
            loss_config=loss_config,
            include_background=True,
            focal_gamma=focal_gamma,
            foreground_only=args.foreground_only
        )

    start_time = time.time()

    # Progress bar
    pbar = tqdm(range(num_steps), desc=f"Epoch {epoch}") if is_main else range(num_steps)

    # Mixed precision setup
    use_amp = getattr(args, 'use_amp', True)
    amp_dtype = torch.bfloat16 if getattr(args, 'amp_dtype', 'bfloat16') == 'bfloat16' else torch.float16

    for i in pbar:
        # Get batches
        try:
            source_batch = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_batch = next(source_iter)

        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)

        # Extract data with ages
        source_images = source_batch["image"].to(device, non_blocking=True)
        source_labels = source_batch["label"].to(device, non_blocking=True)
        source_ages = source_batch.get("age", torch.tensor([40.0]).repeat(source_images.shape[0], 1)).to(device)

        target_images = target_batch["image"].to(device, non_blocking=True)
        target_labels = target_batch["label"].to(device, non_blocking=True)
        target_ages = target_batch.get("age", torch.tensor([40.0]).repeat(target_images.shape[0], 1)).to(device)

        # Fix label dimensions
        if len(source_labels.shape) == 5 and source_labels.shape[1] == 1:
            source_labels = source_labels.squeeze(1)
        if len(target_labels.shape) == 5 and target_labels.shape[1] == 1:
            target_labels = target_labels.squeeze(1)

        optimizer.zero_grad()

        global_step = (epoch - 1) * num_steps + i

        # Forward pass with mixed precision
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                # Compute source segmentation loss with age
                losses = actual_model.compute_losses(
                    source_images,
                    source_labels,
                    seg_criterion,
                    source_ages=source_ages,
                    step=global_step
                )

                # Extract logits
                source_logits = losses.pop('logits')

                # Compute target segmentation loss if enabled
                target_logits = None
                if current_target_weight > 0:
                    target_logits = actual_model.forward(target_images, target_ages)
                    target_seg_loss, target_seg_components = seg_criterion(target_logits, target_labels)
                    losses['target_seg_loss'] = target_seg_loss
                    losses['total'] = losses['total'] + current_target_weight * target_seg_loss
                    target_seg_loss_total += target_seg_loss.item()

                # Apply age-conditioned graph loss
                if age_graph_loss is not None:
                    # Use target domain predictions and ages
                    if target_logits is not None:
                        graph_total, graph_dict = age_graph_loss(target_logits, target_labels, target_ages)
                    else:
                        graph_total, graph_dict = age_graph_loss(source_logits, source_labels, source_ages)

                    losses['total'] = losses['total'] + graph_total
                    losses['graph_total'] = graph_total

                    # Accumulate age-aware losses
                    if 'volume_loss' in graph_dict:
                        volume_loss_total += graph_dict['volume_loss'].item()
                    if 'shape_loss' in graph_dict:
                        shape_loss_total += graph_dict['shape_loss'].item()
                    if 'weighted_adj_loss' in graph_dict:
                        weighted_adj_loss_total += graph_dict['weighted_adj_loss'].item()
        else:
            # Non-AMP version
            losses = actual_model.compute_losses(
                source_images,
                source_labels,
                seg_criterion,
                source_ages=source_ages,
                step=global_step
            )

            source_logits = losses.pop('logits')

            target_logits = None
            if current_target_weight > 0:
                target_logits = actual_model.forward(target_images, target_ages)
                target_seg_loss, target_seg_components = seg_criterion(target_logits, target_labels)
                losses['target_seg_loss'] = target_seg_loss
                losses['total'] = losses['total'] + current_target_weight * target_seg_loss
                target_seg_loss_total += target_seg_loss.item()

            if age_graph_loss is not None:
                if target_logits is not None:
                    graph_total, graph_dict = age_graph_loss(target_logits, target_labels, target_ages)
                else:
                    graph_total, graph_dict = age_graph_loss(source_logits, source_labels, source_ages)

                losses['total'] = losses['total'] + graph_total
                losses['graph_total'] = graph_total

                if 'volume_loss' in graph_dict:
                    volume_loss_total += graph_dict['volume_loss'].item()
                if 'shape_loss' in graph_dict:
                    shape_loss_total += graph_dict['shape_loss'].item()
                if 'weighted_adj_loss' in graph_dict:
                    weighted_adj_loss_total += graph_dict['weighted_adj_loss'].item()

        # Get loss components
        if 'seg_loss_components' in losses:
            dice_loss_total += losses['seg_loss_components'].get('dice', 0)
            ce_loss_total += losses['seg_loss_components'].get('ce', 0)
            focal_loss_total += losses['seg_loss_components'].get('focal', 0)

        # Get age prediction loss if available
        if 'age_loss' in losses:
            age_loss_total += losses['age_loss'].item()
        if 'volume_loss' in losses:
            volume_loss_total += losses['volume_loss'].item()

        # Backward pass
        losses['total'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # Update weights
        optimizer.step()

        # Update metrics
        total_loss += losses['total'].item()
        seg_loss_total += losses['seg_loss'].item()

        # Clean up
        del source_logits
        if target_logits is not None:
            del target_logits

        # Update progress bar
        if is_main and hasattr(pbar, 'set_postfix'):
            postfix_dict = {
                'loss': f"{losses['total'].item():.4f}",
                'seg': f"{losses['seg_loss'].item():.4f}",
            }
            if 'age_loss' in losses:
                postfix_dict['age'] = f"{losses['age_loss'].item():.4f}"
            if 'volume_loss' in losses:
                postfix_dict['vol'] = f"{losses['volume_loss'].item():.4f}"
            pbar.set_postfix(postfix_dict)

        del losses

        # Periodically clear cache
        if i % 50 == 0:
            torch.cuda.empty_cache()

    # Average metrics
    avg_loss = total_loss / num_steps
    avg_seg_loss = seg_loss_total / num_steps
    avg_target_seg_loss = target_seg_loss_total / num_steps if current_target_weight > 0 else 0

    # Component losses
    avg_dice_loss = dice_loss_total / num_steps
    avg_ce_loss = ce_loss_total / num_steps
    avg_focal_loss = focal_loss_total / num_steps

    # Age-aware losses
    avg_age_loss = age_loss_total / num_steps
    avg_volume_loss = volume_loss_total / num_steps
    avg_shape_loss = shape_loss_total / num_steps
    avg_weighted_adj_loss = weighted_adj_loss_total / num_steps

    # Synchronize metrics across processes if distributed
    if is_distributed:
        metrics_to_sync = torch.tensor([
            avg_loss, avg_seg_loss, avg_target_seg_loss, avg_dice_loss,
            avg_ce_loss, avg_focal_loss, avg_age_loss, avg_volume_loss,
            avg_shape_loss, avg_weighted_adj_loss
        ], device=device)

        dist.all_reduce(metrics_to_sync, op=dist.ReduceOp.SUM)
        metrics_to_sync /= world_size

        (avg_loss, avg_seg_loss, avg_target_seg_loss, avg_dice_loss,
         avg_ce_loss, avg_focal_loss, avg_age_loss, avg_volume_loss,
         avg_shape_loss, avg_weighted_adj_loss) = metrics_to_sync.tolist()

    # Log to tensorboard
    if writer and is_main:
        global_step = (epoch - 1) * num_steps
        writer.add_scalar('train/loss', avg_loss, global_step)
        writer.add_scalar('train/seg_loss', avg_seg_loss, global_step)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

        if current_target_weight > 0:
            writer.add_scalar('train/target_seg_loss', avg_target_seg_loss, global_step)
            writer.add_scalar('train/target_weight', current_target_weight, global_step)

        writer.add_scalar('train/dice_loss', avg_dice_loss, global_step)
        writer.add_scalar('train/ce_loss', avg_ce_loss, global_step)
        writer.add_scalar('train/focal_loss', avg_focal_loss, global_step)

        # Age-aware metrics
        writer.add_scalar('train/age_loss', avg_age_loss, global_step)
        writer.add_scalar('train/volume_loss', avg_volume_loss, global_step)
        writer.add_scalar('train/shape_loss', avg_shape_loss, global_step)
        writer.add_scalar('train/weighted_adj_loss', avg_weighted_adj_loss, global_step)

    elapsed = time.time() - start_time

    if is_main:
        print(f"\n✓ Epoch {epoch} completed in {elapsed:.1f}s")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Segmentation loss: {avg_seg_loss:.4f}")
        print(f"    - Dice: {avg_dice_loss:.4f}")
        print(f"    - CE: {avg_ce_loss:.4f}")
        print(f"    - Focal: {avg_focal_loss:.4f}")
        if current_target_weight > 0:
            print(f"  Target segmentation loss: {avg_target_seg_loss:.4f} (weight: {current_target_weight:.3f})")

        # Print age-aware losses
        print(f"  Age-aware losses:")
        if avg_age_loss > 0:
            print(f"    - Age prediction: {avg_age_loss:.4f}")
        if avg_volume_loss > 0:
            print(f"    - Volume consistency: {avg_volume_loss:.4f}")
        if avg_shape_loss > 0:
            print(f"    - Shape consistency: {avg_shape_loss:.4f}")
        if avg_weighted_adj_loss > 0:
            print(f"    - Weighted adjacency: {avg_weighted_adj_loss:.4f}")

    # Clear cache at epoch end
    torch.cuda.empty_cache()

    # Return metrics
    return {
        'loss': avg_loss,
        'seg_loss': avg_seg_loss,
        'target_seg_loss': avg_target_seg_loss,
        'dice_loss': avg_dice_loss,
        'ce_loss': avg_ce_loss,
        'focal_loss': avg_focal_loss,
        'age_loss': avg_age_loss,
        'volume_loss': avg_volume_loss,
        'shape_loss': avg_shape_loss,
        'weighted_adj_loss': avg_weighted_adj_loss,
        'lr': optimizer.param_groups[0]['lr'],
        'epoch_time': elapsed
    }


def val_epoch_age_aware(model, loader, epoch, writer, args,
                        device=None, is_distributed=False, world_size=1, rank=0):
    """Validate one epoch with age-aware model"""

    is_main = (not is_distributed) or rank == 0
    model.eval()

    # Extract actual model from DDP if needed
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model

    # Metrics
    dice_metric = DiceMetric(
        include_background=True,
        reduction="mean",
        get_not_nans=False
    )

    # Per-class dice metric
    dice_metric_per_class = DiceMetric(
        include_background=True,
        reduction="mean_batch",
        get_not_nans=False
    )

    # Post-processing
    post_pred = AsDiscrete(argmax=True, to_onehot=actual_model.num_classes)

    total_loss = 0
    num_steps = len(loader)

    # Component loss tracking
    dice_loss_total = 0
    ce_loss_total = 0
    focal_loss_total = 0
    age_loss_total = 0

    if is_main:
        print(f"\n📊 Validation - Epoch {epoch}")

    # Enhanced inference settings
    use_tta = getattr(args, 'use_tta', True)
    infer_overlap = getattr(args, 'infer_overlap', 0.7)

    if is_main:
        print(f"  🔧 Enhanced Inference Settings:")
        print(f"    - TTA: {'Enabled (8 augmentations)' if use_tta else 'Disabled'}")
        print(f"    - Sliding Window Overlap: {infer_overlap}")
        print(f"    - Gaussian Blending: Enabled")
        if world_size > 1:
            print(f"    - Distributed validation across {world_size} GPUs")

    start_time = time.time()

    # Get loss configuration
    loss_config = getattr(args, 'loss_config', 'dice_ce')
    focal_gamma = getattr(args, 'focal_gamma', 2.0)

    # Create loss function
    if hasattr(actual_model, 'class_weights') and actual_model.class_weights is not None:
        seg_criterion = CombinedSegmentationLoss(
            loss_config=loss_config,
            include_background=True,
            focal_gamma=focal_gamma,
            class_weights=actual_model.class_weights,
            foreground_only=args.foreground_only
        )
    else:
        seg_criterion = CombinedSegmentationLoss(
            loss_config=loss_config,
            include_background=True,
            focal_gamma=focal_gamma,
            foreground_only=args.foreground_only
        )

    # Track predictions and labels
    sample_dice_scores = []

    # Class-wise prediction counts
    class_prediction_counts = np.zeros(actual_model.num_classes)
    class_label_counts = np.zeros(actual_model.num_classes)

    # Track inference time
    total_inference_time = 0
    inference_times = []

    # Progress bar (only on main process)
    pbar = tqdm(loader, desc="Validation") if is_main else loader

    # Setup mixed precision if enabled
    use_amp = getattr(args, 'use_amp', True)
    amp_dtype = torch.bfloat16 if getattr(args, 'amp_dtype', 'bfloat16') == 'bfloat16' else torch.float16

    # Track ages for analysis
    all_ages = []

    with torch.no_grad():
        for i, batch_data in enumerate(pbar):
            data = batch_data["image"].to(device, non_blocking=True)
            target = batch_data["label"].to(device, non_blocking=True)
            age = batch_data.get("age", torch.tensor([40.0]).repeat(data.shape[0], 1)).to(device)

            all_ages.append(age.cpu().numpy())

            # Fix label dimensions
            if len(target.shape) == 5 and target.shape[1] == 1:
                target = target.squeeze(1)

            # Enhanced sliding window inference
            roi_size = (args.roi_x, args.roi_y, args.roi_z)
            sw_batch_size = args.sw_batch_size

            # Time the inference
            infer_start = time.time()

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    # Simple inference without TTA for speed in validation
                    output = sliding_window_inference(
                        data,
                        roi_size,
                        sw_batch_size,
                        lambda x: model(x, age),  # Pass age to model
                        overlap=infer_overlap,
                        mode="gaussian",
                        sigma_scale=0.125,
                        padding_mode="constant",
                        cval=0.0
                    )
            else:
                output = sliding_window_inference(
                    data,
                    roi_size,
                    sw_batch_size,
                    lambda x: model(x, age),  # Pass age to model
                    overlap=infer_overlap,
                    mode="gaussian",
                    sigma_scale=0.125,
                    padding_mode="constant",
                    cval=0.0
                )

            infer_time = time.time() - infer_start
            total_inference_time += infer_time
            inference_times.append(infer_time)

            # Compute loss
            loss, loss_components = seg_criterion(output, target)
            total_loss += loss.item()
            dice_loss_total += loss_components['dice']
            ce_loss_total += loss_components['ce']
            focal_loss_total += loss_components['focal']

            # For metrics computation
            output_one_hot = [post_pred(i) for i in decollate_batch(output)]

            target_one_hot = []
            masks = []
            for t in decollate_batch(target):  # t: (X,Y,Z) with values in {-1, 0..86}
                mask = (t != -1)  # True = foreground voxel
                t_shift = (t + 1).clamp(min=0)  # {-1,0..86} -> {0..87}
                oh = F.one_hot(t_shift.long(),  # (X,Y,Z, 88)  注意：num_classes = 87 -> +1
                               num_classes=actual_model.num_classes + 1)
                oh = oh.permute(3, 0, 1, 2).float()  # -> (88, X, Y, Z)
                oh = oh[1:, ...]  # drop 背景槽 -> (87, X, Y, Z)
                target_one_hot.append(oh)
                masks.append(mask)

            # 背景体素不计分：把预测、标签在背景处都清零
            for j in range(len(output_one_hot)):
                m = masks[j].unsqueeze(0).float()
                output_one_hot[j] = output_one_hot[j] * m
                target_one_hot[j] = target_one_hot[j] * m

            # Compute dice
            dice_metric(y_pred=output_one_hot, y=target_one_hot)
            dice_metric_per_class(y_pred=output_one_hot, y=target_one_hot)

            # Count predictions per class
            pred = torch.argmax(output, dim=1)
            pred_array = pred.detach()
            target_array = target.detach()
            valid_pixels = (target_array != -1)

            for c in range(actual_model.num_classes):
                pred_np = pred_array.cpu().numpy()
                target_np = target_array.cpu().numpy()
                valid_np = valid_pixels.cpu().numpy()

                class_prediction_counts[c] += np.sum((pred_np[valid_np] == c))
                class_label_counts[c] += np.sum((target_np[valid_np] == c))

            # Update progress bar
            if is_main and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'infer_time': f"{infer_time:.2f}s"})

    # Aggregate metrics
    dice_score = dice_metric.aggregate().item()
    dice_per_class = dice_metric_per_class.aggregate()

    avg_loss = total_loss / num_steps
    avg_dice_loss = dice_loss_total / num_steps
    avg_ce_loss = ce_loss_total / num_steps
    avg_focal_loss = focal_loss_total / num_steps
    avg_age_loss = age_loss_total / num_steps

    # Synchronize metrics across processes if distributed
    if is_distributed:
        # Convert metrics to tensors for synchronization
        metrics_tensor = torch.tensor([
            dice_score, avg_loss, avg_dice_loss, avg_ce_loss,
            avg_focal_loss, avg_age_loss
        ], device=device)

        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        metrics_tensor /= world_size

        (dice_score, avg_loss, avg_dice_loss, avg_ce_loss,
         avg_focal_loss, avg_age_loss) = metrics_tensor.tolist()

        # Synchronize per-class dice scores
        if dice_per_class is not None:
            dist.all_reduce(dice_per_class, op=dist.ReduceOp.SUM)
            dice_per_class /= world_size

    # Calculate average inference time
    avg_inference_time = np.mean(inference_times)
    total_val_time = time.time() - start_time

    # Calculate worst 10 classes dice average
    worst_10_dice_avg = 0.0
    if dice_per_class is not None and dice_per_class.numel() > 0:
        dice_per_class_np = dice_per_class.cpu().numpy()
        sorted_dice = np.sort(dice_per_class_np)
        worst_10_dice_avg = np.mean(sorted_dice[:min(10, len(sorted_dice))])

    # Print results (only on main process)
    if is_main:
        # Age statistics
        all_ages_np = np.concatenate(all_ages)
        print(f"\n📊 Validation Results:")
        print(f"  Age range in validation: [{all_ages_np.min():.1f}, {all_ages_np.max():.1f}] weeks")
        print(f"  Overall Dice Score: {dice_score:.4f}")
        print(f"  Worst 10 Classes Avg Dice: {worst_10_dice_avg:.4f}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"    - Dice: {avg_dice_loss:.4f}")
        print(f"    - CE: {avg_ce_loss:.4f}")
        print(f"    - Focal: {avg_focal_loss:.4f}")

        print(f"\n⏱️ Performance Metrics:")
        print(f"  Total validation time: {total_val_time:.1f}s")
        print(f"  Average inference time per sample: {avg_inference_time:.2f}s")
        print(f"  Total samples: {num_steps}")

        # Print per-class dice scores
        if dice_per_class is not None and dice_per_class.numel() > 0:
            dice_per_class_np = dice_per_class.cpu().numpy()
            print(f"\n📊 Per-class Dice Scores:")

            class_indices = list(range(actual_model.num_classes))
            class_dices = dice_per_class_np

            sorted_indices = np.argsort(class_dices)

            print(f"  Worst 10 classes:")
            for idx, i in enumerate(sorted_indices[:min(10, len(sorted_indices))]):
                actual_class = class_indices[i]
                original_class = actual_class + 1 if args.foreground_only else actual_class
                print(f"    {idx + 1}. Class {actual_class} (orig {original_class}): {class_dices[i]:.4f}")

            print(f"  Best 5 classes:")
            for idx, i in enumerate(sorted_indices[-min(5, len(sorted_indices)):]):
                actual_class = class_indices[i]
                original_class = actual_class + 1 if args.foreground_only else actual_class
                print(f"    {idx + 1}. Class {actual_class} (orig {original_class}): {class_dices[i]:.4f}")

    # Reset metrics
    dice_metric.reset()
    dice_metric_per_class.reset()

    # Log to tensorboard (only on main process)
    if writer and is_main:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/dice', dice_score, epoch)
        writer.add_scalar('val/dice_loss', avg_dice_loss, epoch)
        writer.add_scalar('val/ce_loss', avg_ce_loss, epoch)
        writer.add_scalar('val/focal_loss', avg_focal_loss, epoch)
        writer.add_scalar('val/worst_10_dice_avg', worst_10_dice_avg, epoch)
        writer.add_scalar('val/inference_time_avg', avg_inference_time, epoch)

        if dice_per_class is not None and dice_per_class.numel() > 0:
            for c in range(actual_model.num_classes):
                writer.add_scalar(f'val/dice_class_{c}', dice_per_class_np[c], epoch)

    if is_main:
        elapsed = time.time() - start_time
        print(f"\n✓ Validation completed in {elapsed:.1f}s")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Dice Score: {dice_score:.4f}")
        print(f"  Worst 10 Classes Avg: {worst_10_dice_avg:.4f}")

    # Return metrics
    val_metrics = {
        'loss': avg_loss,
        'dice': dice_score,
        'dice_loss': avg_dice_loss,
        'ce_loss': avg_ce_loss,
        'focal_loss': avg_focal_loss,
        'age_loss': avg_age_loss,
        'dice_per_class': dice_per_class_np.tolist() if dice_per_class is not None else None,
        'sample_dice_scores': sample_dice_scores,
        'class_prediction_counts': class_prediction_counts.tolist(),
        'class_label_counts': class_label_counts.tolist(),
        'val_time': total_val_time,
        'worst_10_dice_avg': worst_10_dice_avg,
        'avg_inference_time': avg_inference_time,
        'total_inference_time': total_inference_time,
    }

    return val_metrics


def save_checkpoint_simplified(model, optimizer, epoch, best_acc, args, filepath,
                               dice_history=None, additional_info=None):
    """Save simplified checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'base_model_state_dict': model.base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'args': args,
        'num_classes': model.num_classes,
        'foreground_only': model.foreground_only,
        'enhanced_class_weights': model.enhanced_class_weights,
        'loss_config': getattr(args, 'loss_config', 'dice_focal'),
        'focal_gamma': getattr(args, 'focal_gamma', 2.0),
        'dice_history': dice_history if dice_history is not None else {},
    }

    # Add additional info if provided
    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")