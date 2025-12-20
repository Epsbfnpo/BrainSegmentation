import math
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from trm_core import TRMWeightedLoss


def is_finite(tensor: torch.Tensor) -> bool:
    return torch.isfinite(tensor).all().item()


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def train_epoch(
        target_model: torch.nn.Module,
        source_model: torch.nn.Module,
        loader,
        optimizer,
        trm_manager,
        device: torch.device,
        scaler: torch.cuda.amp.GradScaler,
        epoch: int,
        *,
        warmup_epochs: int,
        accumulation_steps: int = 1,
) -> float:
    target_model.train()
    source_model.eval()

    loss_fn = TRMWeightedLoss(num_classes=trm_manager.num_classes).to(device)
    total_loss = 0.0
    steps = 0

    # 1. Freeze stats after warmup
    if epoch > warmup_epochs and not trm_manager.frozen:
        trm_manager.freeze_statistics()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            source_logits = source_model(images)
            if not trm_manager.frozen:
                trm_manager.update_statistics(source_logits, labels)

            risk_map = trm_manager.compute_risk_map(source_logits, labels)

        with torch.cuda.amp.autocast():
            logits = target_model(images)
            loss = loss_fn(logits, labels, risk_map)
            loss = loss / accumulation_steps

        if not is_finite(loss):
            print(f"Warning: Non-finite loss {loss.item()} at epoch {epoch}, step {steps}")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if (steps + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps
        steps += 1

    return float(total_loss / max(1, steps))


def validate_epoch(model: torch.nn.Module, loader, device: torch.device, *, num_classes: int) -> Tuple[
    float, Dict[str, float]]:
    model.eval()

    # Metrics calculation on CPU to save GPU memory
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # We skip total_loss calculation during validation to avoid OOM
    # Calculating loss on full volume requires holding gradients/logits which is too heavy
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]  # Keep labels on CPU for now if possible, or move later

            # 1. Sliding Window Inference (Efficient)
            # roi_size should match training crop size usually, e.g. (128, 128, 128)
            # Ensure we use the same sw_batch_size as in args if passed, here hardcoded or default
            logits = sliding_window_inference(
                inputs=images,
                roi_size=(128, 128, 128),
                sw_batch_size=1,
                predictor=model,
                overlap=0.25
            )

            # 2. GPU Aggregation -> Reduce to Index immediately
            # Logits: (B, 87, D, H, W) ~ 60GB.
            # Argmax: (B, 1, D, H, W) ~ 0.7GB.
            # HUGE SAVINGS HERE.
            val_pred_idx = torch.argmax(logits, dim=1, keepdim=True)

            # 3. Move lightweight index tensors to CPU
            val_pred_idx = val_pred_idx.cpu()
            val_labels = labels.long().cpu()  # shape (B, 1, D, H, W)

            # Free GPU memory immediately
            del logits
            del images
            torch.cuda.empty_cache()

            # 4. CPU-side One-Hot Conversion & Label Handling
            # Handle background -1 -> 0
            val_labels[val_labels < 0] = 0

            # Expand to One-Hot on CPU (System RAM is cheap)
            # F.one_hot adds dim at the end, so we permute: (B, D, H, W, C) -> (B, C, D, H, W)
            preds_onehot = F.one_hot(val_pred_idx.squeeze(1), num_classes=num_classes).permute(0, 4, 1, 2, 3)
            target_onehot = F.one_hot(val_labels.squeeze(1), num_classes=num_classes).permute(0, 4, 1, 2, 3)

            # 5. Compute Dice on CPU
            dice_metric(y_pred=preds_onehot, y=target_onehot)

            steps += 1

    mean_dice = dice_metric.aggregate().item() if steps > 0 else 0.0
    dice_metric.reset()

    # Return 0.0 for loss since we skipped it to save memory
    return 0.0, {"dice": mean_dice}