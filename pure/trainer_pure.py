"""Training utilities for pure fine-tuning."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete


def build_loss(num_classes: int, include_background: bool) -> DiceCELoss:
    return DiceCELoss(
        include_background=include_background,
        to_onehot_y=True,
        softmax=True,
        lambda_dice=0.5,
        lambda_ce=0.5,
        squared_pred=True,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    )


def build_dice_metric(num_classes: int, include_background: bool) -> DiceMetric:
    return DiceMetric(include_background=include_background, reduction="mean", get_not_nans=False)


@torch.no_grad()
def _prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device)
    labels = batch["label"].long().to(device)
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    return images, labels


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: DiceCELoss,
    device: torch.device,
    *,
    amp: bool = True,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    if amp and scaler is None:
        scaler = torch.cuda.amp.GradScaler()

    for batch in loader:
        images, labels = _prepare_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        epoch_loss += float(loss.detach().item())
        num_batches += 1

    return epoch_loss / max(1, num_batches)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    loss_fn: DiceCELoss,
    dice_metric: DiceMetric,
    device: torch.device,
    *,
    roi_size,
    sw_batch_size: int = 2,
    amp: bool = True,
    num_classes: int = 88,
) -> Tuple[float, float]:
    model.eval()
    num_batches = 0
    val_loss = 0.0

    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=num_classes)
    post_label = AsDiscrete(to_onehot=True, n_classes=num_classes)

    for batch in loader:
        images, labels = _prepare_batch(batch, device)

        if amp:
            with torch.cuda.amp.autocast():
                logits = sliding_window_inference(images, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model)
                loss = loss_fn(logits, labels)
        else:
            logits = sliding_window_inference(images, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model)
            loss = loss_fn(logits, labels)

        val_loss += float(loss.detach().item())
        num_batches += 1

        preds = decollate_batch(logits)
        labs = decollate_batch(labels)
        preds = [post_pred(pred) for pred in preds]
        labs = [post_label(label) for label in labs]
        dice_metric(y_pred=preds, y=labs)

    metric_tensor = dice_metric.aggregate()
    metric = float(metric_tensor.mean().item()) if metric_tensor.numel() > 0 else 0.0
    dice_metric.reset()

    return val_loss / max(1, num_batches), metric


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    *,
    best_metric: Optional[float] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    *,
    map_location: str = "cpu",
) -> Dict:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


__all__ = [
    "build_loss",
    "build_dice_metric",
    "train_epoch",
    "validate",
    "save_checkpoint",
    "load_checkpoint",
]
