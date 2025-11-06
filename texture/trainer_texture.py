"""Training utilities for the texture-centric pipeline."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

__all__ = [
    "build_loss",
    "build_dice_metric",
    "train_epoch",
    "validate",
    "save_checkpoint",
    "load_checkpoint",
]


class ForegroundDiceCELoss(nn.Module):
    def __init__(self, num_classes: int, dice_weight: float = 0.6, ce_weight: float = 0.4, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = labels >= 0
        if not mask.any():
            return logits.new_zeros(())

        valid_labels = torch.clamp(labels, min=0)
        target_one_hot = F.one_hot(valid_labels, num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
        mask = mask.unsqueeze(1).to(logits.dtype)
        target_one_hot = target_one_hot.to(logits.dtype) * mask
        probs = torch.softmax(logits, dim=1) * mask

        dims = tuple(range(2, probs.ndim))
        intersection = (probs * target_one_hot).sum(dim=dims)
        denominator = (probs + target_one_hot).sum(dim=dims)
        dice = (2 * intersection + self.eps) / (denominator + self.eps)
        dice_loss = 1 - dice.mean()

        ce_loss = F.cross_entropy(logits, labels, ignore_index=-1, reduction="mean")
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


def build_loss(num_classes: int, include_background: bool, foreground_only: bool) -> torch.nn.Module:
    if foreground_only:
        return ForegroundDiceCELoss(num_classes=num_classes)
    return DiceCELoss(
        include_background=include_background,
        to_onehot_y=True,
        softmax=True,
        lambda_dice=0.6,
        lambda_ce=0.4,
        squared_pred=True,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    )


def build_dice_metric(num_classes: int, include_background: bool, foreground_only: bool) -> Optional[DiceMetric]:
    if foreground_only:
        return None
    return DiceMetric(include_background=include_background, reduction="mean", get_not_nans=False)


def _prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    images = batch["image"].to(device)
    labels = batch["label"].to(device=device, dtype=torch.long)
    if labels.ndim == 5 and labels.size(1) == 1:
        labels = labels.squeeze(1)
    texture_stats = batch.get("texture_stats")
    if texture_stats is not None:
        texture_stats = texture_stats.to(device)
    domain = batch.get("domain")
    if domain is not None:
        domain = domain.to(device).view(-1)
    return images, labels, texture_stats, domain


def _alignment_loss(embeddings: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
    if embeddings is None or domain is None:
        device = embeddings.device if embeddings is not None else domain.device
        return torch.zeros((), device=device)

    unique_domains = domain.unique(sorted=True)
    if unique_domains.numel() < 2:
        return torch.zeros((), device=embeddings.device)

    means = []
    for dom in unique_domains:
        mask = domain == dom
        if mask.any():
            means.append(embeddings[mask].mean(dim=0, keepdim=True))
    if len(means) < 2:
        return torch.zeros((), device=embeddings.device)
    stacked = torch.cat(means, dim=0)
    diffs = stacked.unsqueeze(0) - stacked.unsqueeze(1)
    mse = (diffs ** 2).mean()
    return mse


def _distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _aggregate_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if _distributed_is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    *,
    amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
    domain_loss_weight: float,
    embed_align_weight: float,
    stats_align_weight: float,
    grl_lambda: float,
    is_distributed: bool,
    world_size: int,
) -> Dict[str, float]:
    model.train()
    sums = {
        "total_loss": 0.0,
        "seg_loss": 0.0,
        "domain_loss": 0.0,
        "align_loss": 0.0,
        "stats_align_loss": 0.0,
        "domain_acc": 0.0,
        "num_batches": 0.0,
    }

    if amp and scaler is None:
        scaler = torch.cuda.amp.GradScaler()

    for batch in loader:
        images, labels, texture_stats, domain = _prepare_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits, texture_embedding, domain_logits = model(
                images,
                texture_stats=texture_stats,
                grl_lambda=grl_lambda,
            )
            seg_loss = loss_fn(logits, labels)
            domain_loss = torch.tensor(0.0, device=device)
            domain_acc = torch.tensor(0.0, device=device)
            if domain is not None:
                domain_loss = F.cross_entropy(domain_logits, domain)
                preds = domain_logits.argmax(dim=1)
                domain_acc = (preds == domain).float().mean()
            align_loss = torch.tensor(0.0, device=device)
            if embed_align_weight > 0.0:
                align_loss = _alignment_loss(texture_embedding, domain)
            stats_align_loss = torch.tensor(0.0, device=device)
            if stats_align_weight > 0.0 and texture_stats is not None:
                norm_stats = F.normalize(texture_stats, dim=1)
                stats_align_loss = _alignment_loss(norm_stats, domain)

            total_loss = (
                seg_loss
                + domain_loss_weight * domain_loss
                + embed_align_weight * align_loss
                + stats_align_weight * stats_align_loss
            )

        if amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        sums["total_loss"] += float(total_loss.detach().item())
        sums["seg_loss"] += float(seg_loss.detach().item())
        sums["domain_loss"] += float(domain_loss.detach().item())
        sums["align_loss"] += float(align_loss.detach().item())
        sums["stats_align_loss"] += float(stats_align_loss.detach().item())
        sums["domain_acc"] += float(domain_acc.detach().item())
        sums["num_batches"] += 1.0

    metrics_tensor = torch.tensor(
        [
            sums["total_loss"],
            sums["seg_loss"],
            sums["domain_loss"],
            sums["align_loss"],
            sums["stats_align_loss"],
            sums["domain_acc"],
            sums["num_batches"],
        ],
        device=device,
        dtype=torch.float32,
    )

    if is_distributed and world_size > 1:
        metrics_tensor = _aggregate_tensor(metrics_tensor)

    totals = metrics_tensor.tolist()
    num_batches = max(1.0, totals[-1])

    return {
        "total_loss": totals[0] / num_batches,
        "seg_loss": totals[1] / num_batches,
        "domain_loss": totals[2] / num_batches,
        "align_loss": totals[3] / num_batches,
        "stats_align_loss": totals[4] / num_batches,
        "domain_acc": totals[5] / num_batches,
        "num_batches": int(num_batches),
    }


def _compute_foreground_dice(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    mask = labels >= 0
    if not mask.any():
        return logits.new_zeros(())
    preds = logits.argmax(dim=1)
    preds_fg = preds[mask]
    labels_fg = labels[mask].clamp(min=0)
    preds_one_hot = F.one_hot(preds_fg, num_classes=num_classes).to(logits.dtype)
    labels_one_hot = F.one_hot(labels_fg, num_classes=num_classes).to(logits.dtype)
    intersection = (preds_one_hot * labels_one_hot).sum(dim=0) * 2.0
    denominator = preds_one_hot.sum(dim=0) + labels_one_hot.sum(dim=0)
    dice = (intersection + eps) / (denominator + eps)
    return dice.mean()


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    loss_fn,
    dice_metric,
    device: torch.device,
    *,
    roi_size,
    sw_batch_size: int,
    amp: bool,
    num_classes: int,
    is_distributed: bool,
    world_size: int,
    foreground_only: bool,
) -> Tuple[float, float]:
    model.eval()
    num_batches = 0
    val_loss = 0.0

    if not foreground_only:
        post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=num_classes)
        post_label = AsDiscrete(to_onehot=True, n_classes=num_classes)
    else:
        post_pred = post_label = None

    dice_sum = 0.0
    dice_count = 0.0

    for batch in loader:
        images, labels, texture_stats, domain = _prepare_batch(batch, device)

        with torch.cuda.amp.autocast(enabled=amp):
            logits, _, _ = model(images, texture_stats=texture_stats, grl_lambda=0.0)
            loss = loss_fn(logits, labels)

        val_loss += float(loss.detach().item())
        num_batches += 1

        if foreground_only:
            batch_dice = _compute_foreground_dice(logits, labels, num_classes)
            dice_sum += float(batch_dice.item())
            dice_count += 1.0
        else:
            preds = decollate_batch(logits)
            labs = decollate_batch(labels)
            preds = [post_pred(pred) for pred in preds]
            labs = [post_label(label) for label in labs]
            dice_metric(y_pred=preds, y=labs)

    if foreground_only:
        metric = dice_sum / max(1.0, dice_count)
        metric_tensor = torch.tensor([metric, dice_count], device=device, dtype=torch.float32)
    else:
        metric_raw = dice_metric.aggregate()
        metric = float(metric_raw.mean().item()) if metric_raw.numel() > 0 else 0.0
        dice_metric.reset()
        metric_tensor = torch.tensor([metric, 1.0 if metric_raw.numel() > 0 else 0.0], device=device, dtype=torch.float32)

    loss_tensor = torch.tensor([val_loss, float(num_batches)], device=device, dtype=torch.float32)

    if is_distributed and world_size > 1:
        loss_tensor = _aggregate_tensor(loss_tensor)
        metric_tensor = _aggregate_tensor(metric_tensor)

    avg_loss = loss_tensor[0].item() / max(1.0, loss_tensor[1].item())
    metric_sum = metric_tensor[0].item()
    metric_count = max(1.0, metric_tensor[1].item())

    return avg_loss, metric_sum / metric_count


def _get_model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _load_model_state(model: torch.nn.Module, state_dict: Dict) -> None:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)


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
            "state_dict": _get_model_state(model),
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
    map_location: str | torch.device = "cpu",
) -> Dict:
    checkpoint = torch.load(path, map_location=map_location)
    _load_model_state(model, checkpoint["state_dict"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
