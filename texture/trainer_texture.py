"""Training utilities for the texture-centric pipeline."""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
from monai.data import MetaTensor, decollate_batch
from tqdm.auto import tqdm
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


def _to_plain_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if isinstance(tensor, MetaTensor):
        return tensor.as_tensor()
    return tensor


def _prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    images = _to_plain_tensor(batch["image"]).to(device)
    labels = _to_plain_tensor(batch["label"]).to(device=device, dtype=torch.long)
    if labels.ndim == 5 and labels.size(1) == 1:
        labels = labels.squeeze(1)
    texture_stats = batch.get("texture_stats")
    if texture_stats is not None:
        texture_stats = _to_plain_tensor(texture_stats).to(device=device, dtype=torch.float32)
        texture_stats = torch.nan_to_num(texture_stats, nan=0.0, posinf=1e6, neginf=-1e6)
        texture_stats = texture_stats.clamp_(-10.0, 10.0)
    domain = batch.get("domain")
    if domain is not None:
        domain = _to_plain_tensor(domain).to(device).view(-1)
    return images, labels, texture_stats, domain


def _alignment_loss(embeddings: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
    if embeddings is None or domain is None:
        device = embeddings.device if embeddings is not None else domain.device
        return torch.zeros((), device=device)

    embeddings = _to_plain_tensor(embeddings)
    domain = _to_plain_tensor(domain)

    embeddings = embeddings.float()
    domain = domain.long()

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
    use_tqdm: bool = False,
    progress_desc: str | None = None,
    debug_interval: int = 0,
    debug_fn: Optional[Callable[[str], None]] = None,
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

    progress = None
    if use_tqdm:
        desc = progress_desc or "Train"
        progress = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
        iterator = enumerate(progress, start=1)
    else:
        iterator = enumerate(loader, start=1)

    for step, batch in iterator:
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
                norm_stats = F.normalize(texture_stats, dim=1, eps=1e-6)
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

        if progress is not None:
            batches = max(1.0, sums["num_batches"])
            progress.set_postfix(
                loss=f"{sums['total_loss']/batches:.3f}",
                seg=f"{sums['seg_loss']/batches:.3f}",
                dom=f"{sums['domain_loss']/batches:.3f}"
            )

        if debug_interval and debug_fn and step % max(1, debug_interval) == 0:
            with torch.no_grad():
                grad_norm = 0.0
                grad_params = [p.grad for p in model.parameters() if p.grad is not None]
                if grad_params:
                    grad_norm = torch.norm(
                        torch.stack([g.detach().float().norm(2) for g in grad_params])
                    ).item()
                label_min = float(labels.min().item())
                label_max = float(labels.max().item())
                unique_labels = torch.unique(labels).to(device="cpu", dtype=torch.long)
                fg_fraction = float((labels >= 0).float().mean().item())
                if domain is not None:
                    domain_cpu = domain.detach().to(device="cpu", dtype=torch.long)
                    max_bins = int(domain_cpu.max().item()) + 1 if domain_cpu.numel() > 0 else 1
                    domain_counts = torch.bincount(domain_cpu, minlength=max(1, max_bins))
                    domain_summary = ", ".join(
                        f"d{idx}={int(count)}" for idx, count in enumerate(domain_counts.tolist())
                    )
                else:
                    domain_summary = "n/a"
            debug_fn(
                (
                    f"step {step:04d}: total={total_loss.item():.4f}, seg={seg_loss.item():.4f}, "
                    f"domain={domain_loss.item():.4f}, align={align_loss.item():.4f}, "
                    f"stats_align={stats_align_loss.item():.4f}, dom_acc={domain_acc.item():.3f}, "
                    f"fg_frac={fg_fraction:.3f}, label_range=[{label_min:.0f},{label_max:.0f}], "
                    f"unique_labels={sorted(unique_labels.tolist())[:12]}{'...' if unique_labels.numel() > 12 else ''}, "
                    f"domain_counts={domain_summary}, grad_norm={grad_norm:.3e}"
                )
            )

    if progress is not None:
        progress.close()

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
    # use float32 to avoid overflow when AMP casts logits to float16
    preds_one_hot = F.one_hot(preds_fg, num_classes=num_classes).to(
        device=logits.device, dtype=torch.float32
    )
    labels_one_hot = F.one_hot(labels_fg, num_classes=num_classes).to(
        device=logits.device, dtype=torch.float32
    )
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
    use_tqdm: bool = False,
    progress_desc: str | None = None,
    debug_batches: int = 0,
    debug_fn: Optional[Callable[[str], None]] = None,
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

    progress = None
    if use_tqdm:
        desc = progress_desc or "Validation"
        progress = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
        iterator = enumerate(progress, start=1)
    else:
        iterator = enumerate(loader, start=1)

    for step, batch in iterator:
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

        if progress is not None:
            processed = max(1.0, num_batches)
            avg_loss = val_loss / processed
            progress.set_postfix(loss=f"{avg_loss:.3f}")

        if debug_fn and step <= max(0, debug_batches):
            with torch.no_grad():
                label_min = float(labels.min().item())
                label_max = float(labels.max().item())
                unique_labels = torch.unique(labels).to(device="cpu", dtype=torch.long)
                fg_fraction = float((labels >= 0).float().mean().item())
                debug_fn(
                    (
                        f"val-step {step:03d}: loss={loss.item():.4f}, fg_frac={fg_fraction:.3f}, "
                        f"label_range=[{label_min:.0f},{label_max:.0f}], "
                        f"unique_labels={sorted(unique_labels.tolist())[:12]}"
                        f"{'...' if unique_labels.numel() > 12 else ''}"
                    )
                )

    if progress is not None:
        progress.close()

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
