from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import torch.distributed as dist


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value


class ExponentialMovingAverage:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = model
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self.register()

    def register(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def _check_finite(tensor: torch.Tensor) -> bool:
    return torch.isfinite(tensor).all()


class CombinedLoss(nn.Module):
    """Dice + CE with background ignored (label -1)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels_dice = labels.clone()
        labels_dice[labels_dice < 0] = 0
        if labels_dice.ndim == 4:
            labels_dice = labels_dice.unsqueeze(1)
        dice_loss = self.dice(logits, labels_dice)
        ce_loss = self.ce(logits, labels.long())
        return 0.5 * dice_loss + 0.5 * ce_loss


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    ema: ExponentialMovingAverage | None,
):
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        img = batch["image"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(img)
            loss = loss_fn(logits, label)

        if not _check_finite(loss):
            print(f"[WARN] Non-finite loss detected ({loss.item()}), skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        scaler.step(optimizer)
        scaler.update()

        if ema:
            ema.update()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


def _prepare_for_metric(pred: torch.Tensor, label: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Map labels: -1 -> 0 (bg), 0..86 -> 1..87 to avoid dropping first class when include_background=False
    label_mapped = label.clone()
    label_mapped[label_mapped < 0] = -1
    label_mapped = label_mapped + 1  # now bg=0, class0=1, ...
    if label_mapped.ndim == 4:
        label_mapped = label_mapped.unsqueeze(1)

    pred_shifted = pred + 1  # predicted classes 0..86 -> 1..87

    label_onehot = F.one_hot(label_mapped.long().squeeze(1), num_classes + 1).permute(0, 4, 1, 2, 3)
    pred_onehot = F.one_hot(pred_shifted.long().squeeze(1), num_classes + 1).permute(0, 4, 1, 2, 3)
    # Remove voxels that were background in original label
    mask = (label >= 0).unsqueeze(1)
    return pred_onehot * mask, label_onehot * mask


def validate(model: nn.Module, loader, device: torch.device, args) -> float:
    model.eval()
    metric = DiceMetric(include_background=False, reduction="mean")

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            label = batch["label"].to(device)

            logits = sliding_window_inference(img, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.25)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)

            y_pred, y_true = _prepare_for_metric(pred, label, args.out_channels)
            metric(y_pred=y_pred, y=y_true)

    dice = metric.aggregate().to(device)
    metric.reset()
    return reduce_mean(dice).item()


__all__ = [
    "CombinedLoss",
    "ExponentialMovingAverage",
    "train_epoch",
    "validate",
]
