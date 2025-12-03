import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.losses import DiceLoss
from monai.metrics import HausdorffDistanceMetric
from monai.inferers import sliding_window_inference


class LocalHausdorffDistanceMetric(HausdorffDistanceMetric):
    """
    HD95 metric that skips DDP all-gather but still concatenates local buffers
    so aggregate() receives a Tensor instead of a list.
    """

    def _sync(self):
        if len(self._buffers) > 0:
            # Concatenate buffered tensors locally; no cross-rank communication.
            self._synced_tensors = torch.cat(self._buffers, dim=0)
        else:
            self._synced_tensors = torch.empty(0, device="cpu")


def compute_dice_score(y_pred: torch.Tensor, y: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Compute per-sample, per-class Dice on one-hot tensors."""

    dims = tuple(range(2, y_pred.ndim))
    intersection = torch.sum(y_pred * y, dim=dims)
    cardinality = torch.sum(y_pred + y, dim=dims)
    return (2.0 * intersection + smooth) / (cardinality + smooth)


def check_finite(name: str, tensor) -> bool:
    if tensor is None:
        return True
    if torch.isfinite(tensor).all():
        return True
    print(f"[NaN DETECTED] {name} has NaN/Inf", flush=True)
    return False


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True, include_background=False, squared_pred=True)
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim == 4:
            labels = labels.unsqueeze(1)

        # Deep supervision label resize to match logits
        if logits.shape[2:] != labels.shape[2:]:
            labels = F.interpolate(labels.float(), size=logits.shape[2:], mode="nearest")

        if labels.ndim == 5:
            labels = labels.squeeze(1)
        labels = labels.long()

        labels_rect = labels.clone()
        labels_rect[labels_rect < 0] = 0

        loss_ce = self.ce(logits, labels)
        loss_dice = self.dice(logits, labels_rect.unsqueeze(1))
        return 0.5 * loss_ce + 0.5 * loss_dice


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, ema=None):
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(images)
            if isinstance(logits, (list, tuple)):
                loss = sum(loss_fn(l, labels) for l in logits) / len(logits)
            else:
                loss = loss_fn(logits, labels)

        if not check_finite("loss", loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        scaler.step(optimizer)
        scaler.update()

        if ema:
            ema.update()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


def validate(model, loader, device, roi_size):
    model.eval()

    total_dice = 0.0
    total_hd95 = 0.0
    count_dice = 0
    count_hd95 = 0

    hd95_metric = LocalHausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    def _predictor(x):
        out = model(x)
        if isinstance(out, (list, tuple)):
            return out[-1]
        return out

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            labels[labels < 0] = 0

            logits = sliding_window_inference(images, roi_size, 1, _predictor, overlap=0.25, mode="gaussian")
            if isinstance(logits, (list, tuple)):
                logits = logits[-1]

            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)
            num_classes = logits.shape[1]

            pred_oh = F.one_hot(pred.squeeze(1), num_classes).permute(0, 4, 1, 2, 3)
            labels_oh = F.one_hot(labels.squeeze(1), num_classes).permute(0, 4, 1, 2, 3)

            dice_tensor = compute_dice_score(pred_oh[:, 1:], labels_oh[:, 1:])
            total_dice += dice_tensor.mean().item()
            count_dice += 1

            hd95_metric.reset()
            hd95_metric(y_pred=pred_oh, y=labels_oh)
            hd95_result = hd95_metric.aggregate()
            if isinstance(hd95_result, torch.Tensor):
                val = hd95_result.item()
                if np.isfinite(val):
                    total_hd95 += val
                    count_hd95 += 1

    avg_dice = total_dice / max(count_dice, 1)
    avg_hd95 = total_hd95 / max(count_hd95, 1)
    return avg_dice, avg_hd95
