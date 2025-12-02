from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric


# --- NaN/Inf guard ---
def check_finite(name: str, tensor: torch.Tensor) -> bool:
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
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim == 5:
            labels = labels.squeeze(1)

        valid_mask = labels >= 0
        labels_rect = labels.clone()
        labels_rect[~valid_mask] = 0

        loss_ce = self.ce(logits, labels_rect.long())
        loss_dice = self.dice(logits, labels_rect.unsqueeze(1))
        return 0.5 * loss_ce + 0.5 * loss_dice


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    ema: Optional[ExponentialMovingAverage] = None,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        img = batch["image"].to(device)
        lbl = batch["label"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(img)
            if isinstance(logits, (list, tuple)):
                loss = sum(loss_fn(l, lbl) for l in logits) / len(logits)
            else:
                loss = loss_fn(logits, lbl)

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


def validate(model: nn.Module, loader, device: torch.device, roi_size: Sequence[int]) -> Tuple[float, float]:
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            lbl = batch["label"].to(device).long()
            lbl[lbl < 0] = 0

            logits = sliding_window_inference(img, roi_size, 1, model, overlap=0.25, mode="gaussian")
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)
            n_class = logits.shape[1]
            pred_oh = F.one_hot(pred.squeeze(1), n_class).permute(0, 4, 1, 2, 3)
            lbl_oh = F.one_hot(lbl.squeeze(1), n_class).permute(0, 4, 1, 2, 3)

            dice_metric(y_pred=pred_oh[:, 1:], y=lbl_oh[:, 1:])
            hd95_metric(y_pred=pred_oh[:, 1:], y=lbl_oh[:, 1:])

    return dice_metric.aggregate().item(), hd95_metric.aggregate().item()
