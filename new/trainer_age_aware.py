from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value


class CombinedSegmentationLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 *,
                 class_weights: Optional[torch.Tensor] = None,
                 foreground_only: bool = True,
                 loss_config: str = "dice_focal",
                 focal_gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.foreground_only = foreground_only
        self.loss_config = loss_config
        self.class_weights = class_weights

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
            raise ValueError(f"Unsupported loss_config: {loss_config}")

        if self.dice_weight > 0:
            self.dice_loss = DiceLoss(
                to_onehot_y=True,
                softmax=True,
                include_background=not foreground_only,
                squared_pred=True,
                ignore_index=-1,
            )
        if self.ce_weight > 0:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        if self.focal_weight > 0:
            self.focal_loss = FocalLoss(
                include_background=not foreground_only,
                to_onehot_y=False,
                gamma=focal_gamma,
                weight=class_weights,
                reduction="mean",
                ignore_index=-1,
            )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        if labels.ndim == 5 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        result: Dict[str, torch.Tensor] = {}

        if self.ce_weight > 0:
            ce = self.ce_loss(logits, labels.long())
        else:
            ce = torch.zeros(1, device=logits.device)
        result["ce"] = ce

        if self.dice_weight > 0:
            dice = self.dice_loss(logits, labels.long())
        else:
            dice = torch.zeros(1, device=logits.device)
        result["dice"] = dice

        if self.focal_weight > 0:
            focal = self.focal_loss(logits, labels.long())
        else:
            focal = torch.zeros(1, device=logits.device)
        result["focal"] = focal

        total = self.dice_weight * dice + self.ce_weight * ce + self.focal_weight * focal
        result["total"] = total
        return result


def train_epoch(model: nn.Module,
                loader,
                optimizer: torch.optim.Optimizer,
                loss_fn: CombinedSegmentationLoss,
                prior_loss,
                *,
                device: torch.device,
                epoch: int,
                use_amp: bool = True) -> Dict[str, float]:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    running = {
        "loss": 0.0,
        "seg": 0.0,
        "prior": 0.0,
        "dice": 0.0,
        "ce": 0.0,
        "focal": 0.0,
        "volume": 0.0,
        "shape": 0.0,
        "edge": 0.0,
        "spectral": 0.0,
    }
    steps = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        ages = batch.get("age")
        if ages is None:
            raise RuntimeError("Data loader must provide 'age' key")
        ages = ages.to(device).view(-1)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            seg_losses = loss_fn(logits, labels)
            probs = torch.softmax(logits, dim=1)
            prior_dict = prior_loss(probs, ages) if prior_loss is not None else {"total": torch.zeros_like(seg_losses["total"]),
                                                                                "volume": torch.zeros_like(seg_losses["total"]),
                                                                                "shape": torch.zeros_like(seg_losses["total"]),
                                                                                "edge": torch.zeros_like(seg_losses["total"]),
                                                                                "spectral": torch.zeros_like(seg_losses["total"])}
            loss = seg_losses["total"] + prior_dict["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
        scaler.step(optimizer)
        scaler.update()

        running["loss"] += loss.detach().item()
        running["seg"] += seg_losses["total"].detach().item()
        running["dice"] += seg_losses["dice"].detach().item()
        running["ce"] += seg_losses["ce"].detach().item()
        running["focal"] += seg_losses["focal"].detach().item()
        running["prior"] += prior_dict["total"].detach().item()
        running["volume"] += prior_dict.get("volume", torch.zeros(1, device=device)).detach().item()
        running["shape"] += prior_dict.get("shape", torch.zeros(1, device=device)).detach().item()
        running["edge"] += prior_dict.get("edge", torch.zeros(1, device=device)).detach().item()
        running["spectral"] += prior_dict.get("spectral", torch.zeros(1, device=device)).detach().item()
        steps += 1

    for key in running:
        value = torch.tensor(running[key] / max(steps, 1), device=device)
        running[key] = float(reduce_mean(value))

    running["steps"] = steps
    running["epoch"] = epoch
    return running


def validate_epoch(model: nn.Module,
                   loader,
                   *,
                   device: torch.device,
                   num_classes: int,
                   foreground_only: bool = True) -> Dict[str, float]:
    model.eval()
    dice_metric = DiceMetric(include_background=not foreground_only, reduction="mean_batch")
    post_pred = AsDiscrete(argmax=True, to_onehot=True, num_classes=num_classes)
    post_label = AsDiscrete(to_onehot=True, num_classes=num_classes)

    val_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            labels_eval = labels.clone()
            labels_eval[labels_eval < 0] = 0
            labels_eval = labels_eval.long()

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = post_pred(probs)

            target = post_label(labels_eval.unsqueeze(1))
            dice_metric(y_pred=preds, y=target)
            steps += 1

    dice = dice_metric.aggregate().to(device)
    dice_metric.reset()
    dice = float(reduce_mean(dice))

    return {"dice": dice, "steps": steps}
