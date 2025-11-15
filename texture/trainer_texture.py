"""Training utilities for the texture-centric segmentation workflow."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Activations
from monai.networks.utils import one_hot

from texture_modules import (
    DomainDiscriminator,
    StyleEncoder,
    TextureFeatureMonitor,
    compute_texture_statistics,
    gaussian_mmd,
)


@dataclass
class TextureTrainingState:
    epoch: int
    best_target_dice: float = -math.inf


class TextureTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        style_encoder: StyleEncoder,
        domain_discriminator: DomainDiscriminator,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        *,
        num_classes: int,
        lambda_domain: float,
        lambda_mmd: float,
        use_amp: bool = False,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        self.model = model.to(device)
        self.style_encoder = style_encoder.to(device)
        self.domain_discriminator = domain_discriminator.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = int(num_classes)
        self.lambda_domain = float(lambda_domain)
        self.lambda_mmd = float(lambda_mmd)
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm

        self.seg_loss = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
        self.domain_monitor = TextureFeatureMonitor()
        self.post_pred = Activations(softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def train_epoch(self, loader: Iterable[Dict[str, torch.Tensor]], epoch: int) -> Dict[str, float]:
        self.model.train()
        self.style_encoder.train()
        self.domain_discriminator.train()
        self.domain_monitor.reset()

        running: Dict[str, float] = {
            "loss_total": 0.0,
            "loss_seg": 0.0,
            "loss_domain": 0.0,
            "loss_mmd": 0.0,
            "domain_acc": 0.0,
        }
        sample_count = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).long()
            domain_ids = batch["domain"].to(self.device).view(-1).long()

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                seg_loss = self.seg_loss(logits, labels)

                style_embeddings = self.style_encoder(images)
                domain_logits = self.domain_discriminator(style_embeddings)
                domain_loss = F.cross_entropy(domain_logits, domain_ids)

                mmd_loss = images.new_tensor(0.0)
                if self.lambda_mmd > 0:
                    src_mask = domain_ids == 0
                    tgt_mask = domain_ids == 1
                    if src_mask.any() and tgt_mask.any():
                        mmd_loss = gaussian_mmd(style_embeddings[src_mask], style_embeddings[tgt_mask])

                total_loss = seg_loss + self.lambda_domain * domain_loss + self.lambda_mmd * mmd_loss

            self.scaler.scale(total_loss).backward()
            if self.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters())
                    + list(self.style_encoder.parameters())
                    + list(self.domain_discriminator.parameters()),
                    max_norm=self.max_grad_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                probs = self.post_pred(logits)
                labels_one_hot = one_hot(labels, num_classes=self.num_classes)
                self.dice_metric(probs, labels_one_hot)
                stats = compute_texture_statistics(images)
                self.domain_monitor.update(stats, domain_ids)
                domain_predictions = domain_logits.argmax(dim=-1)
                batch_accuracy = (domain_predictions == domain_ids).float().mean()

            batch_size = images.size(0)
            sample_count += batch_size
            running["loss_total"] += float(total_loss.detach().cpu()) * batch_size
            running["loss_seg"] += float(seg_loss.detach().cpu()) * batch_size
            running["loss_domain"] += float(domain_loss.detach().cpu()) * batch_size
            running["loss_mmd"] += float(mmd_loss.detach().cpu()) * batch_size
            running["domain_acc"] += float(batch_accuracy.cpu()) * batch_size

        epoch_time = time.time() - start_time
        for key in running:
            running[key] /= max(sample_count, 1)
        running["dice_train"] = float(self.dice_metric.aggregate().cpu())
        self.dice_metric.reset()
        running["epoch_time"] = epoch_time
        running["texture_stats"] = self.domain_monitor.compute()

        return running

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        self.model.eval()
        self.style_encoder.eval()
        self.domain_discriminator.eval()
        self.dice_metric.reset()

        running_loss = 0.0
        sample_count = 0
        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).long()

            logits = self.model(images)
            loss = self.seg_loss(logits, labels)
            probs = self.post_pred(logits)
            labels_one_hot = one_hot(labels, num_classes=self.num_classes)
            self.dice_metric(probs, labels_one_hot)

            batch_size = images.size(0)
            sample_count += batch_size
            running_loss += float(loss.detach().cpu()) * batch_size

        metrics = {
            "loss": running_loss / max(sample_count, 1),
            "dice": float(self.dice_metric.aggregate().cpu()),
        }
        self.dice_metric.reset()
        return metrics

    def step_scheduler(self, metric: Optional[float] = None) -> None:
        if self.scheduler is None:
            return
        if hasattr(self.scheduler, "step"):
            try:
                if metric is not None:
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
            except TypeError:
                self.scheduler.step()


__all__ = ["TextureTrainer", "TextureTrainingState"]
