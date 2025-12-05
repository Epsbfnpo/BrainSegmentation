import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss


class TransferRiskManager:
    """Compute and cache pixel-wise Transfer Risk Maps using LEEP statistics."""

    def __init__(self, num_classes: int, device: torch.device, momentum: float = 0.9) -> None:
        self.num_classes = num_classes
        self.device = device
        self.momentum = momentum

        self.joint_counts = torch.ones((num_classes, num_classes), device=device, dtype=torch.float32) * 1e-6
        self.p_y_given_z = None
        self.frozen = False

    def update_statistics(self, source_logits: torch.Tensor, target_labels: torch.Tensor) -> None:
        if self.frozen:
            return

        with torch.no_grad():
            B, C = source_logits.shape[:2]
            z_probs = torch.softmax(source_logits, dim=1).transpose(0, 1).reshape(C, -1)

            y_flat = target_labels.view(-1).long()
            valid_mask = (y_flat >= 0) & (y_flat < self.num_classes)

            z_probs = z_probs[:, valid_mask]
            y_flat = y_flat[valid_mask]
            if y_flat.numel() == 0:
                return

            y_onehot = F.one_hot(y_flat, num_classes=self.num_classes).float().t()
            batch_counts = torch.matmul(y_onehot, z_probs.t())

            self.joint_counts = self.momentum * self.joint_counts + (1 - self.momentum) * batch_counts
            p_z = self.joint_counts.sum(dim=0, keepdim=True).clamp(min=1e-9)
            self.p_y_given_z = self.joint_counts / p_z

    def freeze_statistics(self) -> None:
        self.frozen = True
        print("[TRM] Transferability statistics frozen.")

    def compute_risk_map(self, source_logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        if self.p_y_given_z is None:
            return torch.ones_like(target_labels, dtype=torch.float32)

        with torch.no_grad():
            z_probs = torch.softmax(source_logits, dim=1)
            B, C, D, H, W = z_probs.shape
            flat_z = z_probs.view(B, C, -1).permute(0, 2, 1)

            expected_y_probs = torch.matmul(flat_z, self.p_y_given_z.t())
            expected_y_probs = expected_y_probs.permute(0, 2, 1).view(B, C, D, H, W)

            labels = target_labels.clone()
            labels[labels < 0] = 0
            if labels.ndim == 4:
                labels = labels.unsqueeze(1)

            leep_probs = torch.gather(expected_y_probs, 1, labels.long())
            leep_score = torch.log(leep_probs.clamp(min=1e-6))
            hardness = -leep_score

            h_flat = hardness.view(B, -1)
            h_min = h_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1, 1)
            h_max = h_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1, 1)
            normalized = (hardness - h_min) / (h_max - h_min + 1e-6)
            weights = torch.pow(10.0, normalized)
            return weights


class TRMWeightedLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        self.dice = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            squared_pred=True,
        )
        self.num_classes = num_classes

    def forward(self, target_logits: torch.Tensor, target_labels: torch.Tensor, risk_map: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(target_logits, target_labels.squeeze(1).long())
        if risk_map.shape[1] == 1:
            risk_map = risk_map.squeeze(1)

        weighted_ce = ce_loss * risk_map
        foreground_mask = (target_labels.squeeze(1) >= 0).float()
        denom = foreground_mask.sum().clamp(min=1.0)
        loss_ce = weighted_ce.sum() / denom

        labels_for_dice = target_labels.clone()
        labels_for_dice[labels_for_dice < 0] = 0
        if labels_for_dice.ndim == 4:
            labels_for_dice = labels_for_dice.unsqueeze(1)

        loss_dice = self.dice(target_logits, labels_for_dice)
        return loss_ce + loss_dice
