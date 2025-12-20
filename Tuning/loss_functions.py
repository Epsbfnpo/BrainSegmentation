"""
Loss functions for brain segmentation
ENHANCED: Better handling of rare classes with ignore_index=-1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import json
import os
import matplotlib.pyplot as plt
import numpy as np


class TverskyLoss(nn.Module):
    """Tversky loss for handling class imbalance

    Better for imbalanced segmentation - can balance FP and FN
    """

    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 smooth: float = 1e-5,
                 ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta   # FN weight
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W, D) logits
            target: (B, H, W, D) integer labels
        """
        # Handle different input shapes for target
        if target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1)

        if target.dim() != 4:
            raise ValueError(f"Expected target to be 4D (B, H, W, D), got {target.dim()}D")

        # Create mask for valid pixels
        valid_mask = (target != self.ignore_index).float()

        # Convert logits to probabilities
        pred = torch.softmax(pred, dim=1)

        # One-hot encode target
        num_classes = pred.shape[1]
        target_one_hot = torch.zeros_like(pred)
        for c in range(num_classes):
            target_one_hot[:, c] = (target == c).float() * valid_mask

        # Calculate Tversky index
        dims = (2, 3, 4)  # Spatial dimensions

        pred_masked = pred * valid_mask.unsqueeze(1)

        true_pos = torch.sum(pred_masked * target_one_hot, dim=dims)
        false_pos = torch.sum(pred_masked * (1 - target_one_hot), dim=dims)
        false_neg = torch.sum((1 - pred_masked) * target_one_hot, dim=dims)

        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)

        # Loss is 1 - Tversky index
        tversky_loss = 1.0 - tversky_index

        return tversky_loss.mean()


class TverskyFocalLoss(nn.Module):
    """Combined Tversky and Focal loss for extreme class imbalance"""

    def __init__(self,
                 tversky_weight: float = 0.5,
                 focal_weight: float = 0.5,
                 tversky_alpha: float = 0.5,
                 tversky_beta: float = 0.5,
                 focal_gamma: float = 2.5,
                 smooth: float = 1e-5,
                 class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = -1):
        super().__init__()
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        self.ignore_index = ignore_index

        self.tversky_loss = TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            smooth=smooth,
            ignore_index=ignore_index
        )

        self.focal_loss = FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            reduction='mean',
            ignore_index=ignore_index
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tversky = self.tversky_loss(pred, target)
        focal = self.focal_loss(pred, target)

        total_loss = self.tversky_weight * tversky + self.focal_weight * focal

        return total_loss


class DiceLoss(nn.Module):
    """Dice loss for segmentation with ignore_index support"""

    def __init__(self,
                 include_background: bool = True,
                 smooth: float = 1e-5,
                 squared_pred: bool = False,
                 reduction: str = 'mean',
                 ignore_index: int = -1,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth
        self.squared_pred = squared_pred
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W, D) logits or probabilities
            target: (B, H, W, D) or (B, 1, H, W, D) integer labels
        """
        # Handle different input shapes for target
        if target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1)

        if target.dim() != 4:
            raise ValueError(f"Expected target to be 4D (B, H, W, D), got {target.dim()}D with shape {target.shape}")

        # Create mask for valid pixels (not ignore_index)
        valid_mask = (target != self.ignore_index).float()

        # Convert logits to probabilities
        if not self.squared_pred:
            pred = torch.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)

        # One-hot encode target (excluding ignore_index pixels)
        num_classes = pred.shape[1]

        # Create one-hot encoding only for valid pixels
        target_one_hot = torch.zeros_like(pred)
        for c in range(num_classes):
            target_one_hot[:, c] = (target == c).float() * valid_mask

        # Calculate dice score per class
        dims = (2, 3, 4)  # Spatial dimensions

        # Apply valid mask to predictions as well
        pred_masked = pred * valid_mask.unsqueeze(1)

        intersection = torch.sum(pred_masked * target_one_hot, dim=dims)
        cardinality = torch.sum(pred_masked + target_one_hot, dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Apply class weights if provided
        if self.class_weights is not None:
            dice_score = dice_score * self.class_weights.unsqueeze(0)

        # Calculate loss
        dice_loss = 1.0 - dice_score

        # Reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance with ignore_index support"""

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.5,
                 reduction: str = 'mean',
                 ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W, D) logits
            target: (B, H, W, D) or (B, 1, H, W, D) integer labels
        """
        # Handle different input shapes for target
        if target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1)

        # Reshape for cross entropy
        B, C, H, W, D = pred.shape
        pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
        target_flat = target.reshape(-1).long()

        # Calculate cross entropy with ignore_index
        log_pt = F.log_softmax(pred_flat, dim=1)
        ce_loss = F.nll_loss(log_pt, target_flat, reduction='none', ignore_index=self.ignore_index)

        # Get probability of correct class
        pt = torch.exp(-ce_loss)

        # Calculate focal term
        focal_term = (1 - pt) ** self.gamma

        # Apply focal term
        focal_loss = focal_term * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            # Create weight tensor for each sample
            alpha_t = torch.ones_like(target_flat, dtype=torch.float32)
            for c in range(len(self.alpha)):
                alpha_t[target_flat == c] = self.alpha[c]

            # Don't apply weight to ignored pixels
            alpha_t[target_flat == self.ignore_index] = 0

            focal_loss = alpha_t * focal_loss

        # Reduction
        if self.reduction == 'mean':
            # Create valid mask
            valid_mask = (target_flat != self.ignore_index).float()
            # Calculate denominator
            denom = valid_mask.sum()
            # Apply mask and sum
            masked_loss = focal_loss * valid_mask
            # Return mean over valid pixels
            return masked_loss.sum() / denom.clamp_min(1.0) if denom > 0 else focal_loss.mean()
        elif self.reduction == 'sum':
            # Create valid mask
            valid_mask = (target_flat != self.ignore_index).float()
            # Return sum of valid pixels only
            return (focal_loss * valid_mask).sum()
        else:
            return focal_loss.reshape(B, H, W, D)


class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy loss with ignore_index support"""

    def __init__(self,
                 dice_weight: float = 0.5,
                 ce_weight: float = 0.5,
                 include_background: bool = True,
                 smooth: float = 1e-5,
                 class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = -1):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index

        self.dice_loss = DiceLoss(
            include_background=include_background,
            smooth=smooth,
            reduction='mean',
            ignore_index=ignore_index,
            class_weights=class_weights
        )

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction='mean',
            ignore_index=ignore_index
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W, D) logits
            target: (B, H, W, D) or (B, 1, H, W, D) integer labels
        """
        # Handle different input shapes for target
        if target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1)

        # Dice loss
        dice = self.dice_loss(pred, target)

        # Cross entropy loss
        # Reshape for CE loss
        B, C, H, W, D = pred.shape
        pred_ce = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
        target_ce = target.reshape(-1).long()
        ce = self.ce_loss(pred_ce, target_ce)

        # Combined loss
        total_loss = self.dice_weight * dice + self.ce_weight * ce

        return total_loss


class DiceFocalLoss(nn.Module):
    """Combined Dice and Focal loss with ignore_index support"""

    def __init__(self,
                 dice_weight: float = 0.5,
                 focal_weight: float = 0.5,
                 include_background: bool = True,
                 smooth: float = 1e-5,
                 gamma: float = 2.5,
                 alpha: Optional[torch.Tensor] = None,
                 ignore_index: int = -1):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ignore_index = ignore_index

        self.dice_loss = DiceLoss(
            include_background=include_background,
            smooth=smooth,
            reduction='mean',
            ignore_index=ignore_index,
            class_weights=alpha
        )

        self.focal_loss = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction='mean',
            ignore_index=ignore_index
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)

        total_loss = self.dice_weight * dice + self.focal_weight * focal

        return total_loss


def get_class_weights(args, device: torch.device) -> Optional[torch.Tensor]:
    """Load or compute class weights for imbalanced data

    ENHANCED: More aggressive weighting for rare classes
    """

    if args.class_weights is None:
        return None

    if args.class_weights == 'auto' and args.class_prior_json:
        # Load from class prior file
        print("📊 Computing class weights from prior distribution...")

        with open(args.class_prior_json, 'r') as f:
            prior_data = json.load(f)

        class_ratios = torch.tensor(prior_data['class_ratios'], dtype=torch.float32)

        if args.foreground_only and args.out_channels == 87:
            # For foreground-only mode, we have 87 output channels
            weights = torch.zeros(args.out_channels)

            # Calculate weights for each brain region
            for i in range(args.out_channels):
                ratio_idx = i + 1  # Skip background ratio

                if ratio_idx < len(class_ratios):
                    # Power scaling for more balanced weights
                    weight = torch.pow(1.0 / (class_ratios[ratio_idx] + 1e-6), args.weight_power)
                    weights[i] = weight
                else:
                    weights[i] = 1.0

            # Normalize weights
            weights = weights / weights.mean()

            # Clip weights to reasonable range
            weights = torch.clamp(weights, min=args.min_weight, max=args.max_weight)

            print(f"\n  Foreground-only class weight statistics:")
            print(f"    Number of classes: {len(weights)}")
            print(f"    Min weight: {weights.min():.3f}")
            print(f"    Max weight: {weights.max():.3f}")
            print(f"    Mean weight: {weights.mean():.3f}")
            print(f"    Std weight: {weights.std():.3f}")

            # Find most and least weighted classes
            sorted_indices = torch.argsort(weights, descending=True)
            print(f"\n  Top 5 highest weighted classes (rarest):")
            for i in range(min(5, len(weights))):
                idx = sorted_indices[i]
                brain_region = idx + 1
                if brain_region < len(class_ratios):
                    print(f"    Brain region {brain_region}: weight={weights[idx]:.3f}, ratio={class_ratios[brain_region]:.6f}")

            print(f"\n  Top 5 lowest weighted classes (most common):")
            for i in range(min(5, len(weights))):
                idx = sorted_indices[-(i+1)]
                brain_region = idx + 1
                if brain_region < len(class_ratios):
                    print(f"    Brain region {brain_region}: weight={weights[idx]:.3f}, ratio={class_ratios[brain_region]:.6f}")

            # Count extreme weights
            high_weight_count = (weights > 10.0).sum()
            if high_weight_count > 0:
                print(f"\n  Classes with weight > 10: {high_weight_count}")
                high_indices = torch.where(weights > 10.0)[0]
                for idx in high_indices[:5]:  # Show first 5
                    brain_region = idx + 1
                    print(f"    Region {brain_region}: weight={weights[idx]:.3f}")

        # Visualize weights if requested
        if hasattr(args, 'visualize_weights') and args.visualize_weights:
            plt.figure(figsize=(15, 8))

            # Main plot
            plt.subplot(2, 1, 1)
            plt.bar(range(len(weights)), weights.cpu().numpy())
            plt.xlabel('Class Index (Brain Region - 1)')
            plt.ylabel('Weight')
            plt.title(f'Class Weights Distribution (Power={args.weight_power:.2f})')
            plt.grid(True, alpha=0.3)

            # Add statistics
            plt.axhline(y=weights.mean().item(), color='r', linestyle='--', label=f'Mean: {weights.mean():.2f}')
            plt.axhline(y=1.0, color='g', linestyle='--', label='Baseline: 1.0')
            plt.axhline(y=10.0, color='orange', linestyle='--', alpha=0.5, label='High weight threshold')
            plt.legend()

            # Histogram
            plt.subplot(2, 1, 2)
            plt.hist(weights.cpu().numpy(), bins=30, edgecolor='black')
            plt.xlabel('Weight Value')
            plt.ylabel('Number of Classes')
            plt.title('Distribution of Class Weights')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = os.path.join(args.results_dir, 'class_weights.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n  Class weights visualization saved to: {save_path}")

        return weights.to(device)

    elif os.path.exists(args.class_weights):
        # Load from file
        print(f"📊 Loading class weights from: {args.class_weights}")
        weights = torch.load(args.class_weights)
        return weights.to(device)

    else:
        print("⚠️ No valid class weights found, using uniform weights")
        return None


def get_loss_function(args, device: torch.device) -> nn.Module:
    """Get loss function based on configuration

    ENHANCED: Better handling of ignore_index=-1
    """

    # Get class weights if needed
    class_weights = get_class_weights(args, device)

    print(f"\n📉 Creating loss function: {args.loss_type}")

    if args.out_channels == 15:
        ignore_index = -100
        print("  ℹ️  AMOS Mode (15 classes): Including background (class 0) in loss.")
    elif args.foreground_only and args.out_channels == 87:
        ignore_index = -1
    else:
        ignore_index = -100

    print(f"  Using ignore_index={ignore_index} for background pixels")

    include_background = True  # Always include for consistency

    if args.loss_type == 'dice':
        loss_fn = DiceLoss(
            include_background=include_background,
            smooth=args.dice_smooth,
            reduction='mean',
            ignore_index=ignore_index,
            class_weights=class_weights
        )

    elif args.loss_type == 'ce':
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction='mean',
            ignore_index=ignore_index
        )

    elif args.loss_type == 'dice_ce':
        loss_fn = DiceCELoss(
            dice_weight=args.dice_weight,
            ce_weight=args.ce_weight,
            include_background=include_background,
            smooth=args.dice_smooth,
            class_weights=class_weights,
            ignore_index=ignore_index
        )
        print(f"  Dice weight: {args.dice_weight}")
        print(f"  CE weight: {args.ce_weight}")

    elif args.loss_type == 'focal':
        loss_fn = FocalLoss(
            alpha=class_weights,
            gamma=args.focal_gamma,
            reduction='mean',
            ignore_index=ignore_index
        )
        print(f"  Focal gamma: {args.focal_gamma}")

    elif args.loss_type == 'dice_focal':
        loss_fn = DiceFocalLoss(
            dice_weight=args.dice_weight,
            focal_weight=args.ce_weight,
            include_background=include_background,
            smooth=args.dice_smooth,
            gamma=args.focal_gamma,
            alpha=class_weights,
            ignore_index=ignore_index
        )
        print(f"  Dice weight: {args.dice_weight}")
        print(f"  Focal weight: {args.ce_weight}")
        print(f"  Focal gamma: {args.focal_gamma}")

    elif args.loss_type == 'tversky_focal':
        loss_fn = TverskyFocalLoss(
            tversky_weight=args.dice_weight,
            focal_weight=args.ce_weight,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            focal_gamma=args.focal_gamma,
            smooth=args.dice_smooth,
            class_weights=class_weights,
            ignore_index=ignore_index
        )
        print(f"  Tversky weight: {args.dice_weight}")
        print(f"  Focal weight: {args.ce_weight}")
        print(f"  Tversky alpha (FP): {args.tversky_alpha}")
        print(f"  Tversky beta (FN): {args.tversky_beta}")
        print(f"  Focal gamma: {args.focal_gamma}")

    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    print(f"  Include background in Dice: {include_background}")

    return loss_fn


class LossTracker:
    """Track and smooth loss values during training"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.losses = []
        self.smooth_loss = 0.0

    def update(self, loss: float):
        """Update with new loss value"""
        self.losses.append(loss)
        if len(self.losses) > self.window_size:
            self.losses.pop(0)

        # Calculate smoothed loss
        if self.losses:
            self.smooth_loss = sum(self.losses) / len(self.losses)

    def get_smooth_loss(self) -> float:
        """Get smoothed loss value"""
        return self.smooth_loss

    def reset(self):
        """Reset tracker"""
        self.losses = []
        self.smooth_loss = 0.0
