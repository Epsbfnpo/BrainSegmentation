"""
Self-supervised learning components for BrainSegFounder
Implements 3-way SSL: Masked Volume Inpainting, Small-Angle Rotation Regression, SimSiam Self-Distillation
FIXED VERSION for registered data with proper small-angle rotation
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Masked Volume Inpainting
# -------------------------------------------------------------------------
class MaskedVolumeInpainting(nn.Module):
    """Randomly masks 3D patches and asks the network to reconstruct."""

    def __init__(self, mask_ratio: float = 0.15, patch_size: int = 16):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def generate_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W, D]
        Returns:
            mask: [B, 1, H, W, D]  (1 = keep, 0 = mask)
        """
        B, _, H, W, D = x.shape
        h_p, w_p, d_p = H // self.patch_size, W // self.patch_size, D // self.patch_size
        num_patches = h_p * w_p * d_p
        num_masked = int(num_patches * self.mask_ratio)

        mask = torch.ones((B, 1, H, W, D), device=x.device)

        for b in range(B):
            to_mask = torch.randperm(num_patches, device=x.device)[:num_masked]
            for idx in to_mask:
                d_idx = idx // (h_p * w_p)
                hw_idx = idx % (h_p * w_p)
                h_idx, w_idx = hw_idx // w_p, hw_idx % w_p

                h0, w0, d0 = h_idx * self.patch_size, w_idx * self.patch_size, d_idx * self.patch_size
                mask[
                    b,
                    :,
                    h0 : h0 + self.patch_size,
                    w0 : w0 + self.patch_size,
                    d0 : d0 + self.patch_size,
                ] = 0

        return mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.generate_mask(x)
        return x * mask, mask


# -------------------------------------------------------------------------
# Small-Angle Rotation Regression for registered data
# -------------------------------------------------------------------------
class RotationPrediction(nn.Module):
    """Predict small rotation angles applied to the 3D volume."""

    def __init__(self, in_dim: int = 768, use_small_angles: bool = True):
        super().__init__()
        self.use_small_angles = use_small_angles

        # Always use regression for registered data
        self.rotation_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 192),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(192, 3),  # Predict 3 rotation angles (x, y, z)
        )

    def create_rotation_matrix_3d(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Create 3D rotation matrices from Euler angles
        Args:
            angles: [B, 3] tensor of rotation angles in radians
        Returns:
            [B, 3, 3] rotation matrices
        """
        B = angles.shape[0]
        device = angles.device

        # Extract individual angles
        ax = angles[:, 0]  # rotation around x-axis
        ay = angles[:, 1]  # rotation around y-axis
        az = angles[:, 2]  # rotation around z-axis

        # Compute sin and cos
        cx, sx = torch.cos(ax), torch.sin(ax)
        cy, sy = torch.cos(ay), torch.sin(ay)
        cz, sz = torch.cos(az), torch.sin(az)

        # Create rotation matrices (using ZYX convention)
        R = torch.zeros(B, 3, 3, device=device)

        R[:, 0, 0] = cy * cz
        R[:, 0, 1] = -cy * sz
        R[:, 0, 2] = sy

        R[:, 1, 0] = cx * sz + sx * sy * cz
        R[:, 1, 1] = cx * cz - sx * sy * sz
        R[:, 1, 2] = -sx * cy

        R[:, 2, 0] = sx * sz - cx * sy * cz
        R[:, 2, 1] = sx * cz + cx * sy * sz
        R[:, 2, 2] = cx * cy

        return R

    def apply_rotation_3d(self, x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Apply small rotations to 3D volumes using grid_sample
        Args:
            x: [B, C, H, W, D] input volumes
            angles: [B, 3] rotation angles in radians
        Returns:
            Rotated volumes [B, C, H, W, D]
        """
        B, C, H, W, D = x.shape
        device = x.device

        # Create rotation matrices
        R = self.create_rotation_matrix_3d(angles)  # [B, 3, 3]

        # Create identity grid
        grid = F.affine_grid(
            torch.eye(3, 4, device=device).unsqueeze(0).repeat(B, 1, 1),
            [B, C, H, W, D],
            align_corners=False
        )  # [B, H, W, D, 3]

        # Apply rotation to grid
        grid_flat = grid.reshape(B, -1, 3)  # [B, H*W*D, 3]
        grid_rot = torch.bmm(grid_flat, R.transpose(1, 2))  # [B, H*W*D, 3]
        grid_rot = grid_rot.reshape(B, H, W, D, 3)  # [B, H, W, D, 3]

        # Apply grid sampling
        x_rot = F.grid_sample(
            x,
            grid_rot,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        return x_rot

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if isinstance(feats, list):
            feats = feats[-1]
        if feats.ndim == 5:  # [B, C, H, W, D] → [B, C]
            feats = F.adaptive_avg_pool3d(feats, 1).flatten(1)
        return self.rotation_head(feats)


# -------------------------------------------------------------------------
# SimSiam Self-Distillation (no negative samples)
# -------------------------------------------------------------------------
class SimSiamLearning(nn.Module):
    """SimSiam self-distillation without negative samples."""

    def __init__(self, feature_dim: int, projection_dim: int = 128):
        super().__init__()

        # Projection head (3-layer MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
            nn.BatchNorm1d(projection_dim, affine=False)  # No learnable params in final BN
        )

        # Predictor head (2-layer MLP) - only applied to one branch
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.BatchNorm1d(projection_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim // 2, projection_dim)
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f1, f2: Two augmented views - either [B, C] or [B, C, H, W, D].
        Returns:
            SimSiam loss (scalar tensor)
        """
        # Pool if needed
        if f1.ndim == 5:
            f1 = F.adaptive_avg_pool3d(f1, 1).flatten(1)
        if f2.ndim == 5:
            f2 = F.adaptive_avg_pool3d(f2, 1).flatten(1)

        # Get projections
        z1 = self.projection_head(f1)
        z2 = self.projection_head(f2)

        # Get predictions (only from one branch)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Stop gradient on one branch (key to SimSiam)
        # Compute symmetrized loss
        loss = -(F.cosine_similarity(p1, z2.detach(), dim=-1).mean() +
                 F.cosine_similarity(p2, z1.detach(), dim=-1).mean()) * 0.5

        # Convert from [-1, 1] to [0, 2] range for consistency with other losses
        loss = loss + 1.0

        return loss


# -------------------------------------------------------------------------
# Combined 3-Way SSL Loss with SimSiam and Small-Angle Rotation
# -------------------------------------------------------------------------
class SSL3WayLoss(nn.Module):
    """Combines Inpainting, Small-Angle Rotation Regression and SimSiam objectives."""

    def __init__(
        self,
        mask_ratio: float = 0.15,
        mask_patch_size: int = 16,
        rotation_angles: List[int] | None = None,  # Not used with small angles
        temperature: float = 0.1,  # Not used with SimSiam
        projection_dim: int = 128,
        feature_dim: int = 768,
        inpainting_weight: float = 1.5,
        rotation_weight: float = 0.1,
        contrastive_weight: float = 0.8,  # Now SimSiam weight
        use_small_rotation: bool = True,
        max_rotation_angle: float = 0.1,  # Maximum rotation in radians (~5.7 degrees)
    ):
        super().__init__()
        self.masking = MaskedVolumeInpainting(mask_ratio, mask_patch_size)
        self.rotation = RotationPrediction(in_dim=feature_dim, use_small_angles=use_small_rotation)
        self.simsiam = SimSiamLearning(feature_dim, projection_dim)

        self.inpainting_weight = inpainting_weight
        self.rotation_weight = rotation_weight
        self.simsiam_weight = contrastive_weight

        self.use_small_rotation = use_small_rotation
        self.max_rotation_angle = max_rotation_angle

    def _extract_feats(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate encoder features."""
        if hasattr(model, "swinViT"):  # SwinUNETR
            feats = model.swinViT(x)
        elif hasattr(model, "encoder"):
            feats = model.encoder(x)
        else:
            feats = model(x, return_features=True)

        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        return feats

    def forward(self, model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            model: the backbone network (e.g. SwinUNETR)
            x:     input volumes [B, C, H, W, D]
        Returns:
            total_loss, dict(loss_name → scalar tensor)
        """
        device = x.device
        B = x.size(0)
        losses: Dict[str, torch.Tensor] = {}

        # ---- 1) Masked Volume Inpainting ---------------------------------
        x_masked, mask = self.masking(x)
        recon = model(x_masked)
        inpaint_loss = F.mse_loss(recon * (1 - mask), x * (1 - mask))
        losses["inpainting"] = inpaint_loss

        # ---- 2) Small-Angle Rotation Regression --------------------------
        if self.use_small_rotation and self.rotation_weight > 0:
            # Sample small random rotation angles
            angles = (torch.rand(B, 3, device=device) - 0.5) * 2 * self.max_rotation_angle  # [-max, +max]

            # Apply rotations to volumes
            rot_vols = self.rotation.apply_rotation_3d(x, angles)

            # Extract features and predict angles
            rot_feats = self._extract_feats(model, rot_vols)
            rot_pred = self.rotation(rot_feats)

            # Regression loss (SmoothL1 is more robust than MSE)
            rot_loss = F.smooth_l1_loss(rot_pred, angles)
        else:
            rot_loss = torch.tensor(0.0, device=device)
        losses["rotation"] = rot_loss

        # ---- 3) SimSiam Self-Distillation --------------------------------
        # View-1: weak augmentation (from data pipeline)
        view1_feats = self._extract_feats(model, x)

        # View-2: intensity-based augmentations for registered data
        view2 = x.clone()

        # Random contrast adjustment
        gamma = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=device) - 0.5) * 0.3
        view2 = torch.pow(view2.clamp(min=1e-7), gamma)

        # Random intensity shift
        shift = (torch.rand(B, 1, 1, 1, 1, device=device) - 0.5) * 0.1
        view2 = view2 + shift

        # Small Gaussian noise
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(view2) * 0.02
            view2 = view2 + noise

        # Clamp to valid range
        view2 = torch.clamp(view2, 0, 1)

        view2_feats = self._extract_feats(model, view2)

        simsiam_loss = self.simsiam(view1_feats, view2_feats)
        losses["contrastive"] = simsiam_loss  # Keep the name for compatibility

        # ---- Combine ------------------------------------------------------
        total = (
            self.inpainting_weight * inpaint_loss
            + self.rotation_weight * rot_loss
            + self.simsiam_weight * simsiam_loss
        )
        losses["total"] = total
        return total, losses