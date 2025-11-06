"""Texture-aware segmentation model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

__all__ = ["TextureAwareModel", "TextureBranchConfig"]


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversalFunction.apply(x, lambda_)


@dataclass
class TextureBranchConfig:
    embed_dim: int = 128
    domain_hidden: int = 128
    stats_projection_dim: int = 128
    grl_lambda: float = 1.0


class TextureEncoder(nn.Module):
    """Lightweight CNN to derive texture embeddings from raw images."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        return self.fc(pooled)


class TextureAwareModel(nn.Module):
    """Wrap SwinUNETR with texture encoding and domain discrimination."""

    def __init__(
        self,
        *,
        img_size: Tuple[int, int, int],
        in_channels: int,
        out_channels: int,
        feature_size: int,
        texture_stats_dim: int,
        branch_cfg: TextureBranchConfig | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.segmenter = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True,
            drop_rate=dropout,
        )

        self.branch_cfg = branch_cfg or TextureBranchConfig()
        self.texture_encoder = TextureEncoder(in_channels, self.branch_cfg.embed_dim)
        self.stats_projector = nn.Linear(texture_stats_dim, self.branch_cfg.stats_projection_dim)

        combined_dim = self.branch_cfg.embed_dim + self.branch_cfg.stats_projection_dim
        self.texture_projection = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, combined_dim),
            nn.GELU(),
            nn.Linear(combined_dim, self.branch_cfg.embed_dim),
        )

        self.domain_classifier = nn.Sequential(
            nn.LayerNorm(self.branch_cfg.embed_dim),
            nn.Linear(self.branch_cfg.embed_dim, self.branch_cfg.domain_hidden),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.branch_cfg.domain_hidden, 2),
        )

    def forward(
        self,
        images: torch.Tensor,
        texture_stats: Optional[torch.Tensor] = None,
        *,
        grl_lambda: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.segmenter(images)

        stats = texture_stats if texture_stats is not None else None
        if stats is not None:
            if stats.dim() == 1:
                stats = stats.unsqueeze(0)
            projected_stats = self.stats_projector(stats)
        else:
            projected_stats = torch.zeros(images.size(0), self.branch_cfg.stats_projection_dim, device=images.device)

        encoded = self.texture_encoder(images)
        combined = torch.cat([encoded, projected_stats], dim=1)
        texture_embedding = self.texture_projection(combined)
        texture_embedding = F.normalize(texture_embedding, dim=1)

        lambda_ = grl_lambda if grl_lambda is not None else self.branch_cfg.grl_lambda
        reversed_features = gradient_reversal(texture_embedding, lambda_)
        domain_logits = self.domain_classifier(reversed_features)

        return logits, texture_embedding, domain_logits

    def freeze_segmenter(self) -> None:
        for param in self.segmenter.parameters():
            param.requires_grad = False

    def unfreeze_segmenter(self) -> None:
        for param in self.segmenter.parameters():
            param.requires_grad = True
