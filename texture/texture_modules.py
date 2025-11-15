"""Texture feature extractors and auxiliary heads for texture-centric adaptation."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def compute_texture_statistics(volume: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute differentiable-friendly texture statistics for monitoring.

    Args:
        volume: input tensor with shape ``(B, C, H, W, D)`` in ``float32``.
    Returns:
        A mapping from metric name to ``(B,)`` tensors.
    """

    if volume.dim() != 5:
        raise ValueError(f"Expected tensor of shape (B, C, H, W, D), got {tuple(volume.shape)}")

    stats: Dict[str, torch.Tensor] = {}
    batch_size = volume.shape[0]
    flattened = volume.view(batch_size, -1)

    stats["intensity_mean"] = flattened.mean(dim=-1)
    stats["intensity_std"] = flattened.std(dim=-1)

    q_values = torch.quantile(flattened, torch.tensor([0.1, 0.5, 0.9], device=volume.device), dim=-1)
    stats["q10"], stats["q50"], stats["q90"] = q_values[0], q_values[1], q_values[2]
    stats["dynamic_range"] = stats["q90"] - stats["q10"]

    # Local contrast through moving window variance
    window = 5
    padding = window // 2
    mean_local = F.avg_pool3d(volume, kernel_size=window, stride=1, padding=padding)
    sq_mean_local = F.avg_pool3d(volume ** 2, kernel_size=window, stride=1, padding=padding)
    local_variance = (sq_mean_local - mean_local ** 2).clamp_min(0.0)
    local_std = torch.sqrt(local_variance + 1e-6)
    stats["local_contrast"] = local_std.view(batch_size, -1).mean(dim=-1)

    # Laplacian energy for blur quantification
    laplace_kernel = torch.tensor(
        [[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
         [[-1, 8, -1], [8, -24, 8], [-1, 8, -1]],
         [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]],
        dtype=volume.dtype,
        device=volume.device,
    ).view(1, 1, 3, 3, 3)
    laplace = F.conv3d(volume, laplace_kernel, padding=1)
    stats["laplacian_var"] = laplace.view(batch_size, -1).var(dim=-1)

    # Gradient magnitude energy (Sobels)
    sobel_x = torch.tensor(
        [[[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]],
         [[-3, 0, 3], [-6, 0, 6], [-3, 0, 3]],
         [[-1, 0, 1], [-3, 0, 3], [-1, 0, 1]]],
        dtype=volume.dtype,
        device=volume.device,
    ).view(1, 1, 3, 3, 3) / 32.0
    sobel_y = sobel_x.transpose(2, 3)
    sobel_z = sobel_x.transpose(2, 4)

    gx = F.conv3d(volume, sobel_x, padding=1)
    gy = F.conv3d(volume, sobel_y, padding=1)
    gz = F.conv3d(volume, sobel_z, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + 1e-6)
    stats["gradient_energy"] = grad_mag.view(batch_size, -1).mean(dim=-1)

    return stats


class StyleEncoder(nn.Module):
    """Light-weight CNN that extracts style embeddings from input intensities."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels, affine=True),
            nn.GELU(),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(base_channels * 2, affine=True),
            nn.GELU(),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(base_channels * 4, affine=True),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, embedding_dim),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="gelu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.head(features)


class DomainDiscriminator(nn.Module):
    """Feed-forward classifier predicting domain ids from style embeddings."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 128, num_domains: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_domains),
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class ProjectionHead(nn.Module):
    """Projection head for contrastive-style objectives on style embeddings."""

    def __init__(self, embedding_dim: int, projection_dim: int = 64) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, projection_dim),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model(embedding), dim=-1)


def gaussian_mmd(source: torch.Tensor, target: torch.Tensor, kernel_mul: float = 2.0, kernel_count: int = 5) -> torch.Tensor:
    """Compute an RBF-kernel Maximum Mean Discrepancy between two sets."""
    if source.size(0) == 0 or target.size(0) == 0:
        return source.new_tensor(0.0)

    total = torch.cat([source, target], dim=0)
    pairwise_dists = torch.cdist(total, total, p=2) ** 2

    bandwidth = pairwise_dists.detach().mean()
    bandwidth = bandwidth / (kernel_mul ** (kernel_count // 2))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_count)]

    kernels = sum(torch.exp(-pairwise_dists / bw.clamp_min(1e-6)) for bw in bandwidth_list)

    n_source = source.size(0)
    n_target = target.size(0)
    xx = kernels[:n_source, :n_source]
    yy = kernels[n_source:, n_source:]
    xy = kernels[:n_source, n_source:]

    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd


class TextureFeatureMonitor:
    """Accumulate per-domain texture statistics for logging."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sums: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._counts: Dict[int, int] = defaultdict(int)

    def update(self, stats: Dict[str, torch.Tensor], domains: torch.Tensor) -> None:
        domain_ids = domains.view(-1).tolist()
        for idx, domain_id in enumerate(domain_ids):
            self._counts[domain_id] += 1
            for name, values in stats.items():
                self._sums[domain_id][name] += float(values[idx].detach().cpu())

    def compute(self) -> Dict[int, Dict[str, float]]:
        report: Dict[int, Dict[str, float]] = {}
        for domain_id, counts in self._counts.items():
            if counts == 0:
                continue
            report[domain_id] = {
                name: total / counts for name, total in self._sums[domain_id].items()
            }
        return report


__all__ = [
    "compute_texture_statistics",
    "StyleEncoder",
    "DomainDiscriminator",
    "ProjectionHead",
    "gaussian_mmd",
    "TextureFeatureMonitor",
]
