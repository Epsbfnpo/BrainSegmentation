import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _get_norm(dim: int, num_features: int) -> nn.Module:
    if dim == 3:
        return nn.InstanceNorm3d(num_features, affine=True)
    if dim == 2:
        return nn.InstanceNorm2d(num_features, affine=True)
    raise ValueError(f"Unsupported dim: {dim}")


def _get_conv(dim: int, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: Optional[int] = None,
              groups: int = 1) -> nn.Module:
    if padding is None:
        padding = kernel_size // 2
    if dim == 3:
        return nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
    if dim == 2:
        return nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
    raise ValueError(f"Unsupported dim: {dim}")


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - deterministic wrapper
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MedNeXtBlock(nn.Module):
    """Lightweight ConvNeXt-style block used by MedNeXt."""

    def __init__(self, channels: int, *, dim: int = 3, expansion: int = 4, drop_path: float = 0.0,
                 use_grn: bool = True):
        super().__init__()
        self.dim = dim
        self.dwconv = _get_conv(dim, channels, channels, kernel_size=7, groups=channels)
        self.norm = nn.GroupNorm(num_groups=32 if channels >= 32 else max(1, channels // 2), num_channels=channels)
        hidden = channels * expansion
        self.pwconv1 = _get_conv(dim, channels, hidden, kernel_size=1, padding=0)
        self.act = nn.GELU()
        self.pwconv2 = _get_conv(dim, hidden, channels, kernel_size=1, padding=0)
        self.drop_path = DropPath(drop_path)
        self.use_grn = use_grn
        if use_grn:
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1, 1) if dim == 3 else torch.zeros(1, channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros_like(self.gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.use_grn:
            gx = x.pow(2).mean(dim=tuple(range(2, x.ndim)), keepdim=True)
            gx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x = self.gamma * x * gx + self.beta + x
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return x + shortcut


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, dim: int = 3):
        super().__init__()
        self.conv = _get_conv(dim, in_ch, out_ch, kernel_size=2, stride=2, padding=0)
        self.norm = _get_norm(dim, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, dim: int = 3):
        super().__init__()
        if dim == 3:
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        elif dim == 2:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unsupported dim: {dim}")
        self.norm = _get_norm(dim, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.norm(x)
        return self.act(x)


class ConvHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, dim: int = 3):
        super().__init__()
        self.conv = _get_conv(dim, in_ch, out_ch, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
