"""Self-contained 3D U-Mamba implementation for neonatal brain segmentation.

This model is independent from the GraphAlign implementation in ``new/`` and is
intended for fully supervised training on the target dataset.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class InstanceNorm3d(nn.InstanceNorm3d):
    """Alias to keep the normalization layer explicit."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return super().forward(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

        self.downsample: nn.Module | None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.act(out)


class MambaLayer(nn.Module):
    """Applies Mamba in flattened depth-height-width space."""

    def __init__(self, dim: int, *, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) -> treat D*H*W as sequence length
        b, c, d, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, L, C)
        seq = self.norm(seq)
        seq = self.mamba(seq)
        return seq.transpose(1, 2).reshape(b, c, d, h, w)


class UMamba3D(nn.Module):
    """U-Net style encoder-decoder with Mamba bottleneck."""

    def __init__(self, *, in_channels: int = 1, out_channels: int = 87, features: tuple[int, ...] = (32, 64, 128, 256, 512)):
        super().__init__()
        self.stem = nn.Conv3d(in_channels, features[0], kernel_size=3, padding=1)

        # Encoder
        self.enc1 = ResidualBlock(features[0], features[0])
        self.enc2 = ResidualBlock(features[0], features[1], stride=2)
        self.enc3 = ResidualBlock(features[1], features[2], stride=2)
        self.enc4 = ResidualBlock(features[2], features[3], stride=2)
        self.enc5 = ResidualBlock(features[3], features[4], stride=2)

        # Bottleneck
        self.mamba = MambaLayer(features[4])

        # Decoder
        self.up4 = nn.ConvTranspose3d(features[4], features[3], kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(features[3] * 2, features[3])

        self.up3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(features[0] * 2, features[0])

        self.out = nn.Conv3d(features[0], out_channels, kernel_size=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, (nn.InstanceNorm3d, nn.LayerNorm)):
            if module.affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        x5 = self.mamba(x5)

        d4 = self.up4(x5)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


__all__ = ["UMamba3D"]
