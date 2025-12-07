from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist


def init_ssf_scale_shift(dim: int):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(scale, mean=1.0, std=0.02)
    nn.init.normal_(shift, mean=0.0, std=0.02)
    return scale, shift


def ssf_ada(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return x * scale + shift


class GlobalFilter3D(nn.Module):
    """Frequency-domain filter adapted from FreqFiT for 3D volumes."""

    def __init__(self, dim: int, h: int = 96, w: int = 96, d: int = 96):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(h, w, d // 2 + 1, dim, 2, dtype=torch.float32) * 0.02
        )
        self.dim = dim
        self.ssf_scale, self.ssf_shift = init_ssf_scale_shift(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W, D)
        _, _, h, w, d = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        residual = x
        x = x.to(torch.float32)
        x_fft = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")

        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight

        x = torch.fft.irfftn(x_fft, s=(h, w, d), dim=(1, 2, 3), norm="ortho")
        x = ssf_ada(x, self.ssf_scale, self.ssf_shift)
        x = x + residual

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


def prepare_class_ratios(prior_data: Dict,
                         expected_num_classes: int,
                         foreground_only: bool,
                         *,
                         is_main: bool = False,
                         context: str = "class prior") -> np.ndarray:
    """Normalise class ratios to match the requested label space."""

    ratios = np.asarray(prior_data.get("class_ratios", []), dtype=np.float64)
    mode = str(prior_data.get("mode", "")).lower()
    foreground_modes = {"foreground_only", "foreground-only", "foreground"}

    inferred_from_volume_stats = False
    if ratios.size == 0:
        # Attempt to interpret as volume_stats.json style mapping
        volume_keys = [k for k, v in prior_data.items() if isinstance(v, dict) and "means" in v]
        if volume_keys:
            accum = np.zeros(expected_num_classes, dtype=np.float64)
            counts = np.zeros(expected_num_classes, dtype=np.float64)
            for key in volume_keys:
                entry = prior_data[key]
                means = np.asarray(entry.get("means", []), dtype=np.float64)
                n = np.asarray(entry.get("n", []), dtype=np.float64)
                if means.size == 0:
                    continue
                if means.size != expected_num_classes and is_main:
                    print(
                        f"  ⚠️  {context}: entry {key} has {means.size} means; expected {expected_num_classes}"
                    )
                means = np.pad(means, (0, max(0, expected_num_classes - means.size)))[:expected_num_classes]
                if n.size == 0:
                    n = np.ones_like(means)
                n = np.pad(n, (0, max(0, expected_num_classes - n.size)))[:expected_num_classes]
                accum += means * n
                counts += n
            valid = counts > 0
            ratios = np.zeros(expected_num_classes, dtype=np.float64)
            ratios[valid] = accum[valid] / counts[valid]
            inferred_from_volume_stats = True
        else:
            ratios = np.asarray(prior_data.get("ratios", []), dtype=np.float64)

    if ratios.size == 0:
        ratios = np.ones(expected_num_classes, dtype=np.float64)

    if inferred_from_volume_stats:
        mode = "foreground_only"

    if foreground_only:
        if mode not in foreground_modes and ratios.size > 0:
            ratios = ratios[1:]
        elif ratios.size == expected_num_classes + 1:
            ratios = ratios[1:]
    else:
        if mode in foreground_modes and ratios.size == expected_num_classes - 1:
            background_ratio = prior_data.get("background_ratio")
            if background_ratio is None:
                background_ratio = max(0.0, 1.0 - ratios.sum())
            ratios = np.concatenate(([background_ratio], ratios))

    if ratios.size != expected_num_classes:
        if is_main:
            print(f"  ⚠️  {context}: adjusting ratios from {ratios.size} to {expected_num_classes}")
        if ratios.size > expected_num_classes:
            ratios = ratios[:expected_num_classes]
        else:
            ratios = np.pad(ratios, (0, expected_num_classes - ratios.size), constant_values=0.0)

    if ratios.sum() <= 0:
        if is_main:
            print(f"  ⚠️  {context}: sum of ratios is non-positive; using uniform prior")
        ratios = np.ones(expected_num_classes, dtype=np.float64)

    return ratios


def _load_class_weights(stats_path: Optional[str],
                        num_classes: int,
                        foreground_only: bool,
                        enhanced: bool,
                        *,
                        device: Optional[torch.device] = None,
                        is_main: bool = False) -> Optional[torch.Tensor]:
    if stats_path is None or not os.path.exists(stats_path):
        if is_main:
            print("  ⚠️  No volume_stats.json provided; using uniform class weights")
        return None

    with open(stats_path, "r") as f:
        prior_data = json.load(f)

    ratios = prepare_class_ratios(
        prior_data,
        expected_num_classes=num_classes,
        foreground_only=foreground_only,
        is_main=is_main,
        context="Volume prior",
    )

    eps = 1e-7
    weights = 1.0 / (ratios + eps)
    weights = weights / weights.mean()

    if enhanced:
        weights = np.log1p(weights)
        small_mask = ratios < 1e-3
        weights[small_mask] *= 2.0
        weights = np.clip(weights, 0.1, 20.0)
        weights = weights / weights.mean()
    else:
        weights = np.sqrt(weights)
        weights = np.clip(weights, 0.1, 10.0)
        weights = weights / weights.mean()

    tensor = torch.as_tensor(weights, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


class SimplifiedDAUnetModule(nn.Module):
    """Thin wrapper around a backbone network with optional class weights."""

    def __init__(self,
                 backbone: nn.Module,
                 num_classes: int,
                 *,
                 volume_stats_path: Optional[str] = None,
                 foreground_only: bool = True,
                 enhanced_class_weights: bool = True,
                 use_age_conditioning: bool = False,
                 debug_mode: bool = False,
                 use_freqfit: bool = False,
                 roi_size: tuple = (96, 96, 96),
                 in_channels: int = 1):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.foreground_only = foreground_only
        self.use_age_conditioning = use_age_conditioning
        self.debug_mode = debug_mode
        self.use_freqfit = use_freqfit
        self._age_strength = torch.tensor(1.0)

        self.freq_filter: Optional[GlobalFilter3D] = None

        is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        device = next(backbone.parameters()).device

        if self.use_freqfit:
            if is_main:
                print("🌊 FreqFiT frequency adapter enabled")
            self.freq_filter = GlobalFilter3D(
                dim=in_channels,
                h=roi_size[0],
                w=roi_size[1],
                d=roi_size[2],
            ).to(device)
        self.class_weights = _load_class_weights(
            volume_stats_path,
            num_classes=num_classes,
            foreground_only=foreground_only,
            enhanced=enhanced_class_weights,
            device=device,
            is_main=is_main,
        )

        if is_main:
            print("✅ SimplifiedDAUnetModule initialised")
            print(f"  Classes: {num_classes}")
            print(f"  Foreground-only remap: {foreground_only}")
            if self.class_weights is not None:
                print("  Loaded class weights from volume prior")
            else:
                print("  Using uniform class weights")

    def forward(self, x: torch.Tensor, age: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Age conditioning can be added here if desired. The placeholder keeps
        # the interface compatible with previous experiments.
        if self.use_freqfit and self.freq_filter is not None:
            x = self.freq_filter(x)
        return self.backbone(x)

    def get_class_weights(self) -> Optional[torch.Tensor]:
        return self.class_weights

    def set_age_conditioning_strength(self, strength: float) -> None:
        self._age_strength = torch.as_tensor(strength)
