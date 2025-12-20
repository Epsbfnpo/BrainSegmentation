import json
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from typing import Optional, Dict


# =========================================================================
# Part 1: SALT Layers (Implementation of SVD + LoRA adaptation)
# =========================================================================


class SALTLayerBase(nn.Module):
    """Base class for SALT layers handling the SVD and LoRA logic."""

    def __init__(self, rank, r_lora, seed=42):
        super().__init__()
        self.salt_rank = rank
        self.r_lora = r_lora
        self.done_svd = False
        self.last_reg_loss = 0.0
        torch.manual_seed(seed)

    def _init_params(self, weight, device):
        # 1. Reshape weight to 2D matrix for SVD
        # Conv3d: [Out, In, D, H, W] -> [Out, In*D*H*W]
        # Linear: [Out, In]
        original_shape = weight.shape
        if len(original_shape) > 2:
            weight_reshaped = rearrange(weight, 'co ... -> co (...)')
        else:
            weight_reshaped = weight

        # 2. SVD
        try:
            U, S, Vt = torch.linalg.svd(weight_reshaped, full_matrices=False)
        except RuntimeError:
            # Fallback for unstable SVD on GPU
            U, S, Vt = torch.linalg.svd(weight_reshaped.float().cpu(), full_matrices=False)
            U, S, Vt = U.to(device), S.to(device), Vt.to(device)

        self.register_buffer('U', U.detach())
        self.register_buffer('S', S.detach())
        self.register_buffer('Vt', Vt.detach())

        # 3. Trainable Parameters
        max_rank = min(self.U.shape[1], self.S.shape[0], self.Vt.shape[0])
        effective_rank = min(self.salt_rank, max_rank)
        self.salt_rank = effective_rank

        # Scale & Shift for top-r singular values
        self.trainable_scale_A = nn.Parameter(torch.ones(effective_rank, device=device))
        self.trainable_shift_B = nn.Parameter(torch.zeros(effective_rank, device=device))

        # LoRA for residual subspace
        residual_rank = max_rank - effective_rank
        if residual_rank > 0 and self.r_lora > 0:
            self.trainable_X = nn.Parameter(torch.randn(residual_rank, self.r_lora, device=device) * 0.01)
            self.trainable_Y = nn.Parameter(torch.randn(self.r_lora, residual_rank, device=device) * 0.01)
        else:
            self.register_parameter('trainable_X', None)
            self.register_parameter('trainable_Y', None)

        self.done_svd = True

    def get_updated_weight_flat(self):
        S_diag = torch.diag(self.S)
        top_s = self.S[:self.salt_rank]

        # Apply Scale & Shift
        modified_top_s = self.trainable_scale_A * top_s + self.trainable_shift_B

        # Apply LoRA
        lora_term = 0
        if self.trainable_X is not None and self.trainable_Y is not None:
            lora_term = self.trainable_X @ self.trainable_Y

        # Reconstruct S matrix
        new_s_matrix = S_diag.clone()
        new_s_matrix[:self.salt_rank, :self.salt_rank] = torch.diag(modified_top_s)

        if isinstance(lora_term, torch.Tensor):
            new_s_matrix[self.salt_rank:, self.salt_rank:] += lora_term

        # Regularization calculation (paper formula)
        scale_shift_diff = torch.norm(torch.diag(modified_top_s) - torch.diag(top_s))
        lora_norm = torch.norm(lora_term) if isinstance(lora_term, torch.Tensor) else 0.0
        self.last_reg_loss = scale_shift_diff + lora_norm

        # Reconstruct Weight: U * S' * Vt
        # Note: Paper suggests ReLU on S to ensure non-negativity, but standard SVD S are positive.
        # We follow standard reconstruction.
        weight_flat = self.U @ new_s_matrix @ self.Vt
        return weight_flat


class SALTConv3d(SALTLayerBase):
    def __init__(self, original_layer: nn.Conv3d, rank: int, r_lora: int):
        super().__init__(rank, r_lora)
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False  # Freeze original
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        self._init_params(original_layer.weight, original_layer.weight.device)

    def forward(self, x):
        weight_flat = self.get_updated_weight_flat()
        # Reshape back to [Co, Ci, D, H, W]
        weight = rearrange(
            weight_flat,
            'co (cin d h w) -> co cin d h w',
            cin=self.original_layer.in_channels,
            d=self.original_layer.kernel_size[0],
            h=self.original_layer.kernel_size[1],
            w=self.original_layer.kernel_size[2],
        )
        return F.conv3d(
            x, weight, self.original_layer.bias,
            self.original_layer.stride, self.original_layer.padding,
            self.original_layer.dilation, self.original_layer.groups,
        )


class SALTLinear(SALTLayerBase):
    def __init__(self, original_layer: nn.Linear, rank: int, r_lora: int):
        super().__init__(rank, r_lora)
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        self._init_params(original_layer.weight, original_layer.weight.device)

    def forward(self, x):
        weight_flat = self.get_updated_weight_flat()
        # Linear weight is [Out, In], matched flat
        return F.linear(x, weight_flat, self.original_layer.bias)


def apply_salt_to_model(model: nn.Module, rank: int, r_lora: int, verbose: bool = True):
    """Recursively replace Conv3d and Linear layers with SALT equivalents."""
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    converted_count = 0
    for _name, _module in model.named_modules():
        # Skip scalar parameters or specialized modules if needed
        pass

    # We iterate and replace in parent modules
    # Note: We must collect replaceables first to avoid modifying OrderedDict while iterating
    replace_list = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            # Heuristic: Skip 1x1x1 convs used for classification heads if desired,
            # but usually fine-tuning everything is okay.
            replace_list.append((name, module, 'conv3d'))
        elif isinstance(module, nn.Linear):
            replace_list.append((name, module, 'linear'))

    for name, module, m_type in replace_list:
        # Navigate to parent
        tokens = name.split('.')
        parent = model
        for token in tokens[:-1]:
            parent = getattr(parent, token)
        child_name = tokens[-1]

        if m_type == 'conv3d':
            salt_layer = SALTConv3d(module, rank, r_lora)
        else:
            salt_layer = SALTLinear(module, rank, r_lora)

        setattr(parent, child_name, salt_layer)
        converted_count += 1

    if verbose:
        print(f"🧂 SALT: Converted {converted_count} layers (Conv3d/Linear) to SALT layers.")

    return model


# =========================================================================
# Part 2: SimplifiedDAUnetModule (Copied & Adapted from L2-SP/age_aware_modules.py)
# =========================================================================


def prepare_class_ratios(prior_data: Dict, expected_num_classes: int, foreground_only: bool, is_main: bool = False):
    ratios = np.asarray(prior_data.get("class_ratios", []), dtype=np.float64)
    if ratios.size == 0:
        volume_keys = [k for k, v in prior_data.items() if isinstance(v, dict) and "means" in v]
        if volume_keys:
            accum = np.zeros(expected_num_classes, dtype=np.float64)
            counts = np.zeros(expected_num_classes, dtype=np.float64)
            for key in volume_keys:
                entry = prior_data[key]
                means = np.asarray(entry.get("means", []), dtype=np.float64)
                n = np.asarray(entry.get("n", []), dtype=np.float64)
                means = np.pad(means, (0, max(0, expected_num_classes - means.size)))[:expected_num_classes]
                if n.size == 0:
                    n = np.ones_like(means)
                n = np.pad(n, (0, max(0, expected_num_classes - n.size)))[:expected_num_classes]
                accum += means * n
                counts += n
            valid = counts > 0
            ratios = np.zeros(expected_num_classes, dtype=np.float64)
            ratios[valid] = accum[valid] / counts[valid]

    if foreground_only and ratios.size > expected_num_classes:
        ratios = ratios[1:]

    if ratios.size != expected_num_classes:
        ratios = np.ones(expected_num_classes, dtype=np.float64)
    return ratios


def _load_class_weights(stats_path, num_classes, foreground_only, enhanced, device=None, is_main=False):
    if stats_path is None or not os.path.exists(stats_path):
        return None
    with open(stats_path, "r") as f:
        prior_data = json.load(f)

    ratios = prepare_class_ratios(prior_data, num_classes, foreground_only, is_main=is_main)
    eps = 1e-7
    weights = 1.0 / (ratios + eps)
    weights = weights / weights.mean()
    if enhanced:
        weights = np.log1p(weights)
        weights = np.clip(weights, 0.1, 20.0)
    else:
        weights = np.sqrt(weights)
        weights = np.clip(weights, 0.1, 10.0)
    weights = weights / weights.mean()
    tensor = torch.as_tensor(weights, dtype=torch.float32)
    if device:
        tensor = tensor.to(device)
    return tensor


class SimplifiedDAUnetModule(nn.Module):
    def __init__(self, backbone, num_classes, *, volume_stats_path=None, foreground_only=True, enhanced_class_weights=True, use_age_conditioning=False, debug_mode=False):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.foreground_only = foreground_only

        is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        device = next(backbone.parameters()).device if list(backbone.parameters()) else None

        self.class_weights = _load_class_weights(
            volume_stats_path, num_classes, foreground_only, enhanced_class_weights, device=device, is_main=is_main
        )
        if is_main:
            print(f"✅ SimplifiedDAUnetModule (SALT-ready) initialised. Weights loaded: {self.class_weights is not None}")

    def forward(self, x, age=None):
        return self.backbone(x)

    def get_class_weights(self):
        return self.class_weights
