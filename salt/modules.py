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
# Part 1: SALT Layers (Optimized for Speed)
# =========================================================================

class SALTLayerBase(nn.Module):
    """Base class for SALT layers handling the SVD and LoRA logic.
    Optimized to avoid constructing dense (Rank, Rank) intermediate matrices.
    """

    def __init__(self, rank, r_lora, seed=42):
        super().__init__()
        self.salt_rank = rank
        self.r_lora = r_lora
        self.done_svd = False
        self.last_reg_loss = 0.0
        torch.manual_seed(seed)

    def _init_params(self, weight, device):
        # 1. Reshape weight to 2D matrix for SVD
        original_shape = weight.shape
        if len(original_shape) > 2:
            # Conv3d: [Out, In, D, H, W] -> [Out, In*D*H*W]
            weight_reshaped = rearrange(weight, 'co ... -> co (...)')
        else:
            # Linear: [Out, In]
            weight_reshaped = weight

        # 2. SVD
        try:
            U, S, Vt = torch.linalg.svd(weight_reshaped, full_matrices=False)
        except RuntimeError:
            U, S, Vt = torch.linalg.svd(weight_reshaped.float().cpu(), full_matrices=False)
            U, S, Vt = U.to(device), S.to(device), Vt.to(device)

        # 3. Truncate SVD components strictly to salt_rank to save memory/compute
        # We only keep the top `salt_rank` components.
        max_rank = min(U.shape[1], S.shape[0], Vt.shape[0])
        effective_rank = min(self.salt_rank, max_rank)
        self.salt_rank = effective_rank

        # Store only the necessary parts
        self.register_buffer('U_trunc', U[:, :effective_rank].detach().clone())  # [Out, Rank]
        self.register_buffer('S_trunc', S[:effective_rank].detach().clone())  # [Rank]
        self.register_buffer('Vt_trunc', Vt[:effective_rank, :].detach().clone())  # [Rank, In_Flat]

        # 4. Trainable Parameters (Scale & Shift)
        self.trainable_scale_A = nn.Parameter(torch.ones(effective_rank, device=device))
        self.trainable_shift_B = nn.Parameter(torch.zeros(effective_rank, device=device))

        # 5. LoRA for residual subspace
        # We project the residual via U_trunc and Vt_trunc for parameter efficiency in SALT
        if self.r_lora > 0:
            self.trainable_X = nn.Parameter(torch.randn(effective_rank, self.r_lora, device=device) * 0.01)
            self.trainable_Y = nn.Parameter(torch.randn(self.r_lora, effective_rank, device=device) * 0.01)
        else:
            self.register_parameter('trainable_X', None)
            self.register_parameter('trainable_Y', None)

        self.done_svd = True

        # Clean up large temporary tensors
        del U, S, Vt, weight_reshaped

    def get_updated_weight_flat(self):
        # Optimization: W = U * (S_mod + XY) * Vt
        # W = (U * S_mod) * Vt + (U * X) * (Y * Vt)

        # 1. Main Path: Scaled SVD
        # modified_s = A * S + B
        modified_top_s = self.trainable_scale_A * self.S_trunc + self.trainable_shift_B

        # Scale columns of U (Broadcasting) -> [Out, Rank]
        # This is faster than matrix multiplication with a diagonal matrix
        U_scaled = self.U_trunc * modified_top_s.view(1, -1)

        # Reconstruct base weight: [Out, Rank] @ [Rank, In_Flat] -> [Out, In_Flat]
        weight_base = U_scaled @ self.Vt_trunc

        # 2. LoRA Path (if applicable)
        if self.trainable_X is not None and self.trainable_Y is not None:
            # Decomposed multiplication to avoid large intermediate matrices
            # (Out, Rank) @ (Rank, r_lora) -> (Out, r_lora) (Small!)
            U_X = self.U_trunc @ self.trainable_X

            # (r_lora, Rank) @ (Rank, In_Flat) -> (r_lora, In_Flat)
            Y_Vt = self.trainable_Y @ self.Vt_trunc

            # (Out, r_lora) @ (r_lora, In_Flat) -> (Out, In_Flat)
            weight_lora = U_X @ Y_Vt

            # Add to base
            weight_flat = weight_base + weight_lora

            # Regularization Calculation (Compute only if needed)
            if self.training:
                lora_term = self.trainable_X @ self.trainable_Y
                scale_shift_diff = torch.norm(modified_top_s - self.S_trunc)
                lora_norm = torch.norm(lora_term)
                self.last_reg_loss = scale_shift_diff + lora_norm
        else:
            weight_flat = weight_base
            if self.training:
                self.last_reg_loss = torch.norm(modified_top_s - self.S_trunc)

        return weight_flat


class SALTConv3d(SALTLayerBase):
    def __init__(self, original_layer: nn.Conv3d, rank: int, r_lora: int):
        super().__init__(rank, r_lora)
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        # Move to device immediately to perform SVD
        device = original_layer.weight.device
        self._init_params(original_layer.weight, device)

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

        device = original_layer.weight.device
        self._init_params(original_layer.weight, device)

    def forward(self, x):
        weight_flat = self.get_updated_weight_flat()
        return F.linear(x, weight_flat, self.original_layer.bias)


def apply_salt_to_model(model: nn.Module, rank: int, r_lora: int, verbose: bool = True):
    """Recursively replace Conv3d and Linear layers with SALT equivalents."""
    for param in model.parameters():
        param.requires_grad = False

    converted_count = 0
    replace_list = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            # Verify it is a standard Conv3d (kernel size > 1 implies spatial features)
            # We adapt all of them to be safe
            replace_list.append((name, module, 'conv3d'))
        elif isinstance(module, nn.Linear):
            replace_list.append((name, module, 'linear'))

    for name, module, m_type in replace_list:
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
# Part 2: SimplifiedDAUnetModule (Helper classes)
# =========================================================================

def prepare_class_ratios(prior_data: Dict, expected_num_classes: int, foreground_only: bool, is_main: bool = False,
                         context: str = None):
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
    def __init__(self, backbone, num_classes, *, volume_stats_path=None, foreground_only=True,
                 enhanced_class_weights=True, use_age_conditioning=False, debug_mode=False):
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
            print(
                f"✅ SimplifiedDAUnetModule (SALT-ready) initialised. Weights loaded: {self.class_weights is not None}")

    def forward(self, x, age=None):
        return self.backbone(x)

    def get_class_weights(self):
        return self.class_weights