#!/usr/bin/env python3
"""
Age-conditioned graph-based anatomical prior loss for brain segmentation
Implements volume priors, shape priors, and weighted adjacency based on age
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
import re
from typing import Optional, Dict, List, Tuple, Any, Union
from monai.data import MetaTensor


_DEFAULT_VOLUME_STD_FLOOR = 0.02


def _to_tensor(x):
    """Convert MetaTensor to regular torch.Tensor if needed"""
    if x is None:
        return None
    if isinstance(x, MetaTensor):
        return x.as_tensor()
    return x

def soft_adjacency_from_probs(probs: torch.Tensor,
                              kernel_size: int = 3,
                              stride: int = 1,
                              temperature: float = 1.0,
                              restricted_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute soft adjacency matrix from probability maps with optional restrictions."""

    probs = _to_tensor(probs)

    if kernel_size > 1:
        probs_pooled = F.avg_pool3d(
            probs,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            count_include_pad=False,
        )
    else:
        probs_pooled = probs

    B, C, X_p, Y_p, Z_p = probs_pooled.shape
    probs_flat = probs_pooled.reshape(B, C, -1)
    probs_norm = probs_flat / (probs_flat.sum(dim=2, keepdim=True) + 1e-8)

    A_batch = torch.bmm(probs_norm, probs_norm.transpose(1, 2))

    if temperature != 1.0:
        A_batch = torch.pow(A_batch + 1e-8, 1.0 / temperature)

    eye = torch.eye(C, device=probs.device).unsqueeze(0)
    A_batch = A_batch * (1 - eye)

    if restricted_mask is not None:
        restricted_mask = _to_tensor(restricted_mask)
        if restricted_mask.dim() == 2:
            restricted_mask = restricted_mask.unsqueeze(0)
        A_batch = A_batch * restricted_mask

    row_sums = A_batch.sum(dim=2, keepdim=True).clamp(min=1e-8)
    A_batch = A_batch / row_sums

    return A_batch.mean(dim=0)


def _parse_age_key(raw_key: Union[str, int, float]) -> Optional[float]:
    """Parse various age bucket keys into a float value."""

    if isinstance(raw_key, (int, float)):
        return float(raw_key)

    if not isinstance(raw_key, str):
        return None

    token = raw_key.strip().lower()
    if not token:
        return None

    if token in {"unknown", "unknown_age", "nan"}:
        return -1.0

    match = re.search(r"-?\d+(?:\.\d+)?", token)
    if match is None:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def _coerce_shape_template_payload(payload: Any) -> Tuple[Dict[float, torch.Tensor], Dict[str, Any]]:
    """Convert torch.load payloads into an age→template dictionary."""

    metadata: Dict[str, Any] = {}
    if payload is None:
        return {}, metadata

    mapping: Optional[Dict] = None
    if isinstance(payload, dict):
        if 'mean' in payload and isinstance(payload['mean'], dict):
            mapping = payload['mean']
            metadata['has_std'] = bool(payload.get('std'))
            if 'num_classes' in payload:
                metadata['num_classes'] = int(payload['num_classes'])
        else:
            mapping = payload

    if mapping is None:
        return {}, metadata

    templates: Dict[float, torch.Tensor] = {}
    ignored_keys: List[str] = []

    for raw_key, value in mapping.items():
        age_key = _parse_age_key(raw_key)
        if age_key is None:
            ignored_keys.append(str(raw_key))
            continue

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().float()
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32)

        templates[age_key] = tensor

    if not templates:
        metadata['ignored_keys'] = ignored_keys
        return {}, metadata

    ages_sorted = sorted(templates.keys())
    metadata['ages'] = ages_sorted
    first_template = templates[ages_sorted[0]]
    metadata['spatial_shape'] = tuple(first_template.shape[1:])
    metadata['num_classes_in_template'] = first_template.shape[0]
    if ignored_keys:
        metadata['ignored_keys'] = ignored_keys

    return templates, metadata

def compute_laplacian(A: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """Compute (optionally normalized) graph Laplacian from adjacency matrix."""

    A = _to_tensor(A)
    A_sym = 0.5 * (A + A.T)

    degree = torch.relu(A_sym).sum(dim=1)
    D = torch.diag(degree)
    L = D - A_sym

    if normalized:
        safe_degree = degree.clamp(min=1e-8)
        inv_sqrt_vals = torch.zeros_like(safe_degree)
        mask = degree > 1e-6
        inv_sqrt_vals[mask] = 1.0 / torch.sqrt(safe_degree[mask])
        d_sqrt_inv = torch.diag(inv_sqrt_vals)
        L = d_sqrt_inv @ L @ d_sqrt_inv

    return L

def spectral_alignment_loss(L_pred: torch.Tensor,
                            L_prior: torch.Tensor,
                            top_k: int = 20,
                            align_vectors: bool = True,
                            eigenvalue_weighted: bool = False) -> torch.Tensor:
    """Align Laplacian spectra (and optionally eigenvectors) between two graphs."""

    L_pred = _to_tensor(L_pred)
    L_prior = _to_tensor(L_prior)

    L_pred_sym = 0.5 * (L_pred + L_pred.T)
    L_prior_sym = 0.5 * (L_prior + L_prior.T)

    evals_pred, evecs_pred = torch.linalg.eigh(L_pred_sym.float())
    evals_prior, evecs_prior = torch.linalg.eigh(L_prior_sym.float())

    evals_pred = evals_pred.to(L_pred.dtype)
    evals_prior = evals_prior.to(L_prior.dtype)
    evecs_pred = evecs_pred.to(L_pred.dtype)
    evecs_prior = evecs_prior.to(L_prior.dtype)

    k = min(top_k, evals_pred.shape[0] - 1)

    loss_evals = F.mse_loss(evals_pred[1:k + 1], evals_prior[1:k + 1])

    if not align_vectors or k <= 0:
        return loss_evals

    U_pred = evecs_pred[:, 1:k + 1]
    U_prior = evecs_prior[:, 1:k + 1]

    dots = (U_pred * U_prior).sum(dim=0)
    signs = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
    U_pred = U_pred * signs.unsqueeze(0)

    if eigenvalue_weighted:
        w = evals_prior[1:k + 1].abs()
        w = w / (w.sum() + 1e-8)
        loss_subspace = 0.0
        for i in range(k):
            cos_sim = torch.dot(U_pred[:, i], U_prior[:, i])
            loss_subspace += w[i] * (1.0 - cos_sim ** 2)
    else:
        P_pred = U_pred @ U_pred.T
        P_prior = U_prior @ U_prior.T
        loss_subspace = F.mse_loss(P_pred, P_prior)

    return loss_evals + 0.5 * loss_subspace

def edge_consistency_loss_with_mismatch(A_pred: torch.Tensor,
                                        A_prior: torch.Tensor,
                                        required_edges: Optional[List[Tuple[int, int]]] = None,
                                        forbidden_edges: Optional[List[Tuple[int, int]]] = None,
                                        margin: float = 0.1,
                                        class_weights: Optional[torch.Tensor] = None,
                                        qap_mismatch_g: float = 1.5,
                                        restricted_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Edge-level alignment with mismatch-aware penalties and optional masks."""

    A_pred = _to_tensor(A_pred)
    A_prior = _to_tensor(A_prior)

    if restricted_mask is not None:
        restricted_mask = _to_tensor(restricted_mask)
        A_pred = A_pred * restricted_mask
        A_prior = A_prior * restricted_mask

    if class_weights is not None:
        w = class_weights.view(-1, 1)
        W = torch.sqrt(w @ w.T)
        W = W / W.mean()
    else:
        W = torch.ones_like(A_pred)

    loss_mse = torch.mean(W * (A_pred - A_prior) ** 2)

    if restricted_mask is not None:
        R = restricted_mask
    else:
        R = torch.ones_like(A_pred)

    M = (A_pred * A_prior) * R
    N = ((1 - A_pred) * (1 - A_prior)) * R
    X = ((1 - A_prior) * A_pred + A_prior * (1 - A_pred)) * R
    loss_qap = qap_mismatch_g * X.mean() - M.mean()
    loss_base = loss_mse + 0.1 * loss_qap

    loss_required = torch.tensor(0.0, device=A_pred.device)
    th_required = 0.02
    if required_edges:
        for i, j in required_edges:
            if i < A_pred.shape[0] and j < A_pred.shape[1]:
                loss_required = loss_required + torch.pow(F.relu(th_required - A_pred[i, j]), 2)

    loss_forbidden = torch.tensor(0.0, device=A_pred.device)
    th_forbidden = 5e-4
    if forbidden_edges:
        for i, j in forbidden_edges:
            if i < A_pred.shape[0] and j < A_pred.shape[1]:
                loss_forbidden = loss_forbidden + torch.pow(F.relu(A_pred[i, j] - th_forbidden), 2)

    num_constraints = len(required_edges or []) + len(forbidden_edges or [])
    if num_constraints > 0:
        constraint_weight = 0.1
        loss_constraints = constraint_weight * (loss_required + loss_forbidden) / num_constraints
    else:
        loss_constraints = torch.tensor(0.0, device=A_pred.device)

    return loss_base + loss_constraints

def compute_restricted_mask(num_classes: int,
                            required_edges: List[Tuple[int, int]],
                            forbidden_edges: List[Tuple[int, int]],
                            lr_pairs: List[Tuple[int, int]],
                            device: torch.device) -> torch.Tensor:
    """Construct binary mask describing which adjacencies are valid."""

    R = torch.ones(num_classes, num_classes, device=device)
    R.fill_diagonal_(0)

    for i, j in forbidden_edges:
        if i < num_classes and j < num_classes:
            R[i, j] = 0
            R[j, i] = 0

    for i, j in required_edges:
        if i < num_classes and j < num_classes:
            R[i, j] = 1
            R[j, i] = 1

    for left, right in lr_pairs:
        if left < num_classes and right < num_classes:
            R[left, right] = 1
            R[right, left] = 1

    return R


def compute_sdt(mask: torch.Tensor, temperature: float = 4.0) -> torch.Tensor:
    """Differentiable surrogate for a signed distance transform.

    Instead of running a hard-thresholded distance transform (which would
    detach the computation graph), we rely on the logit of the soft mask as a
    smooth proxy for the signed distance. The temperature parameter controls
    the slope around the interface so that the dynamic range roughly matches
    pre-computed template SDTs.
    """

    mask = _to_tensor(mask)
    eps = 1e-4
    clamped = mask.clamp(min=eps, max=1.0 - eps)
    signed = -torch.log(clamped) + torch.log1p(-clamped)  # = -logit(mask)

    if temperature is not None and temperature > 0:
        signed = signed / temperature

    return signed


def compute_weighted_adjacency(probs: torch.Tensor, age: torch.Tensor,
                               age_weights: Optional[Dict] = None,
                               temperature: float = 1.0) -> torch.Tensor:
    """
    Compute weighted adjacency matrix based on age

    Args:
        probs: (B, C, X, Y, Z) probability maps
        age: (B, 1) age tensor
        age_weights: Dictionary mapping age ranges to weight matrices
        temperature: Temperature for sharpening

    Returns:
        A_weighted: (B, C, C) weighted adjacency matrix
    """
    B, C, X, Y, Z = probs.shape

    # Flatten spatial dimensions
    probs_flat = probs.reshape(B, C, -1)

    # Compute co-occurrence matrix
    A_batch = torch.bmm(probs_flat, probs_flat.transpose(1, 2))  # (B, C, C)

    # Apply temperature
    if temperature != 1.0:
        A_batch = torch.pow(A_batch + 1e-8, 1.0 / temperature)

    # Zero diagonal
    eye = torch.eye(C, device=probs.device).unsqueeze(0)
    A_batch = A_batch * (1 - eye)

    # Apply age-dependent weights if provided
    if age_weights is not None:
        for b in range(B):
            age_val = age[b].item()
            # Find appropriate weight matrix for this age
            weight_matrix = get_age_weight_matrix(age_val, age_weights, C, probs.device)
            A_batch[b] = A_batch[b] * weight_matrix

    # Row-normalize per-sample  (B, C, C)
    row_sums = A_batch.sum(dim=2, keepdim=True).clamp(min=1e-8)
    A_batch = A_batch / row_sums

    # 返回每个样本的邻接矩阵 (B, C, C)
    return A_batch


def get_age_weight_matrix(age: float, age_weights: Dict,
                          num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Get weight matrix for a specific age through interpolation

    Args:
        age: Age value
        age_weights: Dict with age as key and weight matrix as value
        num_classes: Number of classes
        device: Device

    Returns:
        weight_matrix: (C, C) interpolated weight matrix
    """
    if not age_weights:
        return torch.ones(num_classes, num_classes, device=device)

    # Get sorted ages
    ages = sorted(list(age_weights.keys()))

    # Find bracketing ages
    if age <= ages[0]:
        return torch.tensor(age_weights[ages[0]], device=device)
    if age >= ages[-1]:
        return torch.tensor(age_weights[ages[-1]], device=device)

    # Linear interpolation
    for i in range(len(ages) - 1):
        if ages[i] <= age <= ages[i + 1]:
            alpha = (age - ages[i]) / (ages[i + 1] - ages[i])
            w1 = torch.tensor(age_weights[ages[i]], device=device)
            w2 = torch.tensor(age_weights[ages[i + 1]], device=device)
            return (1 - alpha) * w1 + alpha * w2

    return torch.ones(num_classes, num_classes, device=device)


def _detect_background_index(volume_stats: Dict, num_classes: int) -> Optional[int]:
    """Infer which channel in the prior corresponds to background.

    We treat a channel as background if either (1) the statistics provide one
    more entry than the model predicts classes, or (2) a single channel
    dominates the distribution (mean > 0.5 and the remainder < 0.5)."""

    for stats in volume_stats.values():
        means = np.asarray(stats.get('means', []), dtype=np.float64)
        if means.size == 0:
            continue

        candidate_idx = int(np.argmax(means))
        candidate_val = float(means[candidate_idx])
        remainder = float(means.sum() - candidate_val)

        if means.size > num_classes and candidate_val >= remainder:
            return min(candidate_idx, num_classes - 1)
        if candidate_val > 0.5 and remainder < 0.5:
            return min(candidate_idx, num_classes - 1)

    return None


def _align_volume_stats(volume_stats: Dict, num_classes: int,
                        std_floor: float = _DEFAULT_VOLUME_STD_FLOOR) -> Tuple[Dict, Optional[int], np.ndarray]:
    """Align raw volume statistics to match the model's foreground classes."""

    if not volume_stats:
        return {}, None, np.ones(num_classes, dtype=np.float32)

    background_idx = _detect_background_index(volume_stats, num_classes)
    aligned: Dict[float, Dict[str, List[float]]] = {}

    valid_mask = np.ones(num_classes, dtype=np.float32)

    min_std = std_floor if std_floor is not None else 0.0
    if min_std <= 0:
        min_std = 1e-6

    for age_key, stats in volume_stats.items():
        means = np.asarray(stats.get('means', []), dtype=np.float64)
        stds = np.asarray(stats.get('stds', []), dtype=np.float64)

        if stds.shape[0] != means.shape[0]:
            if stds.shape[0] > means.shape[0]:
                stds = stds[: means.shape[0]]
            else:
                stds = np.pad(stds, (0, means.shape[0] - stds.shape[0]), constant_values=1.0)

        if background_idx is not None and 0 <= background_idx < means.shape[0]:
            means = np.delete(means, background_idx)
            stds = np.delete(stds, background_idx)

        if means.size == 0:
            fg_means = np.zeros(num_classes, dtype=np.float64)
            fg_stds = np.ones(num_classes, dtype=np.float64)
            valid_mask.fill(0.0)
        else:
            if means.size > num_classes:
                fg_means = means[:num_classes]
                fg_stds = stds[:num_classes]
            else:
                pad = num_classes - means.size
                fg_means = np.pad(means, (0, pad), constant_values=0.0)
                fg_stds = np.pad(stds, (0, pad), constant_values=1.0)
                if pad > 0:
                    valid_mask[means.size:] = 0.0

        total = fg_means.sum()
        if total > 1e-6:
            fg_means = fg_means / total
            fg_stds = fg_stds / total

        fg_stds = np.maximum(fg_stds, min_std)

        aligned[float(age_key)] = {
            'means': fg_means.astype(np.float32).tolist(),
            'stds': fg_stds.astype(np.float32).tolist(),
        }

    return aligned, background_idx, valid_mask


def _align_shape_channels(template: torch.Tensor, target_channels: int) -> torch.Tensor:
    """Match template channel dimension to the network output channels."""

    if template.dim() != 4:
        return template

    aligned = template

    if aligned.shape[0] > target_channels:
        flat = aligned.view(aligned.shape[0], -1).abs().sum(dim=1)
        while aligned.shape[0] > target_channels:
            drop_idx = int(torch.argmax(flat).item())
            aligned = torch.cat([aligned[:drop_idx], aligned[drop_idx + 1:]], dim=0)
            flat = aligned.view(aligned.shape[0], -1).abs().sum(dim=1)
    elif aligned.shape[0] < target_channels:
        pad = target_channels - aligned.shape[0]
        if pad > 0:
            zeros = torch.zeros((pad, *aligned.shape[1:]), device=aligned.device, dtype=aligned.dtype)
            aligned = torch.cat([aligned, zeros], dim=0)

    return aligned


def volume_consistency_loss(probs: torch.Tensor, age: torch.Tensor,
                            volume_stats: Dict, num_classes: int,
                            valid_mask: Optional[torch.Tensor] = None,
                            std_floor: float = _DEFAULT_VOLUME_STD_FLOOR) -> torch.Tensor:
    """
    Compute volume consistency loss based on age-specific statistics

    Args:
        probs: (B, C, X, Y, Z) probability maps
        age: (B, 1) age tensor
        volume_stats: Dictionary with age-specific volume statistics
        num_classes: Number of classes

    Returns:
        loss: Volume consistency loss
    """
    B = probs.shape[0]

    # Compute predicted volumes
    predicted_volumes = probs.sum(dim=(2, 3, 4))  # (B, C)
    # 将预测量归一化为“体积分数”，和先验统计（分数）一致
    sum_all = predicted_volumes.sum(dim=1, keepdim=True).clamp(min=1e-6)
    predicted_volumes = predicted_volumes / sum_all
    # Get expected volumes for each age
    expected_volumes = torch.zeros(B, num_classes, device=probs.device)
    expected_stds = torch.ones(B, num_classes, device=probs.device)

    for b in range(B):
        age_val = age[b].item()
        volumes, stds = get_expected_volumes(
            age_val,
            volume_stats,
            num_classes,
            std_floor=std_floor,
        )
        expected_volumes[b] = torch.tensor(volumes, device=probs.device)
        expected_stds[b] = torch.tensor(stds, device=probs.device)

    # Normalize by expected std
    min_std = std_floor if std_floor is not None else 0.0
    if min_std <= 0:
        min_std = 1e-6
    expected_stds = torch.clamp(expected_stds, min=min_std)
    normalized_diff = (predicted_volumes - expected_volumes) / expected_stds

    if valid_mask is not None:
        mask = valid_mask.view(1, -1).to(device=probs.device, dtype=probs.dtype)
        normalized_diff = normalized_diff * mask
        denom = mask.sum().clamp(min=1.0)
    else:
        denom = torch.tensor(normalized_diff.numel(), device=probs.device, dtype=probs.dtype)

    zero_target = torch.zeros_like(normalized_diff)
    loss = F.smooth_l1_loss(normalized_diff, zero_target, reduction='sum') / denom

    return loss


def get_expected_volumes(age: float, volume_stats: Dict,
                         num_classes: int,
                         std_floor: float = _DEFAULT_VOLUME_STD_FLOOR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get expected volume statistics for a given age

    Args:
        age: Age value
        volume_stats: Dictionary with age-specific statistics
        num_classes: Number of classes

    Returns:
        means: Expected volume means
        stds: Expected volume stds
    """
    min_std = std_floor if std_floor is not None else 0.0
    if min_std <= 0:
        min_std = 1e-6

    if not volume_stats:
        means = np.ones(num_classes)
        stds = np.maximum(np.ones(num_classes), min_std)
        return means, stds

    # Find closest age or interpolate
    ages = sorted(list(volume_stats.keys()))

    if age <= ages[0]:
        stats = volume_stats[ages[0]]
        means = np.array(stats['means'])
        stds = np.maximum(np.array(stats['stds']), min_std)
        return means, stds

    if age >= ages[-1]:
        stats = volume_stats[ages[-1]]
        means = np.array(stats['means'])
        stds = np.maximum(np.array(stats['stds']), min_std)
        return means, stds

    # Linear interpolation
    for i in range(len(ages) - 1):
        if ages[i] <= age <= ages[i + 1]:
            alpha = (age - ages[i]) / (ages[i + 1] - ages[i])
            means1 = np.array(volume_stats[ages[i]]['means'])
            stds1 = np.array(volume_stats[ages[i]]['stds'])
            means2 = np.array(volume_stats[ages[i + 1]]['means'])
            stds2 = np.array(volume_stats[ages[i + 1]]['stds'])

            means = (1 - alpha) * means1 + alpha * means2
            stds = (1 - alpha) * stds1 + alpha * stds2
            stds = np.maximum(stds, min_std)
            return means, stds

    means = np.ones(num_classes)
    stds = np.maximum(np.ones(num_classes), min_std)
    return means, stds


def shape_consistency_loss(probs: torch.Tensor,
                           age: torch.Tensor,
                           shape_templates: Optional[Dict] = None,
                           temperature: float = 4.0,
                           normalize_templates: bool = True) -> torch.Tensor:
    """
    Compute shape consistency loss using signed distance transforms

    Args:
        probs: (B, C, X, Y, Z) probability maps
        age: (B, 1) age tensor
        shape_templates: Dictionary with age-specific shape templates

    Returns:
        loss: Shape consistency loss
    """
    if shape_templates is None:
        return torch.tensor(0.0, device=probs.device)

    # Compute differentiable SDT surrogate for predictions
    sdt_pred = compute_sdt(probs, temperature=temperature)

    if normalize_templates:
        pred_norm = sdt_pred / sdt_pred.detach().abs().amax(dim=(2, 3, 4), keepdim=True).clamp(min=1e-6)
    else:
        pred_norm = sdt_pred

    # Get expected SDT for each age
    B = probs.shape[0]
    loss = torch.zeros(1, device=probs.device, dtype=probs.dtype)
    valid = 0

    for b in range(B):
        age_val = age[b].item()
        sdt_expected = get_shape_template(age_val, shape_templates, probs.shape[1:], probs.device)

        if sdt_expected is not None:
            sdt_expected = sdt_expected.to(device=probs.device, dtype=probs.dtype)
            if normalize_templates:
                sdt_expected = sdt_expected / sdt_expected.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)

            loss = loss + F.l1_loss(pred_norm[b], sdt_expected)
            valid += 1

    if valid == 0:
        return torch.zeros(1, device=probs.device, dtype=probs.dtype)

    return loss / valid


def get_shape_template(age: float, shape_templates: Dict,
                       shape: Tuple, device: torch.device) -> Optional[torch.Tensor]:
    """
    Get shape template for a specific age

    Args:
        age: Age value
        shape_templates: Dictionary with age-specific templates
        shape: Expected shape (C, X, Y, Z)
        device: Device

    Returns:
        template: Shape template tensor or None
    """
    if not shape_templates:
        return None

    # Find closest age
    ages = sorted(list(shape_templates.keys()))
    closest_age = min(ages, key=lambda x: abs(x - age))

    template = shape_templates[closest_age]

    # Convert to tensor and resize if needed
    if isinstance(template, np.ndarray):
        template = torch.from_numpy(template).to(device)
    else:
        template = template.to(device)

    template = _align_shape_channels(template, shape[0])

    # Ensure correct spatial shape
    if template.shape[1:] != shape[1:]:
        template = F.interpolate(
            template.unsqueeze(0),
            size=shape[1:],
            mode='trilinear',
            align_corners=False
        ).squeeze(0)

    return template


class AgeConditionedGraphPriorLoss(nn.Module):
    """Unified age-conditioned prior with static cross-domain graph alignment support."""

    def __init__(self,
                 volume_stats_path: Optional[str] = None,
                 shape_templates_path: Optional[str] = None,
                 weighted_adj_path: Optional[str] = None,
                 age_weights_path: Optional[str] = None,
                 prior_adj_path: Optional[str] = None,
                 required_json: Optional[str] = None,
                 forbidden_json: Optional[str] = None,
                 lr_pairs_json: Optional[str] = None,
                 src_prior_adj_path: Optional[str] = None,
                 src_required_json: Optional[str] = None,
                 src_forbidden_json: Optional[str] = None,
                 lambda_volume: float = 0.2,
                 lambda_shape: float = 0.0,
                 lambda_weighted_adj: float = 0.15,
                 lambda_topo: float = 0.02,
                 lambda_sym: float = 0.05,
                 lambda_spec: float = 0.1,
                 lambda_edge: float = 0.1,
                 lambda_spec_src: Optional[float] = None,
                 lambda_edge_src: Optional[float] = None,
                 lambda_spec_tgt: Optional[float] = None,
                 lambda_edge_tgt: Optional[float] = None,
                 top_k: int = 20,
                 temperature: float = 1.0,
                 warmup_epochs: Optional[int] = 10,
                 pool_kernel: int = 3,
                 pool_stride: int = 2,
                 graph_align_mode: str = 'joint',
                 class_weights: Optional[torch.Tensor] = None,
                 align_U_weighted: bool = False,
                 qap_mismatch_g: float = 1.5,
                 use_restricted_mask: bool = False,
                 restricted_mask_path: Optional[str] = None,
                 prior_warmup_epochs: Optional[int] = None,
                 prior_temperature: Optional[float] = None,
                 volume_std_floor: float = _DEFAULT_VOLUME_STD_FLOOR,
                 debug_mode: bool = False,
                 debug_max_batches: int = 2,
                 **extra_kwargs):
        super().__init__()

        # Age-aware coefficients
        self.lambda_volume = lambda_volume
        self.lambda_shape = lambda_shape
        self.lambda_weighted_adj = lambda_weighted_adj
        self.lambda_topo = lambda_topo
        self.lambda_sym = lambda_sym
        self.weighted_adj_presence_threshold = float(
            extra_kwargs.pop('weighted_adj_presence_threshold', 1e-3)
        )
        self.weighted_adj_presence_slope = float(
            extra_kwargs.pop('weighted_adj_presence_slope', 40.0)
        )
        self.shape_temperature = float(extra_kwargs.pop('shape_temperature', 4.0))
        self.normalize_shape_templates = bool(
            extra_kwargs.pop('normalize_shape_templates', True)
        )
        self.volume_std_floor = float(volume_std_floor)

        # Graph alignment coefficients
        self.graph_align_mode = graph_align_mode
        self.align_U_weighted = align_U_weighted
        self.qap_mismatch_g = qap_mismatch_g
        self.lambda_spec_src = lambda_spec_src if lambda_spec_src is not None else (lambda_spec if graph_align_mode in ['src_only', 'joint'] else 0.0)
        self.lambda_edge_src = lambda_edge_src if lambda_edge_src is not None else (lambda_edge if graph_align_mode in ['src_only', 'joint'] else 0.0)
        self.lambda_spec_tgt = lambda_spec_tgt if lambda_spec_tgt is not None else (lambda_spec if graph_align_mode in ['tgt_only', 'joint'] else 0.0)
        self.lambda_edge_tgt = lambda_edge_tgt if lambda_edge_tgt is not None else (lambda_edge if graph_align_mode in ['tgt_only', 'joint'] else 0.0)

        self.top_k = top_k
        self.temperature = temperature
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride

        # Warmup scheduling for prior terms (age-conditioned + graph branches)
        self.graph_warmup_epochs = warmup_epochs if warmup_epochs is not None else 0
        self.age_warmup_epochs = (
            prior_warmup_epochs if prior_warmup_epochs is not None else self.graph_warmup_epochs
        )
        self.current_epoch = 0

        # Separate temperature for weighted adjacency if provided
        self.prior_temperature = prior_temperature if prior_temperature is not None else temperature

        self.debug_mode = bool(debug_mode)
        self.debug_max_batches = max(1, int(debug_max_batches)) if self.debug_mode else 0
        self._debug_batch_count = 0

        # Track whether the loaded priors align with the foreground-only label space
        self._prior_alignment_ok = True
        self._alignment_warning_logged = False

        # ========================= Age-aware priors =========================
        self.volume_stats = None
        if volume_stats_path and os.path.exists(volume_stats_path):
            with open(volume_stats_path, 'r') as f:
                tmp_vs = json.load(f)
                self.volume_stats = {float(k): v for k, v in tmp_vs.items()}
            print(f"✓ Loaded volume statistics from {volume_stats_path}")

        self._volume_stats_aligned_to: Optional[int] = None
        self.volume_background_idx: Optional[int] = None
        self._volume_alignment_logged = False
        self.register_buffer('volume_valid_mask', torch.empty(0), persistent=False)

        self.shape_templates = None
        self.shape_template_metadata: Dict[str, Any] = {}
        self.shape_template_path = shape_templates_path
        if shape_templates_path and os.path.exists(shape_templates_path):
            raw_templates = torch.load(shape_templates_path, map_location='cpu')
            templates, metadata = _coerce_shape_template_payload(raw_templates)
            if templates:
                self.shape_templates = templates
                self.shape_template_metadata = metadata
                ages = metadata.get('ages', [])
                is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
                if is_main:
                    if ages:
                        known_ages = [a for a in ages if a >= 0]
                        if known_ages:
                            age_range = f"{min(known_ages):.1f}-{max(known_ages):.1f}w"
                        else:
                            age_range = "unknown"
                        print(
                            f"✓ Loaded shape templates from {shape_templates_path} "
                            f"({len(ages)} bins, age range: {age_range})"
                        )
                    else:
                        print(f"✓ Loaded shape templates from {shape_templates_path}")
                    if metadata.get('ignored_keys'):
                        print(
                            "  ⚠️ Ignored non-age template keys: "
                            + ", ".join(metadata['ignored_keys'])
                        )
            else:
                is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
                if is_main:
                    print(
                        f"⚠️  Failed to parse shape templates payload at {shape_templates_path}; "
                        "disabling λ_shape."
                    )
                self.lambda_shape = 0.0
        elif shape_templates_path and self.lambda_shape > 0:
            is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
            if is_main:
                print(
                    f"⚠️  Shape templates file not found at {shape_templates_path}; disabling λ_shape."
                )
            self.lambda_shape = 0.0

        self.age_weights = None
        if age_weights_path and os.path.exists(age_weights_path):
            with open(age_weights_path, 'r') as f:
                tmp_aw = json.load(f)
                self.age_weights = {float(k): v for k, v in tmp_aw.items()}
            print(f"✓ Loaded age-dependent weights from {age_weights_path}")

        if weighted_adj_path and os.path.exists(weighted_adj_path):
            A_weighted = np.load(weighted_adj_path).astype(np.float32)
            row_sums = np.clip(A_weighted.sum(axis=1, keepdims=True), 1e-8, None)
            A_weighted = A_weighted / row_sums
            self.register_buffer('A_weighted_prior', torch.from_numpy(A_weighted))
            print(f"✓ Loaded weighted adjacency prior from {weighted_adj_path}")
            if self.lambda_weighted_adj > 0:
                if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                    print(
                        "  Weighted adjacency active rows require "
                        f"≥{self.weighted_adj_presence_threshold:.4f} volume fraction"
                    )
        else:
            self.register_buffer('A_weighted_prior', torch.empty(0))
            if lambda_weighted_adj > 0:
                is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
                if is_main:
                    print("⚠️  Weighted adjacency prior not provided – disabling λ_weighted_adj term.")
                self.lambda_weighted_adj = 0.0

        # ========================= Graph priors =========================
        self.lr_pairs: List[Tuple[int, int]] = []
        if lr_pairs_json and os.path.exists(lr_pairs_json):
            with open(lr_pairs_json, 'r') as f:
                pairs_raw = json.load(f)
                self.lr_pairs = [(int(a) - 1, int(b) - 1) for a, b in pairs_raw if int(a) > 0 and int(b) > 0]

        self.tgt_required_edges: List[Tuple[int, int]] = []
        if required_json and os.path.exists(required_json):
            with open(required_json, 'r') as f:
                data = json.load(f)
                self.tgt_required_edges = [(int(i), int(j)) for i, j in data.get('required', [])]

        self.tgt_forbidden_edges: List[Tuple[int, int]] = []
        if forbidden_json and os.path.exists(forbidden_json):
            with open(forbidden_json, 'r') as f:
                data = json.load(f)
                self.tgt_forbidden_edges = [(int(i), int(j)) for i, j in data.get('forbidden', [])]

        # Alias for topology regularizer
        self.required_edges = self.tgt_required_edges
        self.forbidden_edges = self.tgt_forbidden_edges

        if prior_adj_path and os.path.exists(prior_adj_path):
            A_tgt = torch.from_numpy(np.load(prior_adj_path)).float()
            row_sums = A_tgt.sum(dim=1, keepdim=True).clamp(min=1e-8)
            A_tgt = A_tgt / row_sums
            self.register_buffer('A_tgt', A_tgt)
            self.register_buffer('L_tgt', compute_laplacian(A_tgt, normalized=True))
            self.has_target_prior = True
        else:
            self.register_buffer('A_tgt', torch.empty(0))
            self.register_buffer('L_tgt', torch.empty(0))
            self.has_target_prior = False

        self.src_required_edges: List[Tuple[int, int]] = []
        self.src_forbidden_edges: List[Tuple[int, int]] = []
        if src_required_json and os.path.exists(src_required_json):
            with open(src_required_json, 'r') as f:
                data = json.load(f)
                self.src_required_edges = [(int(i), int(j)) for i, j in data.get('required', [])]

        if src_forbidden_json and os.path.exists(src_forbidden_json):
            with open(src_forbidden_json, 'r') as f:
                data = json.load(f)
                self.src_forbidden_edges = [(int(i), int(j)) for i, j in data.get('forbidden', [])]

        if src_prior_adj_path and os.path.exists(src_prior_adj_path) and graph_align_mode in ['src_only', 'joint']:
            A_src = torch.from_numpy(np.load(src_prior_adj_path)).float()
            row_sums = A_src.sum(dim=1, keepdim=True).clamp(min=1e-8)
            A_src = A_src / row_sums
            self.register_buffer('A_src', A_src)
            self.register_buffer('L_src', compute_laplacian(A_src, normalized=True))
            self.has_source_prior = True
        else:
            self.register_buffer('A_src', torch.empty(0))
            self.register_buffer('L_src', torch.empty(0))
            self.has_source_prior = False

        # Class weights for edge reweighting
        if class_weights is not None:
            class_weights = _to_tensor(class_weights).float()
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        # Keep track of number of classes if known
        self.num_classes = None
        if self.has_target_prior:
            self.num_classes = self.A_tgt.shape[0]
        elif self.has_source_prior:
            self.num_classes = self.A_src.shape[0]
        elif self.A_weighted_prior.numel() > 0:
            self.num_classes = self.A_weighted_prior.shape[0]

        # Restricted mask (load or derive)
        mask_tensor = torch.empty(0)
        if use_restricted_mask:
            if restricted_mask_path and os.path.exists(restricted_mask_path):
                mask_tensor = torch.from_numpy(np.load(restricted_mask_path)).float()
            elif self.num_classes is not None:
                mask_tensor = compute_restricted_mask(
                    self.num_classes,
                    self.tgt_required_edges + self.src_required_edges,
                    self.tgt_forbidden_edges + self.src_forbidden_edges,
                    self.lr_pairs,
                    device=torch.device('cpu'),
                ).float()
        self.use_restricted_mask = use_restricted_mask and mask_tensor.numel() > 0
        self.register_buffer('R_mask', mask_tensor if mask_tensor.numel() > 0 else torch.empty(0))

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_warmup_factor(self) -> float:
        if self.graph_warmup_epochs and self.current_epoch < self.graph_warmup_epochs:
            return self.current_epoch / max(1, self.graph_warmup_epochs)
        return 1.0

    def get_age_warmup_factor(self) -> float:
        if self.age_warmup_epochs and self.current_epoch < self.age_warmup_epochs:
            return self.current_epoch / max(1, self.age_warmup_epochs)
        return 1.0

    def _ensure_volume_stats_alignment(self, num_classes: int):
        if self.volume_stats is None:
            return

        if self._volume_stats_aligned_to == num_classes:
            return

        aligned, background_idx, mask_np = _align_volume_stats(
            self.volume_stats,
            num_classes,
            std_floor=self.volume_std_floor,
        )
        if aligned:
            self.volume_stats = aligned
        self.volume_background_idx = background_idx

        # If a dominant background channel is detected we disable prior terms to avoid
        # enforcing incorrect constraints (old priors were generated before foreground
        # remapping). Users should rebuild priors with the updated script.
        # 无论是否检测到背景通道，都保持先验有效；若存在背景通道则仅剔除该通道
        self._prior_alignment_ok = True

        if mask_np is None or mask_np.size == 0:
            mask_tensor = torch.ones(num_classes, dtype=torch.float32)
        else:
            mask_tensor = torch.from_numpy(mask_np.astype(np.float32))

        if self.volume_valid_mask.numel() == 0 or self.volume_valid_mask.shape[0] != mask_tensor.shape[0]:
            self.volume_valid_mask = mask_tensor
        else:
            self.volume_valid_mask.data = mask_tensor.to(self.volume_valid_mask.device)

        self._volume_stats_aligned_to = num_classes

        if (
            background_idx is not None
            and not self._volume_alignment_logged
            and ((not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0)
        ):
            print(
                f"[GraphPrior] Volume prior alignment: removed dominant channel {background_idx} from priors "
                f"and matched statistics to {num_classes} model classes"
            )
            self._volume_alignment_logged = True

    def forward(self,
                logits: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                age: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = _to_tensor(logits)
        labels = _to_tensor(labels)

        if age is None:
            age = torch.zeros(logits.shape[0], 1, device=logits.device, dtype=logits.dtype)
        else:
            age = _to_tensor(age)
            if age.dim() == 1:
                age = age.unsqueeze(1)
            if age.shape[1] > 1:
                age = age[:, :1]
            age = age.to(device=logits.device, dtype=logits.dtype)

        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]

        self._ensure_volume_stats_alignment(num_classes)

        if not self._prior_alignment_ok:
            if (not self._alignment_warning_logged
                    and ((not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0)):
                print("⚠️  Detected background-heavy priors. Skipping age-conditioned graph losses."
                      " Please regenerate priors with the updated build_graph_priors.py.")
                self._alignment_warning_logged = True

            zero = torch.zeros(1, device=logits.device, dtype=logits.dtype)
            zero_dict: Dict[str, torch.Tensor] = {
                'volume_loss': zero,
                'shape_loss': zero,
                'weighted_adj_loss': zero,
                'topo_loss': zero,
                'graph_spec_src': zero,
                'graph_edge_src': zero,
                'graph_spec_tgt': zero,
                'graph_edge_tgt': zero,
                'graph_sym': zero,
                'structural_violations': {'required_missing': 0, 'forbidden_present': 0},
                'weighted_adj_active_classes': zero,
                'warmup_factor': torch.tensor(self.get_warmup_factor(), device=logits.device, dtype=logits.dtype),
                'age_warmup_factor': torch.tensor(self.get_age_warmup_factor(), device=logits.device, dtype=logits.dtype),
            }
            return zero, zero_dict

        loss_dict: Dict[str, torch.Tensor] = {}
        dtype = logits.dtype
        device = logits.device

        age_loss_sum = torch.zeros(1, device=device, dtype=dtype)
        age_warmup = self.get_age_warmup_factor()

        predicted_volumes = probs.sum(dim=(2, 3, 4))
        total_volume = predicted_volumes.sum(dim=1, keepdim=True).clamp(min=1e-6)
        volume_fractions = predicted_volumes / total_volume

        debug_active = (
            self.debug_mode
            and (self._debug_batch_count < self.debug_max_batches)
            and ((not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0)
        )
        if debug_active:
            debug_prefix = f"[DEBUG][GraphPrior][batch {self._debug_batch_count + 1}]"
            print(f"{debug_prefix} logits shape={tuple(logits.shape)}")
            print(f"{debug_prefix} age range={float(age.min().item()):.2f}-{float(age.max().item()):.2f}")
            print(f"{debug_prefix} graph warmup={self.get_warmup_factor():.3f}, age warmup={age_warmup:.3f}")
            with torch.no_grad():
                vf = volume_fractions.detach()[0]
                top_mass = torch.topk(vf, k=min(5, vf.numel()))
                print(
                    f"{debug_prefix} volume fractions (sample0): "
                    + ", ".join(
                        f"c{idx}={vf[idx].item():.4f}" for idx in top_mass.indices.cpu().tolist()
                    )
                )
                if self.volume_stats is not None:
                    exp_mean, _ = get_expected_volumes(
                        float(age[0].item()),
                        self.volume_stats,
                        num_classes,
                        std_floor=self.volume_std_floor,
                    )
                    exp = torch.tensor(exp_mean, device=vf.device, dtype=vf.dtype)
                    diff = vf - exp
                    top_diff = torch.topk(diff.abs(), k=min(5, diff.numel()))
                    print(
                        f"{debug_prefix} volume diff vs prior: "
                        + ", ".join(
                            f"c{idx}={diff[idx].item():+.4f}" for idx in top_diff.indices.cpu().tolist()
                        )
                    )

        if self.lambda_volume > 0 and self.volume_stats is not None:
            mask = None
            if self.volume_valid_mask.numel() == num_classes:
                mask = self.volume_valid_mask.to(device=probs.device, dtype=probs.dtype)
            loss_volume = volume_consistency_loss(
                probs,
                age,
                self.volume_stats,
                num_classes,
                valid_mask=mask,
                std_floor=self.volume_std_floor,
            )
            loss_dict['volume_loss'] = loss_volume.detach()
            age_loss_sum = age_loss_sum + age_warmup * self.lambda_volume * loss_volume
            if debug_active:
                scaled = age_warmup * self.lambda_volume * loss_volume.detach()
                print(
                    f"{debug_prefix} volume_loss raw={float(loss_volume.detach().item()):.6f}, "
                    f"scaled={float(scaled.item()):.6f} (λ={self.lambda_volume:.3f}, warmup={age_warmup:.3f})"
                )

        if self.lambda_shape > 0 and self.shape_templates is not None:
            loss_shape = shape_consistency_loss(
                probs,
                age,
                self.shape_templates,
                temperature=self.shape_temperature,
                normalize_templates=self.normalize_shape_templates,
            )
            loss_dict['shape_loss'] = loss_shape.detach()
            age_loss_sum = age_loss_sum + age_warmup * self.lambda_shape * loss_shape
            if debug_active:
                scaled = age_warmup * self.lambda_shape * loss_shape.detach()
                print(
                    f"{debug_prefix} shape_loss raw={float(loss_shape.detach().item()):.6f}, "
                    f"scaled={float(scaled.item()):.6f} (λ={self.lambda_shape:.3f}, warmup={age_warmup:.3f})"
                )

        if self.lambda_weighted_adj > 0:
            age_weight_dict = self.age_weights if self.age_weights is not None else None
            A_pred_batch = compute_weighted_adjacency(
                probs,
                age,
                age_weights=age_weight_dict,
                temperature=self.prior_temperature,
            )
            if self.A_weighted_prior.numel() > 0:
                B, C, _ = A_pred_batch.shape
                loss_adj = torch.zeros(1, device=device, dtype=dtype)
                valid_batch_count = 0
                A_prior = self.A_weighted_prior.to(device=device, dtype=dtype)
                volume_fractions_local = volume_fractions.to(device=device, dtype=dtype)
                if self.weighted_adj_presence_slope > 0:
                    presence_logits = (
                            (volume_fractions_local - self.weighted_adj_presence_threshold)
                            * self.weighted_adj_presence_slope
                    )
                    presence = torch.sigmoid(presence_logits)
                else:
                    presence = (volume_fractions_local > self.weighted_adj_presence_threshold).to(dtype=dtype)
                eye = torch.eye(C, device=device, dtype=dtype)
                active_counts: List[int] = []
                for b in range(B):
                    presence_b = presence[b]
                    active_classes = int((presence_b.detach() > 0.5).sum().item())
                    active_counts.append(active_classes)
                    if active_classes <= 1:
                        continue
                    mask = torch.outer(presence_b, presence_b)
                    mask = mask * (1 - eye)
                    if mask.detach().sum() <= 0:
                        continue
                    if self.age_weights is not None:
                        W = get_age_weight_matrix(age[b].item(), self.age_weights, C, device)
                        W = W.to(device=device, dtype=dtype)
                    else:
                        W = torch.ones(C, C, device=device, dtype=dtype)
                    vol_weight = torch.outer(volume_fractions_local[b], volume_fractions_local[b])
                    weight_mask = W * mask * vol_weight
                    denom = weight_mask.sum().clamp(min=1e-6)
                    diff = (A_pred_batch[b] - A_prior) * mask
                    loss_adj = loss_adj + (weight_mask * diff * diff).sum() / denom
                    valid_batch_count += 1
                if valid_batch_count > 0:
                    loss_adj = loss_adj / valid_batch_count
                if active_counts:
                    avg_active = sum(active_counts) / len(active_counts)
                    loss_dict['weighted_adj_active_classes'] = torch.tensor(
                        avg_active,
                        device=device,
                        dtype=dtype,
                    ).detach()
            else:
                loss_adj = torch.zeros(1, device=device, dtype=dtype)

            loss_dict['weighted_adj_loss'] = loss_adj.detach()
            age_loss_sum = age_loss_sum + age_warmup * self.lambda_weighted_adj * loss_adj
            if debug_active:
                scaled = age_warmup * self.lambda_weighted_adj * loss_adj.detach()
                print(
                    f"{debug_prefix} weighted_adj raw={float(loss_adj.detach().item()):.6f}, "
                    f"scaled={float(scaled.item()):.6f} (λ={self.lambda_weighted_adj:.3f}, warmup={age_warmup:.3f})"
                )
                if 'weighted_adj_active_classes' in loss_dict:
                    print(
                        f"{debug_prefix} active classes={float(loss_dict['weighted_adj_active_classes'].item()):.2f}"
                    )

        if self.lambda_topo > 0 and (self.required_edges or self.forbidden_edges):
            if self.use_restricted_mask and self.R_mask.numel() > 0:
                topo_mask = self.R_mask.to(device=device, dtype=dtype)
            else:
                topo_mask = None
            A_simple = soft_adjacency_from_probs(
                probs,
                kernel_size=1,
                stride=1,
                temperature=1.0,
                restricted_mask=topo_mask,
            )
            loss_topo = torch.zeros(1, device=device, dtype=dtype)
            for i, j in self.required_edges:
                if i < num_classes and j < num_classes:
                    loss_topo = loss_topo + F.relu(0.01 - A_simple[i, j])
            for i, j in self.forbidden_edges:
                if i < num_classes and j < num_classes:
                    loss_topo = loss_topo + F.relu(A_simple[i, j] - 0.001)

            loss_dict['topo_loss'] = loss_topo.detach()
            age_loss_sum = age_loss_sum + age_warmup * self.lambda_topo * loss_topo
            if debug_active:
                print(f"{debug_prefix} topo loss={float(loss_topo.detach().item()):.6f}")

        # ========================= Graph alignment branch =========================
        graph_branch_sum = torch.zeros(1, device=device, dtype=dtype)
        loss_spec_src = torch.zeros(1, device=device, dtype=dtype)
        loss_edge_src = torch.zeros(1, device=device, dtype=dtype)
        loss_spec_tgt = torch.zeros(1, device=device, dtype=dtype)
        loss_edge_tgt = torch.zeros(1, device=device, dtype=dtype)
        loss_sym = torch.zeros(1, device=device, dtype=dtype)

        need_graph = (
            self.lambda_spec_src > 0 or self.lambda_edge_src > 0 or
            self.lambda_spec_tgt > 0 or self.lambda_edge_tgt > 0 or
            (self.lambda_sym > 0 and len(self.lr_pairs) > 0)
        )

        A_pred = None
        if need_graph:
            if self.use_restricted_mask and self.R_mask.numel() > 0:
                restricted_mask = self.R_mask.to(device=device, dtype=dtype)
            else:
                restricted_mask = None
            A_pred = soft_adjacency_from_probs(
                probs,
                kernel_size=self.pool_kernel,
                stride=self.pool_stride,
                temperature=self.temperature,
                restricted_mask=restricted_mask,
            )
            L_pred = compute_laplacian(A_pred, normalized=True)

            if self.has_source_prior and self.lambda_spec_src > 0:
                loss_spec_src = spectral_alignment_loss(
                    L_pred, self.L_src,
                    top_k=self.top_k,
                    align_vectors=True,
                    eigenvalue_weighted=self.align_U_weighted,
                )

            if self.has_source_prior and self.lambda_edge_src > 0:
                A_src = self.A_src
                if self.use_restricted_mask and restricted_mask is not None:
                    A_src = A_src * restricted_mask
                loss_edge_src = edge_consistency_loss_with_mismatch(
                    A_pred,
                    A_src,
                    required_edges=self.src_required_edges,
                    forbidden_edges=self.src_forbidden_edges,
                    class_weights=self.class_weights,
                    qap_mismatch_g=self.qap_mismatch_g,
                    restricted_mask=restricted_mask,
                )

            if self.has_target_prior and self.lambda_spec_tgt > 0:
                loss_spec_tgt = spectral_alignment_loss(
                    L_pred, self.L_tgt,
                    top_k=self.top_k,
                    align_vectors=True,
                    eigenvalue_weighted=self.align_U_weighted,
                )

            if self.has_target_prior and self.lambda_edge_tgt > 0:
                A_tgt = self.A_tgt
                if self.use_restricted_mask and restricted_mask is not None:
                    A_tgt = A_tgt * restricted_mask
                loss_edge_tgt = edge_consistency_loss_with_mismatch(
                    A_pred,
                    A_tgt,
                    required_edges=self.tgt_required_edges,
                    forbidden_edges=self.tgt_forbidden_edges,
                    class_weights=self.class_weights,
                    qap_mismatch_g=self.qap_mismatch_g,
                    restricted_mask=restricted_mask,
                )

            if self.lambda_sym > 0 and self.lr_pairs:
                loss_sym = symmetry_consistency_loss(probs, self.lr_pairs)

            graph_branch_sum = (
                self.lambda_spec_src * loss_spec_src +
                self.lambda_edge_src * loss_edge_src +
                self.lambda_spec_tgt * loss_spec_tgt +
                self.lambda_edge_tgt * loss_edge_tgt +
                self.lambda_sym * loss_sym
            )
            if debug_active:
                if A_pred is not None:
                    with torch.no_grad():
                        print(
                            f"{debug_prefix} A_pred stats mean={A_pred.mean().item():.5f}, max={A_pred.max().item():.5f}, "
                            f"min={A_pred.min().item():.5f}"
                        )
                print(
                    f"{debug_prefix} graph losses spec_src={float(loss_spec_src.detach().item()):.6f}, "
                    f"edge_src={float(loss_edge_src.detach().item()):.6f}, "
                    f"spec_tgt={float(loss_spec_tgt.detach().item()):.6f}, "
                    f"edge_tgt={float(loss_edge_tgt.detach().item()):.6f}, sym={float(loss_sym.detach().item()):.6f}"
                )

        warmup_factor = self.get_warmup_factor()
        graph_total = warmup_factor * graph_branch_sum
        if debug_active:
            print(f"{debug_prefix} graph_total={float(graph_total.detach().item()):.6f} (warmup={warmup_factor:.3f})")

        total_loss = age_loss_sum + graph_total

        loss_dict['graph_total'] = graph_total.detach()
        loss_dict['graph_loss'] = graph_total.detach()
        loss_dict['graph_spec_src'] = loss_spec_src.detach()
        loss_dict['graph_edge_src'] = loss_edge_src.detach()
        loss_dict['graph_spec_tgt'] = loss_spec_tgt.detach()
        loss_dict['graph_edge_tgt'] = loss_edge_tgt.detach()
        loss_dict['graph_sym'] = loss_sym.detach()
        loss_dict['sym_loss'] = loss_sym.detach()
        loss_dict['graph_spec'] = (
            (loss_spec_src * self.lambda_spec_src + loss_spec_tgt * self.lambda_spec_tgt)
            / max(self.lambda_spec_src + self.lambda_spec_tgt, 1e-8)
            if (self.lambda_spec_src + self.lambda_spec_tgt) > 0 else torch.zeros_like(loss_spec_src)
        ).detach()
        loss_dict['graph_edge'] = (
            (loss_edge_src * self.lambda_edge_src + loss_edge_tgt * self.lambda_edge_tgt)
            / max(self.lambda_edge_src + self.lambda_edge_tgt, 1e-8)
            if (self.lambda_edge_src + self.lambda_edge_tgt) > 0 else torch.zeros_like(loss_edge_src)
        ).detach()
        loss_dict['warmup_factor'] = torch.tensor(warmup_factor, device=device)
        loss_dict['age_warmup_factor'] = torch.tensor(age_warmup, device=device)

        if A_pred is not None:
            th_required = 0.02
            th_forbidden = 5e-4
            required_missing = 0
            forbidden_present = 0

            for i, j in (self.tgt_required_edges + self.src_required_edges):
                if i < A_pred.shape[0] and j < A_pred.shape[1] and A_pred[i, j].item() < th_required:
                    required_missing += 1

            for i, j in (self.tgt_forbidden_edges + self.src_forbidden_edges):
                if i < A_pred.shape[0] and j < A_pred.shape[1] and A_pred[i, j].item() > th_forbidden:
                    forbidden_present += 1

            loss_dict['structural_violations'] = {
                'required_missing': required_missing,
                'forbidden_present': forbidden_present,
            }
            struct_score = forbidden_present * 1.5 + required_missing
            loss_dict['graph_struct'] = torch.tensor(struct_score, device=device, dtype=dtype)
            loss_dict['A_pred'] = A_pred.detach()
        else:
            loss_dict['structural_violations'] = {
                'required_missing': 0,
                'forbidden_present': 0,
            }
            loss_dict['graph_struct'] = torch.zeros(1, device=device, dtype=dtype)

        if debug_active:
            self._debug_batch_count += 1

        return total_loss, loss_dict


def symmetry_consistency_loss(probs: torch.Tensor,
                              lr_pairs: List[Tuple[int, int]],
                              flip_dim: int = 2) -> torch.Tensor:
    """
    Compute symmetry consistency loss for left-right paired structures

    Args:
        probs: (B, C, X, Y, Z) probability maps
        lr_pairs: List of (left, right) class index pairs (0-based)
        flip_dim: Spatial dimension for left-right flip (2 for X-axis in RAS)

    Returns:
        loss: Scalar loss value
    """
    if not lr_pairs:
        return torch.tensor(0.0, device=probs.device)

    # Flip probabilities along left-right axis
    probs_flipped = torch.flip(probs, dims=[flip_dim])

    # Create swapped version
    probs_swapped = probs_flipped.clone()

    # Swap left-right paired channels
    for left, right in lr_pairs:
        probs_swapped[:, left, ...] = probs_flipped[:, right, ...]
        probs_swapped[:, right, ...] = probs_flipped[:, left, ...]

    # Consistency loss: original should match swapped version
    loss = F.l1_loss(probs, probs_swapped)

    return loss