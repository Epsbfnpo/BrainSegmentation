#!/usr/bin/env python3
"""
Age-conditioned graph-based anatomical prior loss for brain segmentation
Implements volume priors, shape priors, and weighted adjacency based on age
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
from typing import Optional, Dict, List, Tuple
from monai.data import MetaTensor
from scipy.ndimage import distance_transform_edt


def _to_tensor(x):
    """Convert MetaTensor to regular torch.Tensor if needed"""
    if x is None:
        return None
    if isinstance(x, MetaTensor):
        return x.as_tensor()
    return x


def compute_sdt(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute signed distance transform for a mask
    Args:
        mask: (B, C, X, Y, Z) binary or soft mask
    Returns:
        sdt: (B, C, X, Y, Z) signed distance transform
    """
    B, C, X, Y, Z = mask.shape
    sdt = torch.zeros_like(mask)

    # Convert to numpy for distance transform
    mask_np = mask.detach().cpu().numpy()

    for b in range(B):
        for c in range(C):
            # Threshold soft mask
            binary_mask = mask_np[b, c] > 0.5

            if binary_mask.any():
                # Compute distance transform
                dist_inside = distance_transform_edt(binary_mask)
                dist_outside = distance_transform_edt(~binary_mask)

                # Signed distance: negative inside, positive outside
                sdt_np = np.where(binary_mask, -dist_inside, dist_outside)
                sdt[b, c] = torch.from_numpy(sdt_np).to(mask.device)

    return sdt


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
        A_weighted: (C, C) weighted adjacency matrix
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

    # Average over batch
    A_weighted = A_batch.mean(dim=0)

    # Row normalize
    row_sums = A_weighted.sum(dim=1, keepdim=True).clamp(min=1e-8)
    A_weighted = A_weighted / row_sums

    return A_weighted


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


def volume_consistency_loss(probs: torch.Tensor, age: torch.Tensor,
                            volume_stats: Dict, num_classes: int) -> torch.Tensor:
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

    # Get expected volumes for each age
    expected_volumes = torch.zeros(B, num_classes, device=probs.device)
    expected_stds = torch.ones(B, num_classes, device=probs.device)

    for b in range(B):
        age_val = age[b].item()
        volumes, stds = get_expected_volumes(age_val, volume_stats, num_classes)
        expected_volumes[b] = torch.tensor(volumes, device=probs.device)
        expected_stds[b] = torch.tensor(stds, device=probs.device)

    # Normalize by expected std
    normalized_diff = (predicted_volumes - expected_volumes) / (expected_stds + 1e-6)

    # Huber loss for robustness
    loss = F.smooth_l1_loss(normalized_diff, torch.zeros_like(normalized_diff))

    return loss


def get_expected_volumes(age: float, volume_stats: Dict,
                         num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
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
    if not volume_stats:
        return np.ones(num_classes), np.ones(num_classes)

    # Find closest age or interpolate
    ages = sorted(list(volume_stats.keys()))

    if age <= ages[0]:
        stats = volume_stats[ages[0]]
        return np.array(stats['means']), np.array(stats['stds'])

    if age >= ages[-1]:
        stats = volume_stats[ages[-1]]
        return np.array(stats['means']), np.array(stats['stds'])

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
            return means, stds

    return np.ones(num_classes), np.ones(num_classes)


def shape_consistency_loss(probs: torch.Tensor, age: torch.Tensor,
                           shape_templates: Optional[Dict] = None) -> torch.Tensor:
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

    # Compute SDT for predictions
    sdt_pred = compute_sdt(probs)

    # Get expected SDT for each age
    B = probs.shape[0]
    loss = 0.0

    for b in range(B):
        age_val = age[b].item()
        sdt_expected = get_shape_template(age_val, shape_templates, probs.shape[1:], probs.device)

        if sdt_expected is not None:
            # L1 loss on SDT
            loss += F.l1_loss(sdt_pred[b], sdt_expected)

    return loss / B if B > 0 else torch.tensor(0.0, device=probs.device)


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

    # Ensure correct shape
    if template.shape != shape:
        # Simple resize using interpolation
        template = F.interpolate(
            template.unsqueeze(0),
            size=shape[1:],
            mode='trilinear',
            align_corners=False
        ).squeeze(0)

    return template


class AgeConditionedGraphPriorLoss(nn.Module):
    """
    Age-conditioned graph prior loss with volume, shape, and weighted adjacency
    """

    def __init__(self,
                 # Volume statistics paths
                 volume_stats_path: Optional[str] = None,

                 # Shape template paths
                 shape_templates_path: Optional[str] = None,

                 # Weighted adjacency paths
                 weighted_adj_path: Optional[str] = None,
                 age_weights_path: Optional[str] = None,

                 # Legacy topology constraints (kept as weak regularization)
                 prior_adj_path: Optional[str] = None,
                 required_json: Optional[str] = None,
                 forbidden_json: Optional[str] = None,
                 lr_pairs_json: Optional[str] = None,

                 # Loss weights
                 lambda_volume: float = 0.2,
                 lambda_shape: float = 0.1,
                 lambda_weighted_adj: float = 0.15,
                 lambda_topo: float = 0.02,  # Weak weight for topology
                 lambda_sym: float = 0.05,

                 # Other parameters
                 temperature: float = 1.0,
                 warmup_epochs: int = 10):
        super().__init__()

        self.lambda_volume = lambda_volume
        self.lambda_shape = lambda_shape
        self.lambda_weighted_adj = lambda_weighted_adj
        self.lambda_topo = lambda_topo
        self.lambda_sym = lambda_sym
        self.temperature = temperature
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # Load volume statistics
        self.volume_stats = None
        if volume_stats_path and os.path.exists(volume_stats_path):
            with open(volume_stats_path, 'r') as f:
                self.volume_stats = json.load(f)
            print(f"✓ Loaded volume statistics from {volume_stats_path}")

        # Load shape templates
        self.shape_templates = None
        if shape_templates_path and os.path.exists(shape_templates_path):
            self.shape_templates = torch.load(shape_templates_path)
            print(f"✓ Loaded shape templates from {shape_templates_path}")

        # Load age-dependent weights
        self.age_weights = None
        if age_weights_path and os.path.exists(age_weights_path):
            with open(age_weights_path, 'r') as f:
                self.age_weights = json.load(f)
            print(f"✓ Loaded age-dependent weights from {age_weights_path}")

        # Load weighted adjacency prior
        if weighted_adj_path and os.path.exists(weighted_adj_path):
            A_weighted = np.load(weighted_adj_path)
            self.register_buffer('A_weighted_prior', torch.from_numpy(A_weighted).float())
            print(f"✓ Loaded weighted adjacency prior from {weighted_adj_path}")
        else:
            self.A_weighted_prior = None

        # Load legacy topology constraints (weak regularization)
        self.required_edges = []
        if required_json and os.path.exists(required_json):
            with open(required_json, 'r') as f:
                data = json.load(f)
                self.required_edges = [(int(i), int(j)) for i, j in data['required']]

        self.forbidden_edges = []
        if forbidden_json and os.path.exists(forbidden_json):
            with open(forbidden_json, 'r') as f:
                data = json.load(f)
                self.forbidden_edges = [(int(i), int(j)) for i, j in data['forbidden']]

        # Load laterality pairs
        self.lr_pairs = []
        if lr_pairs_json and os.path.exists(lr_pairs_json):
            with open(lr_pairs_json, 'r') as f:
                pairs_raw = json.load(f)
                self.lr_pairs = [(int(a) - 1, int(b) - 1) for a, b in pairs_raw if int(a) > 0 and int(b) > 0]

    def set_epoch(self, epoch: int):
        """Update current epoch for warmup scheduling"""
        self.current_epoch = epoch

    def get_warmup_factor(self) -> float:
        """Get warmup factor for current epoch"""
        if self.current_epoch < self.warmup_epochs:
            return self.current_epoch / max(1, self.warmup_epochs)
        return 1.0

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                age: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute age-conditioned graph prior loss

        Args:
            logits: (B, C, X, Y, Z) model output logits
            labels: (B, X, Y, Z) ground truth labels
            age: (B, 1) age tensor

        Returns:
            total_loss: Weighted sum of all loss components
            loss_dict: Dictionary of individual loss components
        """
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]

        # Initialize losses
        loss_dict = {}
        total_loss = 0.0

        # 1. Volume consistency loss
        if self.lambda_volume > 0 and self.volume_stats is not None:
            loss_volume = volume_consistency_loss(probs, age, self.volume_stats, num_classes)
            loss_dict['volume_loss'] = loss_volume.detach()
            total_loss = total_loss + self.lambda_volume * loss_volume

        # 2. Shape consistency loss
        if self.lambda_shape > 0 and self.shape_templates is not None:
            loss_shape = shape_consistency_loss(probs, age, self.shape_templates)
            loss_dict['shape_loss'] = loss_shape.detach()
            total_loss = total_loss + self.lambda_shape * loss_shape

        # 3. Weighted adjacency loss
        if self.lambda_weighted_adj > 0:
            A_pred = compute_weighted_adjacency(probs, age, self.age_weights, self.temperature)

            if self.A_weighted_prior is not None:
                # Compare with prior
                loss_adj = F.mse_loss(A_pred, self.A_weighted_prior)
            else:
                # Self-consistency: ensure smooth adjacency
                loss_adj = torch.var(A_pred)

            loss_dict['weighted_adj_loss'] = loss_adj.detach()
            total_loss = total_loss + self.lambda_weighted_adj * loss_adj

        # 4. Weak topology regularization (legacy)
        if self.lambda_topo > 0 and (self.required_edges or self.forbidden_edges):
            loss_topo = 0.0

            # Simple adjacency from probabilities
            probs_flat = probs.reshape(probs.shape[0], probs.shape[1], -1)
            A_simple = torch.bmm(probs_flat, probs_flat.transpose(1, 2)).mean(0)

            # Required edges
            for i, j in self.required_edges:
                if i < num_classes and j < num_classes:
                    loss_topo += F.relu(0.01 - A_simple[i, j])

            # Forbidden edges
            for i, j in self.forbidden_edges:
                if i < num_classes and j < num_classes:
                    loss_topo += F.relu(A_simple[i, j] - 0.001)

            if isinstance(loss_topo, float):
                loss_topo = torch.tensor(loss_topo, device=logits.device)

            loss_dict['topo_loss'] = loss_topo.detach()
            total_loss = total_loss + self.lambda_topo * loss_topo

        # 5. Symmetry loss
        if self.lambda_sym > 0 and self.lr_pairs:
            loss_sym = symmetry_consistency_loss(probs, self.lr_pairs)
            loss_dict['sym_loss'] = loss_sym.detach()
            total_loss = total_loss + self.lambda_sym * loss_sym

        # Apply warmup
        warmup_factor = self.get_warmup_factor()
        total_loss = total_loss * warmup_factor

        loss_dict['graph_total'] = total_loss.detach()
        loss_dict['warmup_factor'] = warmup_factor

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