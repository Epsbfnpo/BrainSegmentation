"""
Graph-based anatomical prior loss for brain segmentation with DUAL-BRANCH CROSS-DOMAIN ALIGNMENT
Implements spectral alignment, edge consistency, and symmetry constraints
Enhanced with weighted U-subspace alignment, mismatch-aware penalties, and restricted masks
FIXED: Structural violations tracking and eigenvector sign alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
from typing import Optional, Dict, List, Tuple
from monai.data import MetaTensor


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
    """
    Compute soft adjacency matrix from probability maps with optional restrictions

    Args:
        probs: (B, C, X, Y, Z) softmax probability maps
        kernel_size: Size of pooling kernel for efficiency
        stride: Stride for pooling
        temperature: Temperature for sharpening adjacencies
        restricted_mask: (C, C) mask to restrict valid adjacencies

    Returns:
        A: (C, C) soft adjacency matrix averaged over batch, row-normalized
    """
    # Ensure we're working with pure tensor
    probs = _to_tensor(probs)

    B, C, X, Y, Z = probs.shape

    # Apply average pooling for computational efficiency
    if kernel_size > 1:
        probs_pooled = F.avg_pool3d(
            probs,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            count_include_pad=False
        )
    else:
        probs_pooled = probs

    # Reshape to (B, C, N) where N is number of voxels
    B, C, X_p, Y_p, Z_p = probs_pooled.shape
    probs_flat = probs_pooled.reshape(B, C, -1)

    # Compute pairwise co-occurrence as proxy for adjacency
    # Normalize probabilities per channel
    probs_norm = probs_flat / (probs_flat.sum(dim=2, keepdim=True) + 1e-8)

    # Compute correlation matrix for each batch
    A_batch = torch.bmm(probs_norm, probs_norm.transpose(1, 2))  # (B, C, C)

    # Apply temperature
    if temperature != 1.0:
        A_batch = torch.pow(A_batch + 1e-8, 1.0 / temperature)

    # Zero out diagonal (no self-adjacency)
    eye = torch.eye(C, device=probs.device).unsqueeze(0)
    A_batch = A_batch * (1 - eye)

    # Apply restricted mask if provided
    if restricted_mask is not None:
        restricted_mask = _to_tensor(restricted_mask)
        if restricted_mask.dim() == 2:
            restricted_mask = restricted_mask.unsqueeze(0)  # Add batch dimension
        A_batch = A_batch * restricted_mask

    # Row-normalize to get transition probabilities
    # ENHANCED: Add extra numerical stability for small batches/rare classes
    row_sums = A_batch.sum(dim=2, keepdim=True).clamp(min=1e-8)
    A_batch = A_batch / row_sums

    # Average over batch
    A = A_batch.mean(dim=0)

    return A


def compute_laplacian(A: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """
    Compute graph Laplacian from adjacency matrix

    Args:
        A: (C, C) adjacency matrix (should be row-normalized)
        normalized: Whether to compute normalized Laplacian

    Returns:
        L: (C, C) Laplacian matrix
    """
    # Ensure pure tensor
    A = _to_tensor(A)

    # Ensure symmetry for Laplacian (average with transpose)
    A_sym = 0.5 * (A + A.T)

    # Degree matrix
    D = torch.diag(A_sym.sum(dim=1))

    # Laplacian
    L = D - A_sym

    if normalized:
        # Normalized Laplacian: L_norm = D^(-1/2) L D^(-1/2)
        d_sqrt_inv = torch.diag(1.0 / torch.sqrt(torch.diag(D).clamp(min=1e-8)))
        L = d_sqrt_inv @ L @ d_sqrt_inv

    return L


def spectral_alignment_loss(L_pred: torch.Tensor,
                            L_prior: torch.Tensor,
                            top_k: int = 20,
                            align_vectors: bool = True,
                            eigenvalue_weighted: bool = False) -> torch.Tensor:
    """
    Compute spectral alignment loss between two Laplacians with optional eigenvalue weighting

    Args:
        L_pred: (C, C) predicted Laplacian
        L_prior: (C, C) prior Laplacian
        top_k: Number of eigenvalues/vectors to align
        align_vectors: Whether to align eigenvectors as well
        eigenvalue_weighted: Whether to use eigenvalue-weighted U-subspace alignment

    Returns:
        loss: Scalar loss value
    """
    # Ensure pure tensors
    L_pred = _to_tensor(L_pred)
    L_prior = _to_tensor(L_prior)

    # Make matrices symmetric (required for eigh)
    L_pred_sym = 0.5 * (L_pred + L_pred.T)
    L_prior_sym = 0.5 * (L_prior + L_prior.T)

    # Convert to float32 for eigendecomposition (eigh doesn't support bfloat16)
    L_pred_sym_f32 = L_pred_sym.float()
    L_prior_sym_f32 = L_prior_sym.float()

    # Compute eigendecomposition (eigenvalues in ascending order)
    evals_pred, evecs_pred = torch.linalg.eigh(L_pred_sym_f32)
    evals_prior, evecs_prior = torch.linalg.eigh(L_prior_sym_f32)

    # Convert back to original dtype for loss computation
    evals_pred = evals_pred.to(L_pred.dtype)
    evals_prior = evals_prior.to(L_prior.dtype)
    evecs_pred = evecs_pred.to(L_pred.dtype)
    evecs_prior = evecs_prior.to(L_prior.dtype)

    # IMPORTANT: Ensure these are pure tensors before slicing
    evals_pred = _to_tensor(evals_pred)
    evals_prior = _to_tensor(evals_prior)
    evecs_pred = _to_tensor(evecs_pred)
    evecs_prior = _to_tensor(evecs_prior)

    # Select top-k smallest eigenvalues (excluding the trivial zero)
    k = min(top_k, evals_pred.shape[0] - 1)

    # Align eigenvalues (skip first which is near zero)
    loss_evals = F.mse_loss(evals_pred[1:k + 1], evals_prior[1:k + 1])

    if align_vectors:
        # Align eigenvector subspaces
        U_pred = evecs_pred[:, 1:k + 1]  # (C, k)
        U_prior = evecs_prior[:, 1:k + 1]  # (C, k)

        # FIXED: Handle sign ambiguity without in-place operations
        # Compute dot products for all eigenvectors at once
        dots = (U_pred * U_prior).sum(dim=0)  # (k,)
        # Create sign vector: -1 where dot < 0, +1 otherwise
        signs = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
        # Apply signs without in-place modification
        U_pred = U_pred * signs.unsqueeze(0)  # (C, k) * (1, k) -> (C, k)

        if eigenvalue_weighted:
            # NEW: Use eigenvalue-weighted alignment (LRA-style)
            # Weight by absolute eigenvalues or spectral gap
            w = evals_prior[1:k + 1].abs()
            w = w / (w.sum() + 1e-8)

            # Compute weighted subspace alignment
            loss_subspace = 0
            for i in range(k):
                # Weighted cosine similarity loss
                cos_sim = torch.dot(U_pred[:, i], U_prior[:, i])
                loss_subspace += w[i] * (1.0 - cos_sim ** 2)
        else:
            # Original Procrustes-like approach
            # Projection matrices
            P_pred = U_pred @ U_pred.T
            P_prior = U_prior @ U_prior.T

            # Subspace alignment loss
            loss_subspace = F.mse_loss(P_pred, P_prior)
    else:
        loss_subspace = torch.tensor(0.0, device=L_pred.device)

    return loss_evals + 0.5 * loss_subspace  # Weight subspace less


def edge_consistency_loss_with_mismatch(A_pred: torch.Tensor,
                                        A_prior: torch.Tensor,
                                        required_edges: Optional[List[Tuple[int, int]]] = None,
                                        forbidden_edges: Optional[List[Tuple[int, int]]] = None,
                                        margin: float = 0.1,
                                        class_weights: Optional[torch.Tensor] = None,
                                        qap_mismatch_g: float = 1.5,
                                        restricted_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute edge-level consistency loss with mismatch-aware penalties

    Args:
        A_pred: (C, C) predicted adjacency (row-normalized)
        A_prior: (C, C) prior adjacency (row-normalized)
        required_edges: List of edges that must exist
        forbidden_edges: List of edges that must not exist
        margin: Margin for required/forbidden constraints
        class_weights: (C,) tensor of per-class weights for reweighting
        qap_mismatch_g: Mismatch penalty factor for QAP-like loss
        restricted_mask: (C, C) mask for valid adjacencies

    Returns:
        loss: Scalar loss value
    """
    # Ensure pure tensors
    A_pred = _to_tensor(A_pred)
    A_prior = _to_tensor(A_prior)

    # Apply restricted mask if provided
    if restricted_mask is not None:
        restricted_mask = _to_tensor(restricted_mask)
        A_pred = A_pred * restricted_mask
        A_prior = A_prior * restricted_mask

    # Compute edge weights from class weights if provided
    if class_weights is not None:
        # Create edge weight matrix W[i,j] = sqrt(w_i * w_j)
        w = class_weights.view(-1, 1)
        W = torch.sqrt(w @ w.T)
        # Normalize to have mean 1 to preserve scale
        W = W / W.mean()
    else:
        W = torch.ones_like(A_pred)

    # Weighted MSE for adjacency matching
    loss_mse = torch.mean(W * torch.square(A_pred - A_prior))

    # NEW: QAP-like mismatch-aware loss
    # Compute match, neutral, and mismatch scores
    if restricted_mask is not None:
        R = restricted_mask
    else:
        R = torch.ones_like(A_pred)

    # Soft match/mismatch computation
    M = (A_pred * A_prior) * R  # matches (both high)
    N = ((1 - A_pred) * (1 - A_prior)) * R  # neutral (both low)
    X = ((1 - A_prior) * A_pred + A_prior * (1 - A_pred)) * R  # mismatches

    # QAP-style loss: penalize mismatches more than rewarding matches
    loss_qap = qap_mismatch_g * X.mean() - M.mean()

    # Combine MSE and QAP losses
    loss_base = loss_mse + 0.1 * loss_qap  # Beta factor for QAP

    # Required edge constraints (stricter with adjusted threshold)
    loss_required = torch.tensor(0.0, device=A_pred.device)
    th_required = 0.02  # Slightly stricter threshold
    if required_edges:
        for i, j in required_edges:
            # Penalize if edge probability is too low
            loss_required = loss_required + torch.pow(F.relu(th_required - A_pred[i, j]), 2)

    # Forbidden edge constraints (stricter with adjusted threshold)
    loss_forbidden = torch.tensor(0.0, device=A_pred.device)
    th_forbidden = 5e-4  # Slightly stricter threshold
    if forbidden_edges:
        for i, j in forbidden_edges:
            # Penalize if edge probability is too high
            loss_forbidden = loss_forbidden + torch.pow(F.relu(A_pred[i, j] - th_forbidden), 2)

    # Weight the constraint losses appropriately
    num_constraints = len(required_edges or []) + len(forbidden_edges or [])
    if num_constraints > 0:
        constraint_weight = 0.1
        loss_constraints = constraint_weight * (loss_required + loss_forbidden) / num_constraints
    else:
        loss_constraints = torch.tensor(0.0, device=A_pred.device)

    return loss_base + loss_constraints


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

    # Ensure pure tensor
    probs = _to_tensor(probs)

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


def compute_restricted_mask(num_classes: int,
                            required_edges: List[Tuple[int, int]],
                            forbidden_edges: List[Tuple[int, int]],
                            lr_pairs: List[Tuple[int, int]],
                            device: torch.device) -> torch.Tensor:
    """
    Compute restricted mask R for valid adjacencies

    Args:
        num_classes: Number of classes
        required_edges: List of required edges
        forbidden_edges: List of forbidden edges
        lr_pairs: List of laterality pairs
        device: Device to create tensor on

    Returns:
        R: (C, C) binary mask where 1 indicates valid adjacency, 0 indicates restricted
    """
    # Start with all adjacencies allowed
    R = torch.ones(num_classes, num_classes, device=device)

    # Remove diagonal (no self-adjacency)
    R.fill_diagonal_(0)

    # Set forbidden edges to 0
    for i, j in forbidden_edges:
        if i < num_classes and j < num_classes:
            R[i, j] = 0
            R[j, i] = 0  # Ensure symmetry

    # Ensure required edges are 1 (in case of conflict, required overrides forbidden)
    for i, j in required_edges:
        if i < num_classes and j < num_classes:
            R[i, j] = 1
            R[j, i] = 1  # Ensure symmetry

    # Ensure laterality pairs can connect
    for left, right in lr_pairs:
        if left < num_classes and right < num_classes:
            R[left, right] = 1
            R[right, left] = 1

    return R


class GraphPriorLoss(nn.Module):
    """
    Combined graph-based anatomical prior loss with DUAL-BRANCH CROSS-DOMAIN ALIGNMENT
    Prior branch: stable structural anchoring with class-level priors
    Dynamic branch: flexible spectral alignment (handled in trainer)
    """

    def __init__(self,
                 # Target domain priors (existing)
                 prior_adj_path: str,
                 required_json: Optional[str] = None,
                 forbidden_json: Optional[str] = None,
                 lr_pairs_json: Optional[str] = None,
                 # Source domain priors
                 src_prior_adj_path: Optional[str] = None,
                 src_required_json: Optional[str] = None,
                 src_forbidden_json: Optional[str] = None,
                 # Loss weights
                 lambda_spec: float = 0.1,
                 lambda_edge: float = 0.1,
                 lambda_sym: float = 0.05,
                 # Separate weights for source and target alignment
                 lambda_spec_src: Optional[float] = None,
                 lambda_edge_src: Optional[float] = None,
                 lambda_spec_tgt: Optional[float] = None,
                 lambda_edge_tgt: Optional[float] = None,
                 # Graph parameters
                 top_k: int = 20,
                 temperature: float = 1.0,
                 warmup_epochs: int = 10,
                 pool_kernel: int = 3,
                 pool_stride: int = 2,
                 # Alignment mode
                 graph_align_mode: str = 'joint',
                 # Class weights for edge reweighting
                 class_weights: Optional[torch.Tensor] = None,
                 # NEW: Enhanced alignment parameters
                 align_U_weighted: bool = False,
                 qap_mismatch_g: float = 1.5,
                 use_restricted_mask: bool = False,
                 restricted_mask_path: Optional[str] = None,
                 # Dynamic alignment parameters (for reference, used in trainer)
                 lambda_dyn: float = 0.2,
                 dyn_top_k: int = 12,
                 dyn_start_epoch: int = 50,
                 dyn_ramp_epochs: int = 50):
        """
        Initialize enhanced graph prior loss with dual-branch alignment

        Args:
            align_U_weighted: Use eigenvalue-weighted U-subspace alignment
            qap_mismatch_g: Mismatch penalty factor for QAP-like loss
            use_restricted_mask: Use restricted mask for conflict resolution
            restricted_mask_path: Path to precomputed R_mask.npy
        """
        super().__init__()

        self.graph_align_mode = graph_align_mode
        self.align_U_weighted = align_U_weighted
        self.qap_mismatch_g = qap_mismatch_g
        self.use_restricted_mask = use_restricted_mask

        # Store dynamic alignment parameters (for trainer reference)
        self.lambda_dyn = lambda_dyn
        self.dyn_top_k = dyn_top_k
        self.dyn_start_epoch = dyn_start_epoch
        self.dyn_ramp_epochs = dyn_ramp_epochs

        # ========== TARGET DOMAIN PRIORS (existing) ==========
        # Load target adjacency (should already be row-normalized)
        A_tgt = torch.from_numpy(np.load(prior_adj_path)).float()
        # Ensure row normalization
        row_sums = A_tgt.sum(dim=1, keepdim=True).clamp(min=1e-8)
        A_tgt = A_tgt / row_sums
        self.register_buffer('A_tgt', A_tgt)

        # Precompute target Laplacian
        L_tgt = compute_laplacian(A_tgt, normalized=True)
        self.register_buffer('L_tgt', L_tgt)

        # Load target edge constraints
        self.tgt_required_edges = []
        if required_json and os.path.exists(required_json):
            with open(required_json, 'r') as f:
                data = json.load(f)
                self.tgt_required_edges = [(int(i), int(j)) for i, j in data['required']]

        self.tgt_forbidden_edges = []
        if forbidden_json and os.path.exists(forbidden_json):
            with open(forbidden_json, 'r') as f:
                data = json.load(f)
                self.tgt_forbidden_edges = [(int(i), int(j)) for i, j in data['forbidden']]

        # ========== SOURCE DOMAIN PRIORS ==========
        self.has_source_prior = (src_prior_adj_path is not None and
                                 os.path.exists(src_prior_adj_path) and
                                 graph_align_mode in ['src_only', 'joint'])

        if self.has_source_prior:
            # Load source adjacency
            A_src = torch.from_numpy(np.load(src_prior_adj_path)).float()
            # Ensure row normalization
            row_sums = A_src.sum(dim=1, keepdim=True).clamp(min=1e-8)
            A_src = A_src / row_sums
            self.register_buffer('A_src', A_src)

            # Precompute source Laplacian
            L_src = compute_laplacian(A_src, normalized=True)
            self.register_buffer('L_src', L_src)

            # Load source edge constraints (optional)
            self.src_required_edges = []
            if src_required_json and os.path.exists(src_required_json):
                with open(src_required_json, 'r') as f:
                    data = json.load(f)
                    self.src_required_edges = [(int(i), int(j)) for i, j in data['required']]

            self.src_forbidden_edges = []
            if src_forbidden_json and os.path.exists(src_forbidden_json):
                with open(src_forbidden_json, 'r') as f:
                    data = json.load(f)
                    self.src_forbidden_edges = [(int(i), int(j)) for i, j in data['forbidden']]
        else:
            self.A_src = None
            self.L_src = None
            self.src_required_edges = []
            self.src_forbidden_edges = []

        # ========== LATERALITY PAIRS ==========
        # Load laterality pairs (already 0-based from data loader)
        self.lr_pairs = []
        if lr_pairs_json and os.path.exists(lr_pairs_json):
            with open(lr_pairs_json, 'r') as f:
                pairs_raw = json.load(f)
                # Adjust to 0-based if needed
                self.lr_pairs = [(int(a) - 1, int(b) - 1) for a, b in pairs_raw if int(a) > 0 and int(b) > 0]

        # ========== RESTRICTED MASK ==========
        if use_restricted_mask:
            if restricted_mask_path and os.path.exists(restricted_mask_path):
                # Load precomputed mask
                R_mask = torch.from_numpy(np.load(restricted_mask_path)).float()
                self.register_buffer('R_mask', R_mask)
            else:
                # Compute mask on the fly
                num_classes = A_tgt.shape[0]
                all_forbidden = self.tgt_forbidden_edges + self.src_forbidden_edges
                all_required = self.tgt_required_edges + self.src_required_edges
                R_mask = compute_restricted_mask(
                    num_classes, all_required, all_forbidden, self.lr_pairs, A_tgt.device
                )
                self.register_buffer('R_mask', R_mask)
        else:
            self.R_mask = None

        # ========== LOSS WEIGHTS ==========
        # Determine effective weights based on mode and overrides
        if graph_align_mode == 'src_only':
            # Only align to source
            self.lambda_spec_src = lambda_spec_src if lambda_spec_src is not None else lambda_spec
            self.lambda_edge_src = lambda_edge_src if lambda_edge_src is not None else lambda_edge
            self.lambda_spec_tgt = 0.0
            self.lambda_edge_tgt = 0.0
        elif graph_align_mode == 'tgt_only':
            # Only align to target (original behavior)
            self.lambda_spec_src = 0.0
            self.lambda_edge_src = 0.0
            self.lambda_spec_tgt = lambda_spec_tgt if lambda_spec_tgt is not None else lambda_spec
            self.lambda_edge_tgt = lambda_edge_tgt if lambda_edge_tgt is not None else lambda_edge
        else:  # 'joint'
            # Align to both, with source as primary and target as regularization
            self.lambda_spec_src = lambda_spec_src if lambda_spec_src is not None else lambda_spec
            self.lambda_edge_src = lambda_edge_src if lambda_edge_src is not None else lambda_edge
            self.lambda_spec_tgt = lambda_spec_tgt if lambda_spec_tgt is not None else lambda_spec * 0.3
            self.lambda_edge_tgt = lambda_edge_tgt if lambda_edge_tgt is not None else lambda_edge * 0.3

        self.lambda_sym = lambda_sym

        # ========== OTHER PARAMETERS ==========
        self.top_k = top_k
        self.temperature = temperature
        self.warmup_epochs = warmup_epochs
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride

        # Store class weights if provided
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        # Track current epoch for warmup
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch for warmup scheduling"""
        self.current_epoch = epoch

    def get_warmup_factor(self) -> float:
        """Get warmup factor for current epoch"""
        if self.current_epoch < self.warmup_epochs:
            return self.current_epoch / max(1, self.warmup_epochs)
        return 1.0

    def forward(self, logits: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute graph prior loss with enhanced dual-branch alignment

        Args:
            logits: (B, C, X, Y, Z) model output logits (from target domain)
            labels: (B, X, Y, Z) ground truth labels (optional, for analysis)

        Returns:
            total_loss: Weighted sum of all loss components
            loss_dict: Dictionary of individual loss components
        """
        # IMPORTANT: Convert to pure tensors to avoid MetaTensor metadata issues
        logits = _to_tensor(logits)
        labels = _to_tensor(labels)

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # Compute predicted adjacency from target domain predictions
        A_pred = soft_adjacency_from_probs(
            probs,
            kernel_size=self.pool_kernel,
            stride=self.pool_stride,
            temperature=self.temperature,
            restricted_mask=self.R_mask if self.use_restricted_mask else None
        )

        # Apply restricted mask to prior adjacencies if enabled
        if self.use_restricted_mask and self.R_mask is not None:
            A_tgt_masked = self.A_tgt * self.R_mask
            if self.has_source_prior:
                A_src_masked = self.A_src * self.R_mask
        else:
            A_tgt_masked = self.A_tgt
            if self.has_source_prior:
                A_src_masked = self.A_src

        # Compute predicted Laplacian
        L_pred = compute_laplacian(A_pred, normalized=True)

        # Initialize all loss components
        loss_spec_src = torch.tensor(0.0, device=logits.device)
        loss_edge_src = torch.tensor(0.0, device=logits.device)
        loss_spec_tgt = torch.tensor(0.0, device=logits.device)
        loss_edge_tgt = torch.tensor(0.0, device=logits.device)

        # ========== SOURCE DOMAIN ALIGNMENT (Primary) ==========
        if self.has_source_prior and self.lambda_spec_src > 0:
            loss_spec_src = spectral_alignment_loss(
                L_pred, self.L_src,
                top_k=self.top_k,
                align_vectors=True,
                eigenvalue_weighted=self.align_U_weighted  # NEW: weighted alignment
            )

        if self.has_source_prior and self.lambda_edge_src > 0:
            loss_edge_src = edge_consistency_loss_with_mismatch(
                A_pred, A_src_masked if self.use_restricted_mask else self.A_src,
                required_edges=self.src_required_edges,
                forbidden_edges=self.src_forbidden_edges,
                class_weights=self.class_weights,
                qap_mismatch_g=self.qap_mismatch_g,  # NEW: mismatch penalty
                restricted_mask=self.R_mask if self.use_restricted_mask else None
            )

        # ========== TARGET DOMAIN REGULARIZATION (Secondary) ==========
        if self.lambda_spec_tgt > 0:
            loss_spec_tgt = spectral_alignment_loss(
                L_pred, self.L_tgt,
                top_k=self.top_k,
                align_vectors=True,
                eigenvalue_weighted=self.align_U_weighted  # NEW: weighted alignment
            )

        if self.lambda_edge_tgt > 0:
            loss_edge_tgt = edge_consistency_loss_with_mismatch(
                A_pred, A_tgt_masked if self.use_restricted_mask else self.A_tgt,
                required_edges=self.tgt_required_edges,
                forbidden_edges=self.tgt_forbidden_edges,
                class_weights=self.class_weights,
                qap_mismatch_g=self.qap_mismatch_g,  # NEW: mismatch penalty
                restricted_mask=self.R_mask if self.use_restricted_mask else None
            )

        # ========== SYMMETRY LOSS (Always active if pairs exist) ==========
        loss_sym = symmetry_consistency_loss(
            probs, self.lr_pairs, flip_dim=2  # X-axis for RAS
        )

        # Apply warmup factor
        warmup_factor = self.get_warmup_factor()

        # Weighted sum
        total_loss = warmup_factor * (
                self.lambda_spec_src * loss_spec_src +
                self.lambda_edge_src * loss_edge_src +
                self.lambda_spec_tgt * loss_spec_tgt +
                self.lambda_edge_tgt * loss_edge_tgt +
                self.lambda_sym * loss_sym
        )

        # FIXED: Compute structural violations for dynamic branch conflict detection
        with torch.no_grad():
            # Thresholds for violation detection (same as in edge_consistency_loss)
            th_required = 0.02
            th_forbidden = 5e-4

            # Count required edges that are missing
            required_missing = 0
            all_required = self.tgt_required_edges + self.src_required_edges
            for i, j in all_required:
                if i < A_pred.shape[0] and j < A_pred.shape[1]:
                    if A_pred[i, j].item() < th_required:
                        required_missing += 1

            # Count forbidden edges that are present
            forbidden_present = 0
            all_forbidden = self.tgt_forbidden_edges + self.src_forbidden_edges
            for i, j in all_forbidden:
                if i < A_pred.shape[0] and j < A_pred.shape[1]:
                    if A_pred[i, j].item() > th_forbidden:
                        forbidden_present += 1

        # Build loss dictionary for logging
        loss_dict = {
            'graph_total': total_loss.detach(),
            # Source alignment (primary)
            'graph_spec_src': loss_spec_src.detach(),
            'graph_edge_src': loss_edge_src.detach(),
            # Target regularization (secondary)
            'graph_spec_tgt': loss_spec_tgt.detach(),
            'graph_edge_tgt': loss_edge_tgt.detach(),
            # Common
            'graph_sym': loss_sym.detach(),
            'graph_struct': torch.tensor(0.0),  # Keep for compatibility
            # FIXED: Add structural violations for trainer to use
            'structural_violations': {
                'required_missing': required_missing,
                'forbidden_present': forbidden_present
            },
            # Legacy naming for backward compatibility
            'graph_spec': (loss_spec_src * self.lambda_spec_src +
                           loss_spec_tgt * self.lambda_spec_tgt).detach() /
                          max(self.lambda_spec_src + self.lambda_spec_tgt, 1e-8),
            'graph_edge': (loss_edge_src * self.lambda_edge_src +
                           loss_edge_tgt * self.lambda_edge_tgt).detach() /
                          max(self.lambda_edge_src + self.lambda_edge_tgt, 1e-8),
            # Metadata
            'warmup_factor': warmup_factor,
            'A_pred': A_pred.detach(),  # Store for validation metrics
            'align_mode': self.graph_align_mode,
            'using_restricted_mask': self.use_restricted_mask,
            'using_weighted_U': self.align_U_weighted
        }

        return total_loss, loss_dict


def compute_validation_graph_metrics(pred: torch.Tensor,
                                     target: torch.Tensor,
                                     prior_adj_path: str,
                                     src_prior_adj_path: Optional[str] = None,
                                     lr_pairs: Optional[List[Tuple[int, int]]] = None,
                                     required_edges: Optional[List[Tuple[int, int]]] = None,
                                     forbidden_edges: Optional[List[Tuple[int, int]]] = None) -> Dict[str, any]:
    """
    Compute graph-based validation metrics including cross-domain alignment

    Args:
        pred: (B, C, X, Y, Z) predicted probabilities or (B, X, Y, Z) argmax predictions
        target: (B, X, Y, Z) ground truth labels
        prior_adj_path: Path to target domain prior adjacency matrix
        src_prior_adj_path: Path to source domain prior adjacency matrix
        lr_pairs: Laterality pairs for symmetry computation
        required_edges: List of required adjacencies
        forbidden_edges: List of forbidden adjacencies

    Returns:
        Dictionary of validation metrics
    """
    # Convert to pure tensors
    pred = _to_tensor(pred)
    target = _to_tensor(target)

    device = pred.device

    # Load target domain prior
    A_prior = torch.from_numpy(np.load(prior_adj_path)).float().to(device)
    row_sums = A_prior.sum(dim=1, keepdim=True).clamp(min=1e-8)
    A_prior = A_prior / row_sums

    # Load source domain prior if available
    if src_prior_adj_path and os.path.exists(src_prior_adj_path):
        A_src = torch.from_numpy(np.load(src_prior_adj_path)).float().to(device)
        row_sums = A_src.sum(dim=1, keepdim=True).clamp(min=1e-8)
        A_src = A_src / row_sums
    else:
        A_src = None

    # If pred is argmax, convert to one-hot
    if len(pred.shape) == 4:  # (B, X, Y, Z)
        num_classes = int(pred.max().item()) + 1
        pred_oh = F.one_hot(pred.long(), num_classes=num_classes)
        pred_oh = pred_oh.permute(0, 4, 1, 2, 3).float()
    else:
        pred_oh = pred
        num_classes = pred.shape[1]

    # Compute predicted adjacency from predictions
    A_pred = soft_adjacency_from_probs(pred_oh, kernel_size=5, stride=3)

    # Compute adjacency from ground truth for comparison
    target_oh = F.one_hot(target.long().clamp(min=0), num_classes=num_classes)
    target_oh = target_oh.permute(0, 4, 1, 2, 3).float()
    A_target = soft_adjacency_from_probs(target_oh, kernel_size=5, stride=3)

    # 1. Adjacency errors (target domain)
    adjacency_errors = {
        'mean_abs_error': float(torch.abs(A_pred - A_prior).mean().item()),
        'max_error': float(torch.abs(A_pred - A_prior).max().item()),
        'vs_gt_error': float(torch.abs(A_pred - A_target).mean().item())
    }

    # Add source domain alignment metrics if available
    if A_src is not None:
        adjacency_errors['mean_abs_error_src'] = float(torch.abs(A_pred - A_src).mean().item())
        adjacency_errors['max_error_src'] = float(torch.abs(A_pred - A_src).max().item())

    # Compute spectral distances
    L_pred = compute_laplacian(A_pred, normalized=True)
    L_prior = compute_laplacian(A_prior, normalized=True)

    try:
        # Target domain spectral distance
        L_pred_f32 = (0.5 * (L_pred + L_pred.T)).float()
        L_prior_f32 = (0.5 * (L_prior + L_prior.T)).float()

        evals_pred, _ = torch.linalg.eigh(L_pred_f32)
        evals_prior, _ = torch.linalg.eigh(L_prior_f32)

        evals_pred = evals_pred.to(L_pred.dtype)
        evals_prior = evals_prior.to(L_prior.dtype)

        evals_pred = _to_tensor(evals_pred)
        evals_prior = _to_tensor(evals_prior)

        k = min(20, evals_pred.shape[0] - 1)
        spectral_dist_tgt = float(F.mse_loss(evals_pred[1:k + 1], evals_prior[1:k + 1]).item())
    except:
        spectral_dist_tgt = 0.0

    adjacency_errors['spectral_distance'] = spectral_dist_tgt

    # Source domain spectral distance if available
    if A_src is not None:
        try:
            L_src = compute_laplacian(A_src, normalized=True)
            L_src_f32 = (0.5 * (L_src + L_src.T)).float()
            evals_src, _ = torch.linalg.eigh(L_src_f32)
            evals_src = _to_tensor(evals_src.to(L_pred.dtype))
            spectral_dist_src = float(F.mse_loss(evals_pred[1:k + 1], evals_src[1:k + 1]).item())
            adjacency_errors['spectral_distance_src'] = spectral_dist_src
        except:
            adjacency_errors['spectral_distance_src'] = 0.0

    # 2. Symmetry scores for laterality pairs
    symmetry_scores = []
    if lr_pairs:
        for left, right in lr_pairs:
            if left < num_classes and right < num_classes:
                # Compute Dice coefficient between left and right predictions
                left_mask = (pred == left) if len(pred.shape) == 4 else (pred.argmax(1) == left)
                right_mask = (pred == right) if len(pred.shape) == 4 else (pred.argmax(1) == right)

                # Flip right mask to align with left
                right_flipped = torch.flip(right_mask, dims=[2])  # Flip X-axis

                intersection = (left_mask & right_flipped).float().sum()
                union = (left_mask | right_flipped).float().sum()

                if union > 0:
                    dice = 2 * intersection / (left_mask.float().sum() + right_flipped.float().sum() + 1e-8)
                    symmetry_scores.append(float(dice.item()))
                else:
                    symmetry_scores.append(1.0)  # Both absent = perfect symmetry

    # 3. Structural violations
    structural_violations = {
        'required_missing': 0,
        'forbidden_present': 0,
        'containment_violated': 0,  # Always 0 now
        'exclusivity_violated': 0  # Always 0 now
    }

    # Check required edges
    if required_edges:
        for i, j in required_edges:
            if i < num_classes and j < num_classes:
                if A_pred[i, j] < 0.01:  # Threshold for "missing"
                    structural_violations['required_missing'] += 1

    # Check forbidden edges
    if forbidden_edges:
        for i, j in forbidden_edges:
            if i < num_classes and j < num_classes:
                if A_pred[i, j] > 0.001:  # Threshold for "present"
                    structural_violations['forbidden_present'] += 1

    return {
        'adjacency_errors': adjacency_errors,
        'symmetry_scores': symmetry_scores,
        'structural_violations': structural_violations
    }