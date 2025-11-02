import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch, MetaTensor
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
import time
import math
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
import psutil
import GPUtil
import json
import os
from collections import deque
from graph_prior_loss import soft_adjacency_from_probs, compute_laplacian
from causal_losses import (
    compute_per_sample_segmentation_losses,
    compute_age_bin_indices,
    compute_conditional_vrex_loss,
    compute_laplacian_invariance_loss,
    compute_counterfactual_consistency_loss,
    generate_counterfactuals,
    compute_age_balance_weights,
)


def is_dist():
    return dist.is_initialized()


def dist_mean_scalar(x: torch.Tensor):
    if is_dist():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x


def compute_grad_norm(parameters) -> float:
    total_norm_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_sq += float(param_norm.item()) ** 2
    return math.sqrt(total_norm_sq) if total_norm_sq > 0 else 0.0


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, dice_weight=0.0, ce_weight=0.0, focal_weight=0.0, include_background=True, focal_gamma=2.0,
                 class_weights=None, foreground_only=False, loss_config="dice_ce"):
        super().__init__()
        self.foreground_only = foreground_only
        self.num_classes = 87 if foreground_only else 88
        self.loss_config = loss_config
        self.include_background = include_background
        self.focal_gamma = focal_gamma
        self.class_weights_tensor = None
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                self.class_weights_tensor = class_weights.detach().clone().cpu()
            else:
                self.class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        if loss_config == "dice_ce":
            self.dice_weight = 0.6
            self.ce_weight = 0.4
            self.focal_weight = 0.0
        elif loss_config == "dice_focal":
            self.dice_weight = 0.5
            self.ce_weight = 0.1
            self.focal_weight = 0.4
        elif loss_config == "dice_ce_focal":
            self.dice_weight = 0.4
            self.ce_weight = 0.3
            self.focal_weight = 0.3
        else:
            self.dice_weight = dice_weight
            self.ce_weight = ce_weight
            self.focal_weight = focal_weight

        if self.dice_weight > 0:
            self.dice_loss = DiceLoss(
                to_onehot_y=False,
                softmax=True,
                include_background=True,
                squared_pred=True,
                reduction="mean",
            )
        if self.ce_weight > 0:
            if class_weights is not None:
                self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
            else:
                self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if self.focal_weight > 0:
            self.focal_loss = FocalLoss(include_background=True, to_onehot_y=False, gamma=focal_gamma,
                                        weight=class_weights if class_weights is not None else None, reduction="mean")

    def get_loss_config(self) -> Dict[str, object]:
        return {
            'dice_weight': float(self.dice_weight),
            'ce_weight': float(self.ce_weight),
            'focal_weight': float(self.focal_weight),
            'focal_gamma': float(self.focal_gamma),
            'num_classes': int(self.num_classes),
            'include_background': bool(self.include_background),
            'foreground_only': bool(self.foreground_only),
            'class_weights': self.class_weights_tensor,
            'dice_squared': True,
        }

    def _create_one_hot_with_ignore(self, labels, num_classes):
        valid_mask = labels != -1
        labels_for_onehot = labels.clone()
        labels_for_onehot[~valid_mask] = 0
        one_hot = F.one_hot(labels_for_onehot.long(), num_classes=num_classes)
        one_hot = one_hot.permute(0, 4, 1, 2, 3)
        one_hot = one_hot * valid_mask.unsqueeze(1)
        return one_hot, valid_mask

    def forward(self, pred, target):
        if len(target.shape) == 5 and target.shape[1] == 1:
            target = target.squeeze(1)
        total_loss = 0.0
        loss_dict = {}

        if self.ce_weight > 0:
            ce = self.ce_loss(pred, target.long())
            total_loss += self.ce_weight * ce
            loss_dict['ce'] = ce.item()
        else:
            loss_dict['ce'] = 0.0

        one_hot_target, valid_mask = self._create_one_hot_with_ignore(target, self.num_classes)
        one_hot_target = one_hot_target.to(dtype=pred.dtype)
        mask = valid_mask.unsqueeze(1).to(dtype=pred.dtype)
        pred_for_dice = pred * mask

        if self.dice_weight > 0:
            dice = self.dice_loss(pred_for_dice, one_hot_target)
            total_loss += self.dice_weight * dice
            loss_dict['dice'] = dice.item()
        else:
            loss_dict['dice'] = 0.0

        if self.focal_weight > 0:
            focal = self.focal_loss(pred * mask, one_hot_target)
            total_loss += self.focal_weight * focal
            loss_dict['focal'] = focal.item()
        else:
            loss_dict['focal'] = 0.0

        loss_dict['total'] = total_loss
        return total_loss, loss_dict


def train_epoch_causal(model, source_loader, target_loader, optimizer, epoch, total_epochs, writer, args,
                       device=None, is_distributed=False, world_size=1, rank=0, age_graph_loss=None,
                       causal_config: Optional[Dict[str, object]] = None,
                       age_hist_info: Optional[Dict[str, torch.Tensor]] = None):
    """Training epoch with age-aware and causal losses."""

    is_main = (not is_distributed) or rank == 0
    debug_enabled = bool(getattr(args, 'debug_mode', False))
    debug_step_limit = max(1, getattr(args, 'debug_step_limit', 2))

    def debug_print(msg):
        if debug_enabled and is_main:
            print(f"[DEBUG][Train][Epoch {epoch}] {msg}", flush=True)

    # Helper to safely convert tensors to Python scalars
    def _to_scalar(val):
        if isinstance(val, torch.Tensor):
            return float(val.detach().item())
        if isinstance(val, (int, float)):
            return float(val)
        return 0.0

    causal_cfg = causal_config or getattr(args, 'causal_config', None) or {}
    age_hist_cfg = age_hist_info or getattr(args, 'age_hist_info', None) or {}
    bin_edges = age_hist_cfg.get('bin_edges')
    if isinstance(bin_edges, np.ndarray):
        bin_edges = torch.tensor(bin_edges, dtype=torch.float32)
    elif isinstance(bin_edges, list):
        bin_edges = torch.tensor(bin_edges, dtype=torch.float32)
    source_hist = age_hist_cfg.get('source_hist')
    target_hist = age_hist_cfg.get('target_hist')
    if isinstance(source_hist, np.ndarray):
        source_hist = torch.tensor(source_hist, dtype=torch.float32)
    if isinstance(target_hist, np.ndarray):
        target_hist = torch.tensor(target_hist, dtype=torch.float32)

    # Update epoch for age graph loss if available
    if age_graph_loss is not None:
        age_graph_loss.set_epoch(epoch)

        if is_main:
            print(f"\n🧠 Using Age-Conditioned Graph Prior Loss")
            align_mode = getattr(age_graph_loss, 'graph_align_mode', 'none')
            print(f"  Alignment mode: {align_mode}")
            if getattr(age_graph_loss, 'lambda_dyn', 0.0) > 0:
                print(f"  Dynamic spectral branch scheduled from epoch {age_graph_loss.dyn_start_epoch}")
            print(f"  Current epoch: {epoch}, Warmup factor: {age_graph_loss.get_warmup_factor():.3f}")

    restricted_mask_cached: Optional[torch.Tensor] = None
    if age_graph_loss is not None and hasattr(age_graph_loss, 'R_mask'):
        R_mask = age_graph_loss.R_mask
        if R_mask is not None and isinstance(R_mask, torch.Tensor) and R_mask.numel() > 0:
            restricted_mask_cached = R_mask.to(device=device)

    model.train()
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
    if age_graph_loss is not None and hasattr(actual_model, "age_conditioner"):
        actual_model.age_conditioner.set_strength(age_graph_loss.get_warmup_factor())

    if debug_enabled and is_main:
        try:
            source_len = len(source_loader.dataset)
        except Exception:
            source_len = 'unknown'
        try:
            target_len = len(target_loader.dataset)
        except Exception:
            target_len = 'unknown'
        debug_print(
            f"Debugging first {debug_step_limit} steps (source dataset={source_len}, target dataset={target_len}, "
            f"batch_size={getattr(args, 'batch_size', 'unknown')})"
        )

    # Loss accumulators
    total_loss = 0
    seg_loss_total = 0
    target_seg_loss_total = 0
    dice_loss_total = 0
    ce_loss_total = 0
    focal_loss_total = 0

    # Age-aware loss accumulators
    age_loss_total = 0
    volume_loss_total = 0
    shape_loss_total = 0
    weighted_adj_loss_total = 0
    vrex_loss_total = 0
    vrex_steps = 0
    lapinv_loss_total = 0
    lapinv_steps = 0
    cf_loss_total = 0
    cf_steps = 0
    age_balance_weight_total = 0
    age_balance_batches = 0

    # Graph alignment accumulators
    graph_loss_total = 0
    graph_spec_src_total = 0
    graph_edge_src_total = 0
    graph_spec_tgt_total = 0
    graph_edge_tgt_total = 0
    graph_sym_total = 0
    forbidden_violations_total = 0
    required_violations_total = 0

    # Dynamic spectral accumulators
    dyn_spec_loss_total = 0
    dyn_conflict_suppressions = 0

    structural_violations_tracker = deque(maxlen=5)

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    num_steps = min(len(source_loader), len(target_loader))

    if is_main:
        print(f"\n🚀 Training - Epoch {epoch}/{total_epochs}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        if world_size > 1:
            print(f"  World size: {world_size} GPUs")

    use_target_labels = getattr(args, 'use_target_labels', True)
    target_label_start_epoch = getattr(args, 'target_label_start_epoch', 1)
    target_label_weight = getattr(args, 'target_label_weight', 0.5)

    if use_target_labels and epoch >= target_label_start_epoch:
        progress = (epoch - target_label_start_epoch) / (total_epochs - target_label_start_epoch)
        current_target_weight = target_label_weight * min(1.0, progress * 2)
        if is_main:
            print(f"  📊 Using target labels with weight: {current_target_weight:.3f}")
    else:
        current_target_weight = 0.0

    if age_graph_loss is not None:
        dyn_start_epoch = getattr(age_graph_loss, 'dyn_start_epoch', float('inf'))
        dyn_ramp_epochs = getattr(age_graph_loss, 'dyn_ramp_epochs', 1)
        dyn_top_k_max = getattr(age_graph_loss, 'dyn_top_k', 12)
        lambda_dyn_base = getattr(age_graph_loss, 'lambda_dyn', 0.0)

        dynamic_branch_allowed = getattr(args, 'dynamic_branch_enabled', True)
        if hasattr(age_graph_loss, 'is_dynamic_branch_enabled'):
            dynamic_branch_allowed = dynamic_branch_allowed and age_graph_loss.is_dynamic_branch_enabled()
        if not dynamic_branch_allowed:
            lambda_dyn_base = 0.0

        use_restricted_mask_dyn = getattr(age_graph_loss, 'use_restricted_mask', False)
        if use_restricted_mask_dyn and hasattr(age_graph_loss, 'R_mask'):
            R_mask_tensor = age_graph_loss.R_mask
            R_mask = R_mask_tensor if R_mask_tensor.numel() > 0 else None
        else:
            R_mask = None
        dyn_temperature = getattr(age_graph_loss, 'temperature', 1.0)
        pool_kernel = getattr(age_graph_loss, 'pool_kernel', 3)
        pool_stride = getattr(age_graph_loss, 'pool_stride', 2)
    else:
        dyn_start_epoch = float('inf')
        dyn_ramp_epochs = 1
        dyn_top_k_max = 12
        lambda_dyn_base = 0.0
        use_restricted_mask_dyn = False
        R_mask = None
        dyn_temperature = 1.0
        pool_kernel = 3
        pool_stride = 2

    in_dynamic_stage = epoch >= dyn_start_epoch and lambda_dyn_base > 0
    if in_dynamic_stage:
        ramp_progress = min(1.0, (epoch - dyn_start_epoch + 1) / max(1, dyn_ramp_epochs))
        lambda_dyn_effective = lambda_dyn_base * ramp_progress
        dyn_k_effective = max(4, int(round(4 + (dyn_top_k_max - 4) * ramp_progress)))
        if is_main:
            print(f"  🔄 Dynamic spectral alignment active: λ={lambda_dyn_effective:.3f}, k={dyn_k_effective}")
    else:
        lambda_dyn_effective = 0.0
        dyn_k_effective = 0

    lambda_dyn_state = {'value': lambda_dyn_effective}

    def apply_dynamic_alignment(losses_dict, src_logits, tgt_logits, step_index: int):
        """Run the dynamic spectral alignment branch with optional pooling."""
        nonlocal dyn_spec_loss_total, dyn_conflict_suppressions

        if not (age_graph_loss is not None and in_dynamic_stage
                and src_logits is not None and tgt_logits is not None):
            return

        restricted_mask_dyn = None
        if use_restricted_mask_dyn and R_mask is not None:
            restricted_mask_dyn = R_mask.to(device=device, dtype=src_logits.dtype)

        dyn_pool_kernel = getattr(age_graph_loss, 'dyn_pool_kernel', pool_kernel)
        dyn_pool_stride = getattr(age_graph_loss, 'dyn_pool_stride', pool_stride)
        dyn_pre_pool_kernel = getattr(age_graph_loss, 'dyn_pre_pool_kernel', 1)
        dyn_pre_pool_stride = getattr(age_graph_loss, 'dyn_pre_pool_stride', 1)

        source_logits_dyn = src_logits
        target_logits_dyn = tgt_logits
        if dyn_pre_pool_stride > 1:
            pad = dyn_pre_pool_kernel // 2
            source_logits_dyn = F.avg_pool3d(
                src_logits,
                kernel_size=dyn_pre_pool_kernel,
                stride=dyn_pre_pool_stride,
                padding=pad,
                count_include_pad=False,
            )
            target_logits_dyn = F.avg_pool3d(
                tgt_logits,
                kernel_size=dyn_pre_pool_kernel,
                stride=dyn_pre_pool_stride,
                padding=pad,
                count_include_pad=False,
            )

        P_s = torch.softmax(source_logits_dyn, dim=1)
        P_t = torch.softmax(target_logits_dyn, dim=1)

        dyn_pool_cfg = {'kernel_size': dyn_pool_kernel, 'stride': dyn_pool_stride}

        A_s = soft_adjacency_from_probs(P_s, temperature=dyn_temperature,
                                        restricted_mask=restricted_mask_dyn, **dyn_pool_cfg)
        A_t = soft_adjacency_from_probs(P_t, temperature=dyn_temperature,
                                        restricted_mask=restricted_mask_dyn, **dyn_pool_cfg)

        L_s = compute_laplacian(A_s, normalized=True)
        L_t = compute_laplacian(A_t, normalized=True)

        del P_s, P_t
        if source_logits_dyn is not src_logits:
            del source_logits_dyn
        if target_logits_dyn is not tgt_logits:
            del target_logits_dyn

        L_s_sym = 0.5 * (L_s + L_s.T)
        L_t_sym = 0.5 * (L_t + L_t.T)

        eig_s, _ = torch.linalg.eigh(L_s_sym.float())
        eig_t, _ = torch.linalg.eigh(L_t_sym.float())
        eig_s = eig_s.to(src_logits.dtype)
        eig_t = eig_t.to(tgt_logits.dtype)

        k = min(dyn_k_effective, eig_s.shape[-1] - 1)
        if k <= 0:
            return

        L_dyn = F.smooth_l1_loss(eig_s[1:k + 1], eig_t[1:k + 1])

        effective_lambda = lambda_dyn_state['value']
        conflict_triggered = False
        if len(structural_violations_tracker) >= 3:
            recent = list(structural_violations_tracker)[-3:]
            conflict_triggered = recent[-1] > recent[0] * 1.2
            if conflict_triggered:
                lambda_dyn_state['value'] *= 0.5
                effective_lambda = lambda_dyn_state['value']
                dyn_conflict_suppressions += 1

        if conflict_triggered and is_main and step_index == 0:
            print(f"  ⚠️ Dynamic conflicts detected, reducing λ_dyn to {effective_lambda:.4f}")

        losses_dict['dyn_spec'] = L_dyn
        losses_dict['total'] = losses_dict['total'] + effective_lambda * L_dyn
        dyn_spec_loss_total += L_dyn.item()

    def apply_dynamic_alignment(losses_dict, src_logits, tgt_logits, step_index: int):
        """Run the dynamic spectral alignment branch with optional pooling."""
        nonlocal dyn_spec_loss_total, dyn_conflict_suppressions

        if not (age_graph_loss is not None and in_dynamic_stage
                and src_logits is not None and tgt_logits is not None):
            return

        restricted_mask_dyn = None
        if use_restricted_mask_dyn and R_mask is not None:
            restricted_mask_dyn = R_mask.to(device=device, dtype=src_logits.dtype)

        dyn_pool_kernel = getattr(age_graph_loss, 'dyn_pool_kernel', pool_kernel)
        dyn_pool_stride = getattr(age_graph_loss, 'dyn_pool_stride', pool_stride)
        dyn_pre_pool_kernel = getattr(age_graph_loss, 'dyn_pre_pool_kernel', 1)
        dyn_pre_pool_stride = getattr(age_graph_loss, 'dyn_pre_pool_stride', 1)

        source_logits_dyn = src_logits
        target_logits_dyn = tgt_logits
        if dyn_pre_pool_stride > 1:
            pad = dyn_pre_pool_kernel // 2
            source_logits_dyn = F.avg_pool3d(
                src_logits,
                kernel_size=dyn_pre_pool_kernel,
                stride=dyn_pre_pool_stride,
                padding=pad,
                count_include_pad=False,
            )
            target_logits_dyn = F.avg_pool3d(
                tgt_logits,
                kernel_size=dyn_pre_pool_kernel,
                stride=dyn_pre_pool_stride,
                padding=pad,
                count_include_pad=False,
            )

        P_s = torch.softmax(source_logits_dyn, dim=1)
        P_t = torch.softmax(target_logits_dyn, dim=1)

        dyn_pool_cfg = {'kernel_size': dyn_pool_kernel, 'stride': dyn_pool_stride}

        A_s = soft_adjacency_from_probs(P_s, temperature=dyn_temperature,
                                        restricted_mask=restricted_mask_dyn, **dyn_pool_cfg)
        A_t = soft_adjacency_from_probs(P_t, temperature=dyn_temperature,
                                        restricted_mask=restricted_mask_dyn, **dyn_pool_cfg)

        L_s = compute_laplacian(A_s, normalized=True)
        L_t = compute_laplacian(A_t, normalized=True)

        del P_s, P_t
        if source_logits_dyn is not src_logits:
            del source_logits_dyn
        if target_logits_dyn is not tgt_logits:
            del target_logits_dyn

        L_s_sym = 0.5 * (L_s + L_s.T)
        L_t_sym = 0.5 * (L_t + L_t.T)

        eig_s, _ = torch.linalg.eigh(L_s_sym.float())
        eig_t, _ = torch.linalg.eigh(L_t_sym.float())
        eig_s = eig_s.to(src_logits.dtype)
        eig_t = eig_t.to(tgt_logits.dtype)

        k = min(dyn_k_effective, eig_s.shape[-1] - 1)
        if k <= 0:
            return

        L_dyn = F.smooth_l1_loss(eig_s[1:k + 1], eig_t[1:k + 1])

        effective_lambda = lambda_dyn_state['value']
        conflict_triggered = False
        if len(structural_violations_tracker) >= 3:
            recent = list(structural_violations_tracker)[-3:]
            conflict_triggered = recent[-1] > recent[0] * 1.2
            if conflict_triggered:
                lambda_dyn_state['value'] *= 0.5
                effective_lambda = lambda_dyn_state['value']
                dyn_conflict_suppressions += 1

        if conflict_triggered and is_main and step_index == 0:
            print(f"  ⚠️ Dynamic conflicts detected, reducing λ_dyn to {effective_lambda:.4f}")

        losses_dict['dyn_spec'] = L_dyn
        losses_dict['total'] = losses_dict['total'] + effective_lambda * L_dyn
        dyn_spec_loss_total += L_dyn.item()

    # Create segmentation loss
    loss_config = getattr(args, 'loss_config', 'dice_focal')
    focal_gamma = getattr(args, 'focal_gamma', 2.0)

    if hasattr(actual_model, 'class_weights') and actual_model.class_weights is not None:
        seg_criterion = CombinedSegmentationLoss(
            loss_config=loss_config, include_background=True,
            focal_gamma=focal_gamma, class_weights=actual_model.class_weights,
            foreground_only=args.foreground_only
        )
    else:
        seg_criterion = CombinedSegmentationLoss(
            loss_config=loss_config,
            include_background=True,
            focal_gamma=focal_gamma,
            foreground_only=args.foreground_only
        )

    start_time = time.time()

    # Progress bar
    pbar = tqdm(range(num_steps), desc=f"Epoch {epoch}") if is_main else range(num_steps)

    # Mixed precision setup
    use_amp = getattr(args, 'use_amp', True)
    amp_dtype = torch.bfloat16 if getattr(args, 'amp_dtype', 'bfloat16') == 'bfloat16' else torch.float16

    for i in pbar:
        # Get batches
        try:
            source_batch = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_batch = next(source_iter)

        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)

        # Extract data with ages
        source_images = source_batch["image"].to(device, non_blocking=True)
        source_labels = source_batch["label"].to(device, non_blocking=True)
        source_ages = source_batch.get("age", torch.tensor([40.0]).repeat(source_images.shape[0], 1)).to(device)

        target_images = target_batch["image"].to(device, non_blocking=True)
        target_labels = target_batch["label"].to(device, non_blocking=True)
        target_ages = target_batch.get("age", torch.tensor([40.0]).repeat(target_images.shape[0], 1)).to(device)

        # Fix label dimensions
        if len(source_labels.shape) == 5 and source_labels.shape[1] == 1:
            source_labels = source_labels.squeeze(1)
        if len(target_labels.shape) == 5 and target_labels.shape[1] == 1:
            target_labels = target_labels.squeeze(1)

        optimizer.zero_grad()

        global_step = (epoch - 1) * num_steps + i
        debug_active = debug_enabled and (i < debug_step_limit)

        if debug_active:
            debug_print(
                f"Step {i}: src{tuple(source_images.shape)}->lbl{tuple(source_labels.shape)}, "
                f"tgt{tuple(target_images.shape)}->lbl{tuple(target_labels.shape)}"
            )
            if i == 0:
                with torch.no_grad():
                    src_stats = (
                        float(source_images.min().item()),
                        float(source_images.max().item()),
                        float(source_images.mean().item()),
                        float(source_images.std().item()),
                    )
                    tgt_stats = (
                        float(target_images.min().item()),
                        float(target_images.max().item()),
                        float(target_images.mean().item()),
                        float(target_images.std().item()),
                    )
                    src_unique = torch.unique(source_labels.detach())
                    tgt_unique = torch.unique(target_labels.detach())
                    debug_print(
                        "         src[min={:.3f}, max={:.3f}, mean={:.3f}, std={:.3f}], "
                        "tgt[min={:.3f}, max={:.3f}, mean={:.3f}, std={:.3f}]".format(
                            *src_stats, *tgt_stats
                        )
                    )
                    debug_print(
                        f"         src labels unique={src_unique.numel()} (range {int(src_unique.min().item())}-{int(src_unique.max().item())}), "
                        f"tgt labels unique={tgt_unique.numel()} (range {int(tgt_unique.min().item())}-{int(tgt_unique.max().item())})"
                    )
                    if source_ages is not None:
                        ages = source_ages.detach().cpu().numpy()
                        debug_print(
                            f"         src ages {ages.min():.2f}-{ages.max():.2f} (mean {ages.mean():.2f})"
                        )
                    if target_ages is not None:
                        ages_t = target_ages.detach().cpu().numpy()
                        debug_print(
                            f"         tgt ages {ages_t.min():.2f}-{ages_t.max():.2f} (mean {ages_t.mean():.2f})"
                        )

        graph_debug_info = None
        target_seg_components_saved = None

        # Forward pass with mixed precision
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                # Compute source segmentation loss with age
                losses = actual_model.compute_losses(
                    source_images,
                    source_labels,
                    seg_criterion,
                    source_ages=source_ages,
                    step=global_step
                )

                # Extract logits
                source_logits = losses.pop('logits')

                # Compute target segmentation loss if enabled
                target_logits = None
                if current_target_weight > 0:
                    target_logits = actual_model.forward(target_images, target_ages)
                    target_seg_loss, target_seg_components = seg_criterion(target_logits, target_labels)
                    losses['target_seg_loss'] = target_seg_loss
                    losses['total'] = losses['total'] + current_target_weight * target_seg_loss
                    target_seg_loss_total += target_seg_loss.item()
                    target_seg_components_saved = target_seg_components

                # Apply age-conditioned graph loss
                if age_graph_loss is not None:
                    # Use target domain predictions and ages
                    if target_logits is not None:
                        graph_total, graph_dict = age_graph_loss(target_logits, target_labels, target_ages)
                    else:
                        graph_total, graph_dict = age_graph_loss(source_logits, source_labels, source_ages)

                    losses['total'] = losses['total'] + graph_total
                    losses['graph_total'] = graph_total
                    graph_debug_info = graph_dict

                    # Accumulate age-aware losses
                    if 'volume_loss' in graph_dict:
                        volume_loss_total += graph_dict['volume_loss'].item()
                    if 'shape_loss' in graph_dict:
                        shape_loss_total += graph_dict['shape_loss'].item()
                    if 'weighted_adj_loss' in graph_dict:
                        weighted_adj_loss_total += graph_dict['weighted_adj_loss'].item()

                    graph_loss_total += _to_scalar(graph_total)
                    graph_spec_src_total += _to_scalar(graph_dict.get('graph_spec_src', 0))
                    graph_edge_src_total += _to_scalar(graph_dict.get('graph_edge_src', 0))
                    graph_spec_tgt_total += _to_scalar(graph_dict.get('graph_spec_tgt', 0))
                    graph_edge_tgt_total += _to_scalar(graph_dict.get('graph_edge_tgt', 0))
                    graph_sym_total += _to_scalar(graph_dict.get('graph_sym', 0))

                    structural = graph_dict.get('structural_violations', {})
                    if isinstance(structural, dict):
                        forbidden_count = structural.get('forbidden_present', 0)
                        required_count = structural.get('required_missing', 0)
                        weighted_violations = forbidden_count * 1.5 + required_count
                        structural_violations_tracker.append(weighted_violations)
                        forbidden_violations_total += forbidden_count
                        required_violations_total += required_count

                apply_dynamic_alignment(losses, source_logits, target_logits, i)
        else:
            # Non-AMP version
            losses = actual_model.compute_losses(
                source_images,
                source_labels,
                seg_criterion,
                source_ages=source_ages,
                step=global_step
            )

            source_logits = losses.pop('logits')

            target_logits = None
            if current_target_weight > 0:
                target_logits = actual_model.forward(target_images, target_ages)
                target_seg_loss, target_seg_components = seg_criterion(target_logits, target_labels)
                losses['target_seg_loss'] = target_seg_loss
                losses['total'] = losses['total'] + current_target_weight * target_seg_loss
                target_seg_loss_total += target_seg_loss.item()
                target_seg_components_saved = target_seg_components

            if age_graph_loss is not None:
                if target_logits is not None:
                    graph_total, graph_dict = age_graph_loss(target_logits, target_labels, target_ages)
                else:
                    graph_total, graph_dict = age_graph_loss(source_logits, source_labels, source_ages)

                losses['total'] = losses['total'] + graph_total
                losses['graph_total'] = graph_total
                graph_debug_info = graph_dict

                if 'volume_loss' in graph_dict:
                    volume_loss_total += graph_dict['volume_loss'].item()
                if 'shape_loss' in graph_dict:
                    shape_loss_total += graph_dict['shape_loss'].item()
                if 'weighted_adj_loss' in graph_dict:
                    weighted_adj_loss_total += graph_dict['weighted_adj_loss'].item()

                graph_loss_total += _to_scalar(graph_total)
                graph_spec_src_total += _to_scalar(graph_dict.get('graph_spec_src', 0))
                graph_edge_src_total += _to_scalar(graph_dict.get('graph_edge_src', 0))
                graph_spec_tgt_total += _to_scalar(graph_dict.get('graph_spec_tgt', 0))
                graph_edge_tgt_total += _to_scalar(graph_dict.get('graph_edge_tgt', 0))
                graph_sym_total += _to_scalar(graph_dict.get('graph_sym', 0))

                structural = graph_dict.get('structural_violations', {})
                if isinstance(structural, dict):
                    forbidden_count = structural.get('forbidden_present', 0)
                    required_count = structural.get('required_missing', 0)
                    weighted_violations = forbidden_count * 1.5 + required_count
                    structural_violations_tracker.append(weighted_violations)
                    forbidden_violations_total += forbidden_count
                    required_violations_total += required_count

            apply_dynamic_alignment(losses, source_logits, target_logits, i)

        # === Causal regularizers ===
        loss_config = seg_criterion.get_loss_config() if hasattr(seg_criterion, 'get_loss_config') else {
            'dice_weight': getattr(seg_criterion, 'dice_weight', 0.0),
            'ce_weight': getattr(seg_criterion, 'ce_weight', 0.0),
            'focal_weight': getattr(seg_criterion, 'focal_weight', 0.0),
            'focal_gamma': getattr(seg_criterion, 'focal_gamma', 2.0),
            'num_classes': getattr(seg_criterion, 'num_classes', source_logits.shape[1]),
            'class_weights': getattr(seg_criterion, 'class_weights_tensor', None),
            'dice_squared': True,
        }

        per_sample_source = compute_per_sample_segmentation_losses(source_logits, source_labels, loss_config)
        bin_edges_device: Optional[torch.Tensor] = None
        if bin_edges is not None:
            bin_edges_device = bin_edges.to(device=device, dtype=source_ages.dtype)
            bins_source = compute_age_bin_indices(source_ages, bin_edges_device)
        else:
            bins_source = None

        per_sample_target = None
        bins_target = None
        if target_logits is not None:
            per_sample_target = compute_per_sample_segmentation_losses(target_logits, target_labels, loss_config)
            if bin_edges is not None:
                bins_target = compute_age_bin_indices(target_ages, bin_edges_device)

        # Age balance reweighting for source domain
        if (causal_cfg.get('enable_age_balance', False)
                and epoch >= int(causal_cfg.get('age_balance_start', 0))
                and bin_edges_device is not None
                and source_hist is not None and target_hist is not None):
            weights = compute_age_balance_weights(
                source_ages,
                bin_edges_device,
                source_hist.to(device=device, dtype=source_ages.dtype),
                target_hist.to(device=device, dtype=source_ages.dtype),
            )
            weights = weights.detach()
            weights = weights / weights.mean().clamp(min=1e-6)
            weighted_seg = (per_sample_source['total'] * weights).sum() / weights.sum().clamp(min=1e-6)
            losses['total'] = losses['total'] - losses['seg_loss'] + weighted_seg
            losses['seg_loss'] = weighted_seg
            age_balance_weight_total += float(weights.mean().item())
            age_balance_batches += 1

        # Conditional V-REx regularizer
        if (causal_cfg.get('enable_vrex', False)
                and epoch >= int(causal_cfg.get('vrex_start_epoch', 0))
                and bins_source is not None):
            domain_map = {'source': (per_sample_source['total'], bins_source)}
            if per_sample_target is not None and bins_target is not None:
                domain_map['target'] = (per_sample_target['total'], bins_target)
            if len(domain_map) > 1:
                vrex_raw = compute_conditional_vrex_loss(
                    domain_map,
                    min_count=int(causal_cfg.get('vrex_min_count', 1)),
                )
                lambda_vrex = float(causal_cfg.get('vrex_lambda', 0.0))
                if lambda_vrex != 0.0:
                    losses['total'] = losses['total'] + lambda_vrex * vrex_raw
                losses['vrex_loss'] = vrex_raw
                vrex_loss_total += float(vrex_raw.item())
                vrex_steps += 1

        # Laplacian residual invariance
        if (causal_cfg.get('enable_lapinv', False)
                and epoch >= int(causal_cfg.get('lapinv_start_epoch', 0))
                and per_sample_target is not None and bins_source is not None and bins_target is not None):
            lap_loss = compute_laplacian_invariance_loss(
                {
                    'source': torch.softmax(source_logits, dim=1),
                    'target': torch.softmax(target_logits, dim=1),
                },
                {
                    'source': bins_source,
                    'target': bins_target,
                },
                soft_adjacency_from_probs,
                compute_laplacian,
                pool_kernel=int(causal_cfg.get('lapinv_pool_kernel', 3)),
                pool_stride=int(causal_cfg.get('lapinv_pool_stride', 2)),
                temperature=float(causal_cfg.get('lapinv_temperature', 1.0)),
                min_count=int(causal_cfg.get('lapinv_min_count', 1)),
                restricted_mask=restricted_mask_cached,
            )
            lambda_lap = float(causal_cfg.get('lapinv_lambda', 0.0))
            if lambda_lap != 0.0:
                losses['total'] = losses['total'] + lambda_lap * lap_loss
            losses['lapinv_loss'] = lap_loss
            lapinv_loss_total += float(lap_loss.item())
            lapinv_steps += 1

        # Counterfactual consistency
        if (causal_cfg.get('enable_counterfactual', False)
                and epoch >= int(causal_cfg.get('cf_start_epoch', 0))
                and float(causal_cfg.get('cf_sample_rate', 0.0)) > 0):
            sample_rate = float(causal_cfg.get('cf_sample_rate', 0.0))
            cf_mask = torch.rand(source_images.size(0), device=device) < sample_rate
            if cf_mask.any():
                cf_indices = torch.nonzero(cf_mask, as_tuple=True)[0]

                if isinstance(source_images, MetaTensor):
                    source_images_tensor = source_images.as_tensor()
                else:
                    source_images_tensor = source_images

                cf_images = generate_counterfactuals(
                    source_images_tensor[cf_indices],
                    intensity_scale=float(causal_cfg.get('cf_intensity_scale', 0.25)),
                    intensity_shift=float(causal_cfg.get('cf_intensity_shift', 0.15)),
                    noise_std=float(causal_cfg.get('cf_noise_std', 0.03)),
                    bias_field_strength=float(causal_cfg.get('cf_bias_strength', 0.1)),
                    clip_min=float(causal_cfg.get('cf_clip_min', -3.0)),
                    clip_max=float(causal_cfg.get('cf_clip_max', 3.0)),
                )
                if isinstance(source_ages, MetaTensor):
                    source_ages_tensor = source_ages.as_tensor()
                else:
                    source_ages_tensor = source_ages

                if isinstance(source_logits, MetaTensor):
                    source_logits_tensor = source_logits.as_tensor()
                else:
                    source_logits_tensor = source_logits

                cf_logits = actual_model.forward(cf_images, source_ages_tensor[cf_indices])
                cf_raw = compute_counterfactual_consistency_loss(
                    source_logits_tensor[cf_indices],
                    cf_logits,
                    confidence_threshold=float(causal_cfg.get('cf_confidence_threshold', 0.6)),
                )
                lambda_cf = float(causal_cfg.get('cf_lambda', 0.0))
                if lambda_cf != 0.0:
                    losses['total'] = losses['total'] + lambda_cf * cf_raw
                losses['counterfactual_loss'] = cf_raw
                cf_loss_total += float(cf_raw.item())
                cf_steps += 1

        if debug_active:
            summary_parts = [
                f"total={losses['total'].item():.4f}",
                f"seg={losses['seg_loss'].item():.4f}",
            ]
            if 'target_seg_loss' in losses:
                summary_parts.append(f"tgt={losses['target_seg_loss'].item():.4f}")
            if 'age_loss' in losses:
                summary_parts.append(f"age={losses['age_loss'].item():.4f}")
            if 'volume_loss' in losses:
                summary_parts.append(f"vol={losses['volume_loss'].item():.2f}")
            if 'graph_total' in losses:
                summary_parts.append(f"graph={losses['graph_total'].item():.4f}")
            if 'dyn_spec' in losses:
                summary_parts.append(f"dyn={losses['dyn_spec'].item():.4f}")
            if 'vrex_loss' in losses:
                summary_parts.append(f"vrex={losses['vrex_loss'].item():.4f}")
            if 'lapinv_loss' in losses:
                summary_parts.append(f"lap={losses['lapinv_loss'].item():.4f}")
            if 'counterfactual_loss' in losses:
                summary_parts.append(f"cf={losses['counterfactual_loss'].item():.4f}")
            debug_print("Step {} summary: {}".format(i, ", ".join(summary_parts)))

            if torch.isnan(losses['total']).any():
                debug_print("         ⚠️ NaN detected in total loss")

            seg_comps = losses.get('seg_loss_components')
            if seg_comps:
                debug_print(
                    "         seg_parts: "
                    + ", ".join(f"{k}={float(v):.5f}" for k, v in seg_comps.items())
                )
            if target_seg_components_saved:
                debug_print(
                    "         tgt_parts: "
                    + ", ".join(
                        f"{k}={float(v):.5f}" for k, v in target_seg_components_saved.items()
                    )
                )

            if graph_debug_info:
                warmup = graph_debug_info.get('warmup_factor')
                age_warm = graph_debug_info.get('age_warmup_factor')
                graph_parts = []
                if warmup is not None:
                    graph_parts.append(f"warmup={float(_to_scalar(warmup)):.3f}")
                if age_warm is not None:
                    graph_parts.append(f"age_warm={float(_to_scalar(age_warm)):.3f}")
                for key, label in [
                    ('volume_loss', 'vol'),
                    ('shape_loss', 'shape'),
                    ('weighted_adj_loss', 'w_adj'),
                    ('graph_spec_src', 'spec_src'),
                    ('graph_edge_src', 'edge_src'),
                    ('graph_spec_tgt', 'spec_tgt'),
                    ('graph_edge_tgt', 'edge_tgt'),
                    ('graph_sym', 'sym'),
                ]:
                    if key in graph_debug_info and graph_debug_info[key] is not None:
                        graph_parts.append(f"{label}={float(_to_scalar(graph_debug_info[key])):.4f}")
                if graph_parts:
                    debug_print("         graph_parts: " + ", ".join(graph_parts))
                structural = graph_debug_info.get('structural_violations', {})
                if isinstance(structural, dict):
                    debug_print(
                        "         structural: req_missing={}, forb_present={}".format(
                            structural.get('required_missing', 0),
                            structural.get('forbidden_present', 0),
                        )
                    )
                if 'weighted_adj_active_classes' in graph_debug_info:
                    val = graph_debug_info['weighted_adj_active_classes']
                    debug_print(f"         active_classes={float(_to_scalar(val)):.2f}")

        # Get loss components
        if 'seg_loss_components' in losses:
            dice_loss_total += losses['seg_loss_components'].get('dice', 0)
            ce_loss_total += losses['seg_loss_components'].get('ce', 0)
            focal_loss_total += losses['seg_loss_components'].get('focal', 0)

        # Get age prediction loss if available
        if 'age_loss' in losses:
            age_loss_total += losses['age_loss'].item()
        if 'volume_loss' in losses:
            volume_loss_total += losses['volume_loss'].item()

        # Backward pass
        if debug_active:
            debug_print("         Backpropagation start")
        losses['total'].backward()

        grad_norm_pre = compute_grad_norm(model.parameters()) if debug_active else 0.0
        if debug_active:
            debug_print(f"         Grad norm (pre-clip)={grad_norm_pre:.6f}")

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if debug_active:
            grad_norm_post = compute_grad_norm(model.parameters())
            debug_print(f"         Grad norm (post-clip)={grad_norm_post:.6f}")

        # Update weights
        optimizer.step()

        if debug_active:
            lr_val = optimizer.param_groups[0]['lr']
            debug_print(f"         Optimizer step complete, lr={lr_val:.6e}")
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                mem_max = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                debug_print(f"         CUDA memory alloc={mem_alloc:.2f} GB, max_alloc={mem_max:.2f} GB")

        # Update metrics
        total_loss += losses['total'].item()
        seg_loss_total += losses['seg_loss'].item()

        # Clean up
        del source_logits
        if target_logits is not None:
            del target_logits

        # Update progress bar
        if is_main and hasattr(pbar, 'set_postfix'):
            postfix_dict = {
                'loss': f"{losses['total'].item():.4f}",
                'seg': f"{losses['seg_loss'].item():.4f}",
            }
            if 'age_loss' in losses:
                postfix_dict['age'] = f"{losses['age_loss'].item():.4f}"
            if 'volume_loss' in losses:
                postfix_dict['vol'] = f"{losses['volume_loss'].item():.4f}"
            if 'graph_total' in losses:
                postfix_dict['graph'] = f"{losses['graph_total'].item():.4f}"
            if 'dyn_spec' in losses:
                postfix_dict['dyn'] = f"{losses['dyn_spec'].item():.4f}"
            if 'vrex_loss' in losses:
                postfix_dict['vrex'] = f"{losses['vrex_loss'].item():.4f}"
            if 'lapinv_loss' in losses:
                postfix_dict['lap'] = f"{losses['lapinv_loss'].item():.4f}"
            if 'counterfactual_loss' in losses:
                postfix_dict['cf'] = f"{losses['counterfactual_loss'].item():.4f}"
            pbar.set_postfix(postfix_dict)

        del losses

        # Periodically clear cache
        if i % 50 == 0:
            torch.cuda.empty_cache()

    # Average metrics
    avg_loss = total_loss / num_steps
    avg_seg_loss = seg_loss_total / num_steps
    avg_target_seg_loss = target_seg_loss_total / num_steps if current_target_weight > 0 else 0

    # Component losses
    avg_dice_loss = dice_loss_total / num_steps
    avg_ce_loss = ce_loss_total / num_steps
    avg_focal_loss = focal_loss_total / num_steps

    # Age-aware losses
    avg_age_loss = age_loss_total / num_steps
    avg_volume_loss = volume_loss_total / num_steps
    avg_shape_loss = shape_loss_total / num_steps
    avg_weighted_adj_loss = weighted_adj_loss_total / num_steps
    avg_vrex_loss = vrex_loss_total / max(1, vrex_steps) if vrex_steps > 0 else 0
    avg_lapinv_loss = lapinv_loss_total / max(1, lapinv_steps) if lapinv_steps > 0 else 0
    avg_cf_loss = cf_loss_total / max(1, cf_steps) if cf_steps > 0 else 0
    avg_age_balance_weight = (
        age_balance_weight_total / max(1, age_balance_batches)
        if age_balance_batches > 0 else 0
    )

    avg_graph_loss = graph_loss_total / num_steps if age_graph_loss is not None else 0
    avg_graph_spec_src = graph_spec_src_total / num_steps if age_graph_loss is not None else 0
    avg_graph_edge_src = graph_edge_src_total / num_steps if age_graph_loss is not None else 0
    avg_graph_spec_tgt = graph_spec_tgt_total / num_steps if age_graph_loss is not None else 0
    avg_graph_edge_tgt = graph_edge_tgt_total / num_steps if age_graph_loss is not None else 0
    avg_graph_sym = graph_sym_total / num_steps if age_graph_loss is not None else 0
    avg_forbidden_violations = forbidden_violations_total / num_steps if age_graph_loss is not None else 0
    avg_required_violations = required_violations_total / num_steps if age_graph_loss is not None else 0

    avg_dyn_spec = dyn_spec_loss_total / num_steps if (
                age_graph_loss is not None and epoch >= dyn_start_epoch and lambda_dyn_base > 0) else 0
    dyn_lambda_final = lambda_dyn_state['value'] if age_graph_loss is not None else 0

    # Synchronize metrics across processes if distributed
    if is_distributed:
        metrics_to_sync = torch.tensor([
            avg_loss, avg_seg_loss, avg_target_seg_loss, avg_dice_loss,
            avg_ce_loss, avg_focal_loss, avg_age_loss, avg_volume_loss,
            avg_shape_loss, avg_weighted_adj_loss, avg_vrex_loss,
            avg_lapinv_loss, avg_cf_loss, avg_age_balance_weight,
            avg_graph_loss, avg_graph_spec_src, avg_graph_edge_src,
            avg_graph_spec_tgt, avg_graph_edge_tgt, avg_graph_sym,
            avg_dyn_spec, avg_forbidden_violations, avg_required_violations
        ], device=device)

        dist.all_reduce(metrics_to_sync, op=dist.ReduceOp.SUM)
        metrics_to_sync /= world_size

        (avg_loss, avg_seg_loss, avg_target_seg_loss, avg_dice_loss,
         avg_ce_loss, avg_focal_loss, avg_age_loss, avg_volume_loss,
         avg_shape_loss, avg_weighted_adj_loss, avg_vrex_loss,
         avg_lapinv_loss, avg_cf_loss, avg_age_balance_weight,
         avg_graph_loss, avg_graph_spec_src, avg_graph_edge_src,
         avg_graph_spec_tgt, avg_graph_edge_tgt, avg_graph_sym,
         avg_dyn_spec, avg_forbidden_violations, avg_required_violations) = metrics_to_sync.tolist()

    # Log to tensorboard
    if writer and is_main:
        global_step = (epoch - 1) * num_steps
        writer.add_scalar('train/loss', avg_loss, global_step)
        writer.add_scalar('train/seg_loss', avg_seg_loss, global_step)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

        if current_target_weight > 0:
            writer.add_scalar('train/target_seg_loss', avg_target_seg_loss, global_step)
            writer.add_scalar('train/target_weight', current_target_weight, global_step)

        writer.add_scalar('train/dice_loss', avg_dice_loss, global_step)
        writer.add_scalar('train/ce_loss', avg_ce_loss, global_step)
        writer.add_scalar('train/focal_loss', avg_focal_loss, global_step)

        # Age-aware metrics
        writer.add_scalar('train/age_loss', avg_age_loss, global_step)
        writer.add_scalar('train/volume_loss', avg_volume_loss, global_step)
        writer.add_scalar('train/shape_loss', avg_shape_loss, global_step)
        writer.add_scalar('train/weighted_adj_loss', avg_weighted_adj_loss, global_step)
        if causal_cfg.get('enable_vrex', False):
            writer.add_scalar('train/vrex_loss', avg_vrex_loss, global_step)
        if causal_cfg.get('enable_lapinv', False):
            writer.add_scalar('train/lapinv_loss', avg_lapinv_loss, global_step)
        if causal_cfg.get('enable_counterfactual', False):
            writer.add_scalar('train/counterfactual_loss', avg_cf_loss, global_step)
        if causal_cfg.get('enable_age_balance', False):
            writer.add_scalar('train/age_balance_weight', avg_age_balance_weight, global_step)

        if age_graph_loss is not None:
            writer.add_scalar('train/graph_total', avg_graph_loss, global_step)
            writer.add_scalar('train/graph_spec_src', avg_graph_spec_src, global_step)
            writer.add_scalar('train/graph_edge_src', avg_graph_edge_src, global_step)
            writer.add_scalar('train/graph_spec_tgt', avg_graph_spec_tgt, global_step)
            writer.add_scalar('train/graph_edge_tgt', avg_graph_edge_tgt, global_step)
            writer.add_scalar('train/graph_sym', avg_graph_sym, global_step)
            writer.add_scalar('train/forbidden_violations', avg_forbidden_violations, global_step)
            writer.add_scalar('train/required_violations', avg_required_violations, global_step)
            writer.add_scalar('train/graph_warmup', age_graph_loss.get_warmup_factor(), global_step)

        if in_dynamic_stage and age_graph_loss is not None:
            writer.add_scalar('train/dyn_spec', avg_dyn_spec, global_step)
            writer.add_scalar('train/dyn_lambda', dyn_lambda_final, global_step)
            writer.add_scalar('train/dyn_k', dyn_k_effective, global_step)
            writer.add_scalar('train/dyn_conflicts', dyn_conflict_suppressions, global_step)

    elapsed = time.time() - start_time

    if is_main:
        print(f"\n✓ Epoch {epoch} completed in {elapsed:.1f}s")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Segmentation loss: {avg_seg_loss:.4f}")
        print(f"    - Dice: {avg_dice_loss:.4f}")
        print(f"    - CE: {avg_ce_loss:.4f}")
        print(f"    - Focal: {avg_focal_loss:.4f}")
        if current_target_weight > 0:
            print(f"  Target segmentation loss: {avg_target_seg_loss:.4f} (weight: {current_target_weight:.3f})")

        # Print age-aware losses
        print(f"  Age-aware losses:")
        if avg_age_loss > 0:
            print(f"    - Age prediction: {avg_age_loss:.4f}")
        if avg_volume_loss > 0:
            print(f"    - Volume consistency: {avg_volume_loss:.4f}")
        if avg_shape_loss > 0:
            print(f"    - Shape consistency: {avg_shape_loss:.4f}")
        if avg_weighted_adj_loss > 0:
            print(f"    - Weighted adjacency: {avg_weighted_adj_loss:.4f}")

        if age_graph_loss is not None:
            print(f"  Graph prior losses:")
            print(f"    - Total: {avg_graph_loss:.4f} (warmup: {age_graph_loss.get_warmup_factor():.3f})")
            align_mode = getattr(age_graph_loss, 'graph_align_mode', 'none')
            if align_mode in ['src_only', 'joint']:
                print(f"    - Source spectral: {avg_graph_spec_src:.4f}")
                print(f"    - Source edge: {avg_graph_edge_src:.4f}")
            if align_mode in ['tgt_only', 'joint']:
                print(f"    - Target spectral: {avg_graph_spec_tgt:.4f}")
                print(f"    - Target edge: {avg_graph_edge_tgt:.4f}")
            print(f"    - Symmetry: {avg_graph_sym:.4f}")
            print(f"    - Forbidden violations: {avg_forbidden_violations:.2f}")
            print(f"    - Required violations: {avg_required_violations:.2f}")

        if causal_cfg.get('enable_vrex', False) and vrex_steps > 0:
            print(f"  Causal regularizers:")
            print(f"    - V-REx (λ={causal_cfg.get('vrex_lambda', 0.0)}): {avg_vrex_loss:.4f}")
            if causal_cfg.get('enable_lapinv', False) and lapinv_steps > 0:
                print(f"    - Laplacian invariance (λ={causal_cfg.get('lapinv_lambda', 0.0)}): {avg_lapinv_loss:.4f}")
            if causal_cfg.get('enable_counterfactual', False) and cf_steps > 0:
                print(f"    - Counterfactual consistency (λ={causal_cfg.get('cf_lambda', 0.0)}): {avg_cf_loss:.4f}")
            if causal_cfg.get('enable_age_balance', False) and age_balance_batches > 0:
                print(f"    - Age balance weight mean: {avg_age_balance_weight:.4f}")

        if in_dynamic_stage and age_graph_loss is not None:
            print(f"  Dynamic spectral alignment:")
            print(f"    - Spec loss: {avg_dyn_spec:.4f}")
            print(f"    - λ_dyn (final): {dyn_lambda_final:.4f}")
            print(f"    - Conflicts suppressed: {dyn_conflict_suppressions}")

            # Clear cache at epoch end
    torch.cuda.empty_cache()

    # Return metrics
    return {
        'loss': avg_loss,
        'seg_loss': avg_seg_loss,
        'target_seg_loss': avg_target_seg_loss,
        'dice_loss': avg_dice_loss,
        'ce_loss': avg_ce_loss,
        'focal_loss': avg_focal_loss,
        'age_loss': avg_age_loss,
        'volume_loss': avg_volume_loss,
        'shape_loss': avg_shape_loss,
        'weighted_adj_loss': avg_weighted_adj_loss,
        'vrex_loss': avg_vrex_loss,
        'lapinv_loss': avg_lapinv_loss,
        'counterfactual_loss': avg_cf_loss,
        'age_balance_weight': avg_age_balance_weight,
        'graph_loss': avg_graph_loss,
        'graph_total': avg_graph_loss,
        'graph_spec_src': avg_graph_spec_src,
        'graph_edge_src': avg_graph_edge_src,
        'graph_spec_tgt': avg_graph_spec_tgt,
        'graph_edge_tgt': avg_graph_edge_tgt,
        'graph_sym': avg_graph_sym,
        'graph_struct': 0,
        'graph_spec': (avg_graph_spec_src + avg_graph_spec_tgt) / 2 if age_graph_loss is not None else 0,
        'graph_edge': (avg_graph_edge_src + avg_graph_edge_tgt) / 2 if age_graph_loss is not None else 0,
        'forbidden_violations': avg_forbidden_violations,
        'required_violations': avg_required_violations,
        'dyn_spec': avg_dyn_spec,
        'dyn_lambda': dyn_lambda_final,
        'dyn_conflicts': dyn_conflict_suppressions,
        'lr': optimizer.param_groups[0]['lr'],
        'epoch_time': elapsed
    }


def val_epoch_causal(model, loader, epoch, writer, args,
                     device=None, is_distributed=False, world_size=1, rank=0):
    """Validate one epoch with age-aware model"""

    is_main = (not is_distributed) or rank == 0
    debug_enabled = bool(getattr(args, 'debug_mode', False))
    debug_val_limit = max(1, getattr(args, 'debug_validate_samples', getattr(args, 'debug_step_limit', 2)))

    def debug_print(msg):
        if debug_enabled and is_main:
            print(f"[DEBUG][Val][Epoch {epoch}] {msg}", flush=True)

    model.eval()

    # Extract actual model from DDP if needed
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model

    # Metrics
    dice_metric = DiceMetric(
        include_background=True,
        reduction="mean",
        get_not_nans=False
    )

    # Per-class dice metric
    dice_metric_per_class = DiceMetric(
        include_background=True,
        reduction="mean_batch",
        get_not_nans=False
    )

    # Post-processing
    post_pred = AsDiscrete(argmax=True, to_onehot=actual_model.num_classes)

    total_loss = 0
    num_steps = len(loader)

    # Component loss tracking
    dice_loss_total = 0
    ce_loss_total = 0
    focal_loss_total = 0
    age_loss_total = 0

    if is_main:
        print(f"\n📊 Validation - Epoch {epoch}")

    # Enhanced inference settings
    use_tta = getattr(args, 'use_tta', True)
    infer_overlap = getattr(args, 'infer_overlap', 0.7)

    if is_main:
        print(f"  🔧 Enhanced Inference Settings:")
        print(f"    - TTA: {'Enabled (8 augmentations)' if use_tta else 'Disabled'}")
        print(f"    - Sliding Window Overlap: {infer_overlap}")
        print(f"    - Gaussian Blending: Enabled")
        if world_size > 1:
            print(f"    - Distributed validation across {world_size} GPUs")

    start_time = time.time()

    # Get loss configuration
    loss_config = getattr(args, 'loss_config', 'dice_ce')
    focal_gamma = getattr(args, 'focal_gamma', 2.0)

    # Create loss function
    if hasattr(actual_model, 'class_weights') and actual_model.class_weights is not None:
        seg_criterion = CombinedSegmentationLoss(
            loss_config=loss_config,
            include_background=True,
            focal_gamma=focal_gamma,
            class_weights=actual_model.class_weights,
            foreground_only=args.foreground_only
        )
    else:
        seg_criterion = CombinedSegmentationLoss(
            loss_config=loss_config,
            include_background=True,
            focal_gamma=focal_gamma,
            foreground_only=args.foreground_only
        )

    # Track predictions and labels
    sample_dice_scores = []

    # Class-wise prediction counts
    class_prediction_counts = np.zeros(actual_model.num_classes)
    class_label_counts = np.zeros(actual_model.num_classes)

    # Track inference time
    total_inference_time = 0
    inference_times = []

    # Progress bar (only on main process)
    pbar = tqdm(loader, desc="Validation") if is_main else loader

    # Setup mixed precision if enabled
    use_amp = getattr(args, 'use_amp', True)
    amp_dtype = torch.bfloat16 if getattr(args, 'amp_dtype', 'bfloat16') == 'bfloat16' else torch.float16

    # Track ages for analysis
    all_ages = []

    with torch.no_grad():
        for i, batch_data in enumerate(pbar):
            data = batch_data["image"].to(device, non_blocking=True)
            target = batch_data["label"].to(device, non_blocking=True)
            age = batch_data.get("age", torch.tensor([40.0]).repeat(data.shape[0], 1)).to(device)

            all_ages.append(age.cpu().numpy())

            debug_active = debug_enabled and (i < debug_val_limit)
            if debug_active:
                debug_print(
                    f"Sample {i}: data shape={tuple(data.shape)}, label shape={tuple(target.shape)}"
                )
                debug_print(
                    f"           image stats min={float(data.min().item()):.3f}, max={float(data.max().item()):.3f}, "
                    f"mean={float(data.mean().item()):.3f}, std={float(data.std().item()):.3f}"
                )
                debug_print(
                    f"           age min={float(age.min().item()):.2f}, max={float(age.max().item()):.2f}"
                )
                tgt_unique = torch.unique(target)
                debug_print(
                    f"           target unique={tgt_unique.numel()} (min={int(tgt_unique.min().item())}, max={int(tgt_unique.max().item())})"
                )

            # Fix label dimensions
            if len(target.shape) == 5 and target.shape[1] == 1:
                target = target.squeeze(1)

            # Enhanced sliding window inference
            roi_size = (args.roi_x, args.roi_y, args.roi_z)
            sw_batch_size = args.sw_batch_size

            # Time the inference
            infer_start = time.time()

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    # Simple inference without TTA for speed in validation
                    output = sliding_window_inference(
                        data,
                        roi_size,
                        sw_batch_size,
                        lambda x: model(x, age),  # Pass age to model
                        overlap=infer_overlap,
                        mode="gaussian",
                        sigma_scale=0.125,
                        padding_mode="constant",
                        cval=0.0
                    )
            else:
                output = sliding_window_inference(
                    data,
                    roi_size,
                    sw_batch_size,
                    lambda x: model(x, age),  # Pass age to model
                    overlap=infer_overlap,
                    mode="gaussian",
                    sigma_scale=0.125,
                    padding_mode="constant",
                    cval=0.0
                )

            infer_time = time.time() - infer_start
            total_inference_time += infer_time
            inference_times.append(infer_time)

            if debug_active:
                debug_print(f"           inference time={infer_time:.3f}s")
                debug_print(
                    f"           output stats mean={float(output.mean().item()):.4f}, std={float(output.std().item()):.4f}, "
                    f"min={float(output.min().item()):.4f}, max={float(output.max().item()):.4f}"
                )

            # Compute loss
            loss, loss_components = seg_criterion(output, target)
            total_loss += loss.item()
            dice_loss_total += loss_components['dice']
            ce_loss_total += loss_components['ce']
            focal_loss_total += loss_components['focal']

            if debug_active:
                debug_print(
                    f"           loss={loss.item():.6f}, dice={loss_components['dice']:.6f}, "
                    f"ce={loss_components['ce']:.6f}, focal={loss_components['focal']:.6f}"
                )

            # For metrics computation
            output_one_hot = [post_pred(i) for i in decollate_batch(output)]

            target_one_hot = []
            masks = []
            for t in decollate_batch(target):  # t: (X,Y,Z) with values in {-1, 0..86}
                mask = (t != -1)  # True = foreground voxel
                t_shift = (t + 1).clamp(min=0)  # {-1,0..86} -> {0..87}
                oh = F.one_hot(t_shift.long(),  # (X,Y,Z, 88)  注意：num_classes = 87 -> +1
                               num_classes=actual_model.num_classes + 1)
                oh = oh.permute(3, 0, 1, 2).float()  # -> (88, X, Y, Z)
                oh = oh[1:, ...]  # drop 背景槽 -> (87, X, Y, Z)
                target_one_hot.append(oh)
                masks.append(mask)

            # 背景体素不计分：把预测、标签在背景处都清零
            for j in range(len(output_one_hot)):
                m = masks[j].unsqueeze(0).float()
                output_one_hot[j] = output_one_hot[j] * m
                target_one_hot[j] = target_one_hot[j] * m

            # Compute dice
            dice_metric(y_pred=output_one_hot, y=target_one_hot)
            dice_metric_per_class(y_pred=output_one_hot, y=target_one_hot)

            if debug_active:
                sample_dice = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
                sample_dice(y_pred=[output_one_hot[0]], y=[target_one_hot[0]])
                sample_score = sample_dice.aggregate().item()
                debug_print(f"           sample dice (first volume)={sample_score:.4f}")
                sample_dice.reset()

            # Count predictions per class
            pred = torch.argmax(output, dim=1)
            pred_array = pred.detach()
            target_array = target.detach()
            valid_pixels = (target_array != -1)

            for c in range(actual_model.num_classes):
                pred_np = pred_array.cpu().numpy()
                target_np = target_array.cpu().numpy()
                valid_np = valid_pixels.cpu().numpy()

                class_prediction_counts[c] += np.sum((pred_np[valid_np] == c))
                class_label_counts[c] += np.sum((target_np[valid_np] == c))

            # Update progress bar
            if is_main and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'infer_time': f"{infer_time:.2f}s"})

    # Aggregate metrics
    dice_score = dice_metric.aggregate().item()
    dice_per_class = dice_metric_per_class.aggregate()

    avg_loss = total_loss / num_steps
    avg_dice_loss = dice_loss_total / num_steps
    avg_ce_loss = ce_loss_total / num_steps
    avg_focal_loss = focal_loss_total / num_steps
    avg_age_loss = age_loss_total / num_steps

    # Synchronize metrics across processes if distributed
    if is_distributed:
        # Convert metrics to tensors for synchronization
        metrics_tensor = torch.tensor([
            dice_score, avg_loss, avg_dice_loss, avg_ce_loss,
            avg_focal_loss, avg_age_loss
        ], device=device)

        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        metrics_tensor /= world_size

        (dice_score, avg_loss, avg_dice_loss, avg_ce_loss,
         avg_focal_loss, avg_age_loss) = metrics_tensor.tolist()

        # Synchronize per-class dice scores
        if dice_per_class is not None:
            dist.all_reduce(dice_per_class, op=dist.ReduceOp.SUM)
            dice_per_class /= world_size

    # Calculate average inference time
    avg_inference_time = np.mean(inference_times)
    total_val_time = time.time() - start_time

    # Calculate worst 10 classes dice average
    worst_10_dice_avg = 0.0
    if dice_per_class is not None and dice_per_class.numel() > 0:
        dice_per_class_np = dice_per_class.cpu().numpy()
        sorted_dice = np.sort(dice_per_class_np)
        worst_10_dice_avg = np.mean(sorted_dice[:min(10, len(sorted_dice))])

    # Print results (only on main process)
    if is_main:
        # Age statistics
        all_ages_np = np.concatenate(all_ages)
        print(f"\n📊 Validation Results:")
        print(f"  Age range in validation: [{all_ages_np.min():.1f}, {all_ages_np.max():.1f}] weeks")
        print(f"  Overall Dice Score: {dice_score:.4f}")
        print(f"  Worst 10 Classes Avg Dice: {worst_10_dice_avg:.4f}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"    - Dice: {avg_dice_loss:.4f}")
        print(f"    - CE: {avg_ce_loss:.4f}")
        print(f"    - Focal: {avg_focal_loss:.4f}")

        print(f"\n⏱️ Performance Metrics:")
        print(f"  Total validation time: {total_val_time:.1f}s")
        print(f"  Average inference time per sample: {avg_inference_time:.2f}s")
        print(f"  Total samples: {num_steps}")

        # Print per-class dice scores
        if dice_per_class is not None and dice_per_class.numel() > 0:
            dice_per_class_np = dice_per_class.cpu().numpy()
            print(f"\n📊 Per-class Dice Scores:")

            class_indices = list(range(actual_model.num_classes))
            class_dices = dice_per_class_np

            sorted_indices = np.argsort(class_dices)

            print(f"  Worst 10 classes:")
            for idx, i in enumerate(sorted_indices[:min(10, len(sorted_indices))]):
                actual_class = class_indices[i]
                original_class = actual_class + 1 if args.foreground_only else actual_class
                print(f"    {idx + 1}. Class {actual_class} (orig {original_class}): {class_dices[i]:.4f}")

            print(f"  Best 5 classes:")
            for idx, i in enumerate(sorted_indices[-min(5, len(sorted_indices)):]):
                actual_class = class_indices[i]
                original_class = actual_class + 1 if args.foreground_only else actual_class
                print(f"    {idx + 1}. Class {actual_class} (orig {original_class}): {class_dices[i]:.4f}")

    # Reset metrics
    dice_metric.reset()
    dice_metric_per_class.reset()

    # Log to tensorboard (only on main process)
    if writer and is_main:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/dice', dice_score, epoch)
        writer.add_scalar('val/dice_loss', avg_dice_loss, epoch)
        writer.add_scalar('val/ce_loss', avg_ce_loss, epoch)
        writer.add_scalar('val/focal_loss', avg_focal_loss, epoch)
        writer.add_scalar('val/worst_10_dice_avg', worst_10_dice_avg, epoch)
        writer.add_scalar('val/inference_time_avg', avg_inference_time, epoch)

        if dice_per_class is not None and dice_per_class.numel() > 0:
            for c in range(actual_model.num_classes):
                writer.add_scalar(f'val/dice_class_{c}', dice_per_class_np[c], epoch)

    if is_main:
        elapsed = time.time() - start_time
        print(f"\n✓ Validation completed in {elapsed:.1f}s")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Dice Score: {dice_score:.4f}")
        print(f"  Worst 10 Classes Avg: {worst_10_dice_avg:.4f}")

    # Return metrics
    val_metrics = {
        'loss': avg_loss,
        'dice': dice_score,
        'dice_loss': avg_dice_loss,
        'ce_loss': avg_ce_loss,
        'focal_loss': avg_focal_loss,
        'age_loss': avg_age_loss,
        'dice_per_class': dice_per_class_np.tolist() if dice_per_class is not None else None,
        'sample_dice_scores': sample_dice_scores,
        'class_prediction_counts': class_prediction_counts.tolist(),
        'class_label_counts': class_label_counts.tolist(),
        'val_time': total_val_time,
        'worst_10_dice_avg': worst_10_dice_avg,
        'avg_inference_time': avg_inference_time,
        'total_inference_time': total_inference_time,
    }

    return val_metrics


def save_checkpoint_simplified(model, optimizer, epoch, best_acc, args, filepath,
                               dice_history=None, additional_info=None):
    """Save simplified checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'base_model_state_dict': model.base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'args': args,
        'num_classes': model.num_classes,
        'foreground_only': model.foreground_only,
        'enhanced_class_weights': model.enhanced_class_weights,
        'loss_config': getattr(args, 'loss_config', 'dice_focal'),
        'focal_gamma': getattr(args, 'focal_gamma', 2.0),
        'dice_history': dice_history if dice_history is not None else {},
    }

    # Add additional info if provided
    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")