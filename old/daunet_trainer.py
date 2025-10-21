import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
import time
from typing import Dict, Optional, List
from tqdm import tqdm
import psutil
import GPUtil
from scipy.ndimage import distance_transform_edt
import json
import os
from graph_prior_loss import GraphPriorLoss
from collections import deque
from graph_prior_loss import soft_adjacency_from_probs, compute_laplacian


def is_dist():
    return dist.is_initialized()


def dist_mean_scalar(x: torch.Tensor):
    if is_dist():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x


def _load_lr_pairs_for_val(args):
    """Load laterality pairs for validation/inference"""
    pairs = []
    if getattr(args, 'laterality_pairs_json', None) and os.path.exists(args.laterality_pairs_json):
        with open(args.laterality_pairs_json, 'r') as f:
            raw = json.load(f)  # Expected format: [[17,18],[36,37],...]
        for a, b in raw:
            # If foreground_only: map 1..87 -> 0..86
            if args.foreground_only:
                a_idx = a - 1
                b_idx = b - 1
            else:
                a_idx = a
                b_idx = b
            pairs.append((a_idx, b_idx))
    return pairs


def _swap_channels_for_laterality(t, pairs):
    """Swap channels in prediction tensor for laterality pairs
    t: (B, C, X, Y, Z) logits/probs tensor
    pairs: list of (a, b) channel indices to swap
    """
    if not pairs:
        return t

    out = t.clone()
    for a, b in pairs:
        # Swap channels a and b
        temp = out[:, a].clone()
        out[:, a] = out[:, b]
        out[:, b] = temp

    return out


class TopKDiceLoss(nn.Module):
    def __init__(self, topk_ratio=0.3, include_background=True, squared_pred=True, warmup_epochs=30, current_epoch=0):
        super().__init__()
        self.max_topk_ratio = topk_ratio
        self.include_background = include_background
        self.squared_pred = squared_pred
        self.warmup_epochs = warmup_epochs
        self.current_epoch = current_epoch

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def get_current_topk_ratio(self):
        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            return 0.1 + (self.max_topk_ratio - 0.1) * progress
        else:
            return self.max_topk_ratio

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        batch_size = pred.shape[0]
        num_classes = pred.shape[1]
        topk_ratio = self.get_current_topk_ratio()
        pred_flat = pred.view(batch_size, num_classes, -1)
        target_flat = target.view(batch_size, num_classes, -1)
        dice_losses = []
        for b in range(batch_size):
            batch_dice_losses = []
            for c in range(num_classes):
                if not self.include_background and c == 0:
                    continue
                p = pred_flat[b, c]
                t = target_flat[b, c]
                if t.sum() == 0:
                    continue
                error = torch.abs(p - t)
                num_voxels = error.shape[0]
                k = max(1, int(num_voxels * topk_ratio))
                topk_values, topk_indices = torch.topk(error, k, largest=True)
                p_topk = p[topk_indices]
                t_topk = t[topk_indices]
                if self.squared_pred:
                    intersection = (p_topk * p_topk * t_topk).sum()
                    union = (p_topk * p_topk).sum() + t_topk.sum()
                else:
                    intersection = (p_topk * t_topk).sum()
                    union = p_topk.sum() + t_topk.sum()
                dice = (2 * intersection + 1e-5) / (union + 1e-5)
                batch_dice_losses.append(1 - dice)
            if batch_dice_losses:
                dice_losses.append(torch.stack(batch_dice_losses).mean())
        if dice_losses:
            return torch.stack(dice_losses).mean()
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, dice_weight=0.0, ce_weight=0.0, focal_weight=0.0, include_background=True, focal_gamma=2.0,
                 class_weights=None, foreground_only=False, loss_config="dice_ce", use_topk_dice=False, topk_ratio=0.3,
                 topk_warmup_epochs=30, current_epoch=0):
        super().__init__()
        self.foreground_only = foreground_only
        self.num_classes = 87 if foreground_only else 88
        self.loss_config = loss_config
        self.use_topk_dice = use_topk_dice
        self.topk_ratio = topk_ratio
        self.current_epoch = current_epoch
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
            if use_topk_dice:
                self.dice_loss = TopKDiceLoss(topk_ratio=topk_ratio, include_background=True, squared_pred=True,
                                              warmup_epochs=topk_warmup_epochs, current_epoch=current_epoch)
            else:
                self.dice_loss = DiceLoss(to_onehot_y=False, softmax=True, include_background=True, squared_pred=True,
                                          reduction="mean")
        if self.ce_weight > 0:
            if class_weights is not None:
                self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
            else:
                self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if self.focal_weight > 0:
            self.focal_loss = FocalLoss(include_background=True, to_onehot_y=False, gamma=focal_gamma,
                                        weight=class_weights if class_weights is not None else None, reduction="mean")

    def update_epoch(self, epoch):
        self.current_epoch = epoch
        if self.use_topk_dice and hasattr(self.dice_loss, 'update_epoch'):
            self.dice_loss.update_epoch(epoch)

    def _create_one_hot_with_ignore(self, labels, num_classes):
        valid_mask = labels != -1
        labels_for_onehot = labels.clone()
        labels_for_onehot[~valid_mask] = 0
        one_hot = F.one_hot(labels_for_onehot.long(), num_classes=num_classes)
        one_hot = one_hot.permute(0, 4, 1, 2, 3)
        # Don't force to float32, keep the dtype flexible
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

        # Use the same dtype as pred (important for AMP)
        one_hot_target = one_hot_target.to(dtype=pred.dtype)
        mask = valid_mask.unsqueeze(1).to(dtype=pred.dtype)

        if self.use_topk_dice:
            pred_for_dice = pred
        else:
            pred_for_dice = pred * mask

        if self.dice_weight > 0:
            if self.use_topk_dice:
                dice = self.dice_loss(pred_for_dice, one_hot_target)
            else:
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


def compute_gradient_statistics(model) -> Dict[str, float]:
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
    grad_stats = {'grad_norm_total': 0.0, 'grad_norm_base': 0.0, 'grad_max': 0.0, 'grad_min': 1e10, 'num_zero_grad': 0,
                  'num_params_with_grad': 0}
    base_grads = []
    for name, param in actual_model.base_model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            base_grads.append(grad_norm)
            grad_stats['grad_max'] = max(grad_stats['grad_max'], param.grad.data.max().item())
            grad_stats['grad_min'] = min(grad_stats['grad_min'], param.grad.data.min().item())
            grad_stats['num_params_with_grad'] += 1
            if grad_norm < 1e-8:
                grad_stats['num_zero_grad'] += 1
    if base_grads:
        grad_stats['grad_norm_base'] = np.mean(base_grads)
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    grad_stats['grad_norm_total'] = total_norm ** 0.5
    return grad_stats


def compute_weight_statistics(model) -> Dict[str, float]:
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
    weight_stats = {'weight_norm_total': 0.0, 'weight_norm_base': 0.0, 'weight_mean': 0.0, 'weight_std': 0.0,
                    'num_dead_neurons': 0}
    all_weights = []
    base_weights = []
    for name, param in actual_model.base_model.named_parameters():
        if 'weight' in name:
            weights = param.data.cpu().numpy().flatten()
            base_weights.extend(weights)
            all_weights.extend(weights)
            if len(param.shape) >= 2:
                neuron_norms = param.data.norm(2, dim=0)
                weight_stats['num_dead_neurons'] += (neuron_norms < 1e-6).sum().item()
    if all_weights:
        weight_stats['weight_mean'] = np.mean(all_weights)
        weight_stats['weight_std'] = np.std(all_weights)
    if base_weights:
        weight_stats['weight_norm_base'] = np.linalg.norm(base_weights)
    total_norm = 0
    for p in model.parameters():
        total_norm += p.data.norm(2).item() ** 2
    weight_stats['weight_norm_total'] = total_norm ** 0.5
    return weight_stats


def compute_activation_statistics(activations: List[torch.Tensor]) -> Dict[str, float]:
    if not activations:
        return {}
    stats = {'activation_mean': 0.0, 'activation_std': 0.0, 'activation_sparsity': 0.0, 'activation_max': 0.0,
             'activation_min': 0.0}
    all_acts = []
    for act in activations:
        t = act.detach()
        if t.is_cuda:
            t = t.cpu()
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.float()
        acts = t.numpy().ravel()
        all_acts.extend(acts)
    if all_acts:
        all_acts = np.array(all_acts)
        stats['activation_mean'] = np.mean(all_acts)
        stats['activation_std'] = np.std(all_acts)
        stats['activation_sparsity'] = (np.abs(all_acts) < 0.01).mean()
        stats['activation_max'] = np.max(all_acts)
        stats['activation_min'] = np.min(all_acts)
    return stats


def get_system_stats(rank=0) -> Dict[str, float]:
    stats = {}
    stats['cpu_percent'] = psutil.cpu_percent()
    stats['memory_percent'] = psutil.virtual_memory().percent
    try:
        gpus = GPUtil.getGPUs()
        if gpus and rank < len(gpus):
            gpu = gpus[rank]
            stats['gpu_memory_used_gb'] = gpu.memoryUsed / 1024
            stats['gpu_memory_total_gb'] = gpu.memoryTotal / 1024
            stats['gpu_utilization'] = gpu.load * 100
            stats['gpu_temperature'] = gpu.temperature
    except:
        pass
    if torch.cuda.is_available():
        stats['pytorch_allocated_gb'] = torch.cuda.memory_allocated(rank) / 1024 ** 3
        stats['pytorch_reserved_gb'] = torch.cuda.memory_reserved(rank) / 1024 ** 3
    return stats


def train_epoch_simplified(model, source_loader, target_loader, optimizer, epoch, total_epochs, writer, args,
                           device=None, is_distributed=False, world_size=1, rank=0, graph_loss=None):
    """Training epoch with DUAL-BRANCH graph alignment: prior + dynamic spectral

    Args:
        graph_loss: Pre-initialized GraphPriorLoss object (already on correct device)
    """

    is_main = (not is_distributed) or rank == 0

    # Update epoch for graph loss if available
    if graph_loss is not None:
        graph_loss.set_epoch(epoch)

        if is_main:
            print(f"\nðŸ§  Using Dual-Branch GraphPriorLoss")
            print(f"  Prior branch: structural anchoring")
            if epoch >= graph_loss.dyn_start_epoch:
                print(f"  Dynamic branch: ACTIVE (epoch {epoch})")
            align_mode = graph_loss.graph_align_mode
            print(f"  âœ“ Graph alignment mode: '{align_mode}'")
            print(f"  Current epoch: {epoch}, Warmup factor: {graph_loss.get_warmup_factor():.3f}")

    # Track structural violations for conflict gating
    structural_violations_tracker = deque(maxlen=5)  # Track last 5 epochs

    model.train()
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model

    # Loss accumulators
    total_loss = 0
    seg_loss_total = 0
    target_seg_loss_total = 0
    dice_loss_total = 0
    ce_loss_total = 0
    focal_loss_total = 0

    # Graph loss accumulators (prior branch)
    graph_loss_total = 0
    graph_spec_src_total = 0
    graph_edge_src_total = 0
    graph_spec_tgt_total = 0
    graph_edge_tgt_total = 0
    graph_sym_total = 0

    # NEW: Dynamic spectral loss accumulator
    dyn_spec_loss_total = 0
    dyn_conflict_suppressions = 0

    # Track forbidden violations for monitoring
    forbidden_violations_total = 0
    required_violations_total = 0

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    num_steps = min(len(source_loader), len(target_loader))

    if is_main:
        print(f"\nðŸš€ Training - Epoch {epoch}/{total_epochs}")
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
            print(f"  ðŸ”Š Using target labels with weight: {current_target_weight:.3f}")
    else:
        current_target_weight = 0.0

    # Get dynamic alignment parameters
    if graph_loss is not None:
        dyn_start_epoch = graph_loss.dyn_start_epoch
        dyn_ramp_epochs = graph_loss.dyn_ramp_epochs
        dyn_top_k_max = graph_loss.dyn_top_k
        lambda_dyn_base = graph_loss.lambda_dyn
        use_restricted_mask = graph_loss.use_restricted_mask

        # Get R_mask from graph_loss if available
        R_mask = graph_loss.R_mask if hasattr(graph_loss, 'R_mask') and use_restricted_mask else None

        # Sanity check for R_mask
        if is_main and R_mask is not None:
            print(f"  ðŸ“Š R_mask loaded: shape={R_mask.shape}, sum={R_mask.sum():.0f}, device={R_mask.device}")
    else:
        dyn_start_epoch = 50
        dyn_ramp_epochs = 50
        dyn_top_k_max = 12
        lambda_dyn_base = 0.2
        use_restricted_mask = False
        R_mask = None

    # Check if we're in dynamic alignment stage
    in_dynamic_stage = epoch >= dyn_start_epoch

    # Dynamic alignment ramp factor
    if in_dynamic_stage:
        ramp_progress = min(1.0, (epoch - dyn_start_epoch + 1) / dyn_ramp_epochs)
        lambda_dyn_effective = lambda_dyn_base * ramp_progress
        # ENHANCED: Gradual k growth
        dyn_k_effective = max(4, int(round(4 + (dyn_top_k_max - 4) * ramp_progress)))
        if is_main:
            print(
                f"  ðŸ”„ Dynamic spectral: Î»={lambda_dyn_effective:.3f}, k={dyn_k_effective} (ramp {ramp_progress * 100:.0f}%)")
    else:
        lambda_dyn_effective = 0.0
        dyn_k_effective = 0

    # Create segmentation loss
    from daunet_trainer import CombinedSegmentationLoss

    loss_config = getattr(args, 'loss_config', 'dice_focal')
    focal_gamma = getattr(args, 'focal_gamma', 2.0)
    use_topk = getattr(args, 'use_topk_dice', False)
    topk_ratio = getattr(args, 'topk_ratio', 0.3)
    topk_warmup = getattr(args, 'topk_warmup_epochs', 30)

    if hasattr(actual_model, 'class_weights') and actual_model.class_weights is not None:
        seg_criterion = CombinedSegmentationLoss(
            loss_config=loss_config, include_background=True,
            focal_gamma=focal_gamma, class_weights=actual_model.class_weights,
            foreground_only=args.foreground_only, use_topk_dice=use_topk,
            topk_ratio=topk_ratio, topk_warmup_epochs=topk_warmup,
            current_epoch=epoch
        )
    else:
        seg_criterion = CombinedSegmentationLoss(
            loss_config=loss_config,
            include_background=True,
            focal_gamma=focal_gamma,
            foreground_only=args.foreground_only,
            use_topk_dice=use_topk,
            topk_ratio=topk_ratio,
            topk_warmup_epochs=topk_warmup,
            current_epoch=epoch
        )

    seg_criterion.update_epoch(epoch)

    start_time = time.time()
    initial_system_stats = get_system_stats(rank)
    activations = []

    # Progress bar
    pbar = tqdm(range(num_steps), desc=f"Epoch {epoch}") if is_main else range(num_steps)

    # Mixed precision setup
    use_amp = getattr(args, 'use_amp', True)
    amp_dtype = torch.bfloat16 if getattr(args, 'amp_dtype', 'bfloat16') == 'bfloat16' else torch.float16

    for i in pbar:
        step_start_time = time.time()

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

        # Extract data
        source_images = source_batch["image"].to(device, non_blocking=True)
        source_labels = source_batch["label"].to(device, non_blocking=True)
        target_images = target_batch["image"].to(device, non_blocking=True)
        target_labels = target_batch["label"].to(device, non_blocking=True)

        # Fix label dimensions
        if len(source_labels.shape) == 5 and source_labels.shape[1] == 1:
            source_labels = source_labels.squeeze(1)
        if len(target_labels.shape) == 5 and target_labels.shape[1] == 1:
            target_labels = target_labels.squeeze(1)

        optimizer.zero_grad()

        global_step = (epoch - 1) * num_steps + i

        # Forward pass with mixed precision
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                # Compute source segmentation loss
                losses = actual_model.compute_losses(
                    source_images,
                    source_labels,
                    seg_criterion,
                    step=global_step
                )

                # Extract logits
                source_logits = losses.pop('logits')

                # Compute target segmentation loss if enabled
                target_logits = None
                if current_target_weight > 0:
                    target_logits = actual_model.forward(target_images)
                    target_seg_loss, target_seg_components = seg_criterion(target_logits, target_labels)
                    losses['target_seg_loss'] = target_seg_loss
                    losses['total'] = losses['total'] + current_target_weight * target_seg_loss
                    target_seg_loss_total += target_seg_loss.item()

                # PRIOR BRANCH: Apply graph prior loss
                if graph_loss is not None:
                    if target_logits is not None:
                        graph_total, graph_dict = graph_loss(target_logits, target_labels)
                    else:
                        graph_total, graph_dict = graph_loss(source_logits, source_labels)

                    losses['total'] = losses['total'] + graph_total
                    losses['graph_total'] = graph_total

                    # Accumulate prior branch losses
                    graph_loss_total += graph_total.item()
                    graph_spec_src_total += graph_dict['graph_spec_src'].item()
                    graph_edge_src_total += graph_dict['graph_edge_src'].item()
                    graph_spec_tgt_total += graph_dict['graph_spec_tgt'].item()
                    graph_edge_tgt_total += graph_dict['graph_edge_tgt'].item()
                    graph_sym_total += graph_dict['graph_sym'].item()

                    # FIXED: Track structural violations from dictionary
                    current_violations = graph_dict.get('structural_violations', {})
                    if isinstance(current_violations, dict):
                        forbidden_count = current_violations.get('forbidden_present', 0)
                        required_count = current_violations.get('required_missing', 0)
                        # IMPROVED: Use weighted sum for violation tracking
                        weighted_violations = forbidden_count * 1.5 + required_count
                        structural_violations_tracker.append(weighted_violations)
                        forbidden_violations_total += forbidden_count
                        required_violations_total += required_count

                # DYNAMIC BRANCH: Dynamic spectral alignment
                if in_dynamic_stage and source_logits is not None and target_logits is not None:
                    from graph_prior_loss import soft_adjacency_from_probs, compute_laplacian

                    # Convert to probabilities
                    P_s = torch.softmax(source_logits, dim=1)
                    P_t = torch.softmax(target_logits, dim=1)

                    # Compute soft adjacencies with pooling for efficiency
                    pool_cfg = {'kernel_size': 3, 'stride': 2}
                    temperature = getattr(graph_loss, 'temperature', 1.0) if graph_loss else 1.0

                    A_s = soft_adjacency_from_probs(P_s, temperature=temperature, **pool_cfg, restricted_mask=R_mask)
                    A_t = soft_adjacency_from_probs(P_t, temperature=temperature, **pool_cfg, restricted_mask=R_mask)

                    # Compute Laplacians
                    L_s = compute_laplacian(A_s, normalized=True)
                    L_t = compute_laplacian(A_t, normalized=True)

                    # Low-frequency spectral alignment
                    L_s_sym = 0.5 * (L_s + L_s.T)
                    L_t_sym = 0.5 * (L_t + L_t.T)

                    eig_s, _ = torch.linalg.eigh(L_s_sym.float())
                    eig_t, _ = torch.linalg.eigh(L_t_sym.float())

                    # Convert back to original dtype
                    eig_s = eig_s.to(source_logits.dtype)
                    eig_t = eig_t.to(target_logits.dtype)

                    # FIXED: Use correct slicing syntax
                    # ENHANCED: Use gradually increasing k
                    k = min(dyn_k_effective, eig_s.shape[-1] - 1)
                    L_dyn = F.smooth_l1_loss(eig_s[1:k + 1], eig_t[1:k + 1])

                    # IMPROVED: Conflict gating with weighted violations
                    if len(structural_violations_tracker) >= 3:
                        recent_violations = list(structural_violations_tracker)[-3:]
                        if recent_violations[-1] > recent_violations[0] * 1.2:  # 20% increase
                            lambda_dyn_effective *= 0.5
                            dyn_conflict_suppressions += 1
                            if is_main and i == 0:
                                print(f"  âš ï¸ Conflict detected: reducing dynamic weight to {lambda_dyn_effective:.3f}")

                    # Add to total loss
                    losses['dyn_spec'] = L_dyn
                    losses['total'] = losses['total'] + lambda_dyn_effective * L_dyn
                    dyn_spec_loss_total += L_dyn.item()
        else:
            # FIXED: Non-AMP version with complete dynamic branch implementation
            # Compute source segmentation loss
            losses = actual_model.compute_losses(
                source_images,
                source_labels,
                seg_criterion,
                step=global_step
            )

            # Extract logits
            source_logits = losses.pop('logits')

            # Compute target segmentation loss if enabled
            target_logits = None
            if current_target_weight > 0:
                target_logits = actual_model.forward(target_images)
                target_seg_loss, target_seg_components = seg_criterion(target_logits, target_labels)
                losses['target_seg_loss'] = target_seg_loss
                losses['total'] = losses['total'] + current_target_weight * target_seg_loss
                target_seg_loss_total += target_seg_loss.item()

            # PRIOR BRANCH: Apply graph prior loss
            if graph_loss is not None:
                if target_logits is not None:
                    graph_total, graph_dict = graph_loss(target_logits, target_labels)
                else:
                    graph_total, graph_dict = graph_loss(source_logits, source_labels)

                losses['total'] = losses['total'] + graph_total
                losses['graph_total'] = graph_total

                # Accumulate prior branch losses
                graph_loss_total += graph_total.item()
                graph_spec_src_total += graph_dict['graph_spec_src'].item()
                graph_edge_src_total += graph_dict['graph_edge_src'].item()
                graph_spec_tgt_total += graph_dict['graph_spec_tgt'].item()
                graph_edge_tgt_total += graph_dict['graph_edge_tgt'].item()
                graph_sym_total += graph_dict['graph_sym'].item()

                # Track structural violations from dictionary
                current_violations = graph_dict.get('structural_violations', {})
                if isinstance(current_violations, dict):
                    forbidden_count = current_violations.get('forbidden_present', 0)
                    required_count = current_violations.get('required_missing', 0)
                    # Use weighted sum for violation tracking
                    weighted_violations = forbidden_count * 1.5 + required_count
                    structural_violations_tracker.append(weighted_violations)
                    forbidden_violations_total += forbidden_count
                    required_violations_total += required_count

            # DYNAMIC BRANCH: Dynamic spectral alignment (NON-AMP)
            if in_dynamic_stage and source_logits is not None and target_logits is not None:
                from graph_prior_loss import soft_adjacency_from_probs, compute_laplacian

                # Convert to probabilities
                P_s = torch.softmax(source_logits, dim=1)
                P_t = torch.softmax(target_logits, dim=1)

                # Compute soft adjacencies with pooling for efficiency
                pool_cfg = {'kernel_size': 3, 'stride': 2}
                temperature = getattr(graph_loss, 'temperature', 1.0) if graph_loss else 1.0

                A_s = soft_adjacency_from_probs(P_s, temperature=temperature, **pool_cfg, restricted_mask=R_mask)
                A_t = soft_adjacency_from_probs(P_t, temperature=temperature, **pool_cfg, restricted_mask=R_mask)

                # Compute Laplacians
                L_s = compute_laplacian(A_s, normalized=True)
                L_t = compute_laplacian(A_t, normalized=True)

                # Low-frequency spectral alignment
                L_s_sym = 0.5 * (L_s + L_s.T)
                L_t_sym = 0.5 * (L_t + L_t.T)

                eig_s, _ = torch.linalg.eigh(L_s_sym.float())
                eig_t, _ = torch.linalg.eigh(L_t_sym.float())

                # Convert back to original dtype
                eig_s = eig_s.to(source_logits.dtype)
                eig_t = eig_t.to(target_logits.dtype)

                # FIXED: Use correct slicing syntax
                # Use gradually increasing k
                k = min(dyn_k_effective, eig_s.shape[-1] - 1)
                L_dyn = F.smooth_l1_loss(eig_s[1:k + 1], eig_t[1:k + 1])

                # Conflict gating with weighted violations
                if len(structural_violations_tracker) >= 3:
                    recent_violations = list(structural_violations_tracker)[-3:]
                    if recent_violations[-1] > recent_violations[0] * 1.2:  # 20% increase
                        lambda_dyn_effective *= 0.5
                        dyn_conflict_suppressions += 1
                        if is_main and i == 0:
                            print(f"  âš ï¸ Conflict detected: reducing dynamic weight to {lambda_dyn_effective:.3f}")

                # Add to total loss
                losses['dyn_spec'] = L_dyn
                losses['total'] = losses['total'] + lambda_dyn_effective * L_dyn
                dyn_spec_loss_total += L_dyn.item()

        # Get loss components
        if 'seg_loss_components' in losses:
            dice_loss_total += losses['seg_loss_components'].get('dice', 0)
            ce_loss_total += losses['seg_loss_components'].get('ce', 0)
            focal_loss_total += losses['seg_loss_components'].get('focal', 0)

        # Backward pass
        losses['total'].backward()

        # Compute gradient statistics
        grad_stats_before = compute_gradient_statistics(model)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # Update weights
        optimizer.step()

        # Update metrics
        total_loss += losses['total'].item()
        seg_loss_total += losses['seg_loss'].item()

        # Store activations for first batch
        if i == 0:
            with torch.no_grad():
                activations.append(source_logits[:, :8].detach().cpu())

        # Clean up
        del source_logits
        if target_logits is not None:
            del target_logits

        # FIXED: Update progress bar with dynamic loss
        if is_main and hasattr(pbar, 'set_postfix'):
            postfix_dict = {
                'loss': f"{losses['total'].item():.4f}",
                'seg': f"{losses['seg_loss'].item():.4f}",
            }
            if 'target_seg_loss' in losses:
                postfix_dict['tgt_seg'] = f"{losses['target_seg_loss'].item():.4f}"
            if 'graph_total' in losses:
                postfix_dict['graph'] = f"{losses['graph_total'].item():.4f}"
            # FIXED: Add dynamic loss to progress bar
            if 'dyn_spec' in losses:
                postfix_dict['dyn'] = f"{losses['dyn_spec'].item():.4f}"
            pbar.set_postfix(postfix_dict)

        del losses

        # Periodically clear cache
        if i % 50 == 0:
            torch.cuda.empty_cache()

    # Compute statistics
    activation_stats = compute_activation_statistics(activations)
    weight_stats = compute_weight_statistics(model)
    final_system_stats = get_system_stats(rank)

    # Average metrics
    avg_loss = total_loss / num_steps
    avg_seg_loss = seg_loss_total / num_steps
    avg_target_seg_loss = target_seg_loss_total / num_steps if current_target_weight > 0 else 0

    # Component losses
    avg_dice_loss = dice_loss_total / num_steps
    avg_ce_loss = ce_loss_total / num_steps
    avg_focal_loss = focal_loss_total / num_steps

    # Graph losses
    avg_graph_loss = graph_loss_total / num_steps if graph_loss is not None else 0
    avg_graph_spec_src = graph_spec_src_total / num_steps if graph_loss is not None else 0
    avg_graph_edge_src = graph_edge_src_total / num_steps if graph_loss is not None else 0
    avg_graph_spec_tgt = graph_spec_tgt_total / num_steps if graph_loss is not None else 0
    avg_graph_edge_tgt = graph_edge_tgt_total / num_steps if graph_loss is not None else 0
    avg_graph_sym = graph_sym_total / num_steps if graph_loss is not None else 0

    # NEW: Dynamic spectral loss average
    avg_dyn_spec = dyn_spec_loss_total / num_steps if in_dynamic_stage else 0

    # NEW: Violation averages
    avg_forbidden_violations = forbidden_violations_total / num_steps if graph_loss is not None else 0
    avg_required_violations = required_violations_total / num_steps if graph_loss is not None else 0

    # Synchronize metrics across processes if distributed
    if is_distributed:
        metrics_to_sync = torch.tensor([
            avg_loss, avg_seg_loss, avg_target_seg_loss, avg_dice_loss,
            avg_ce_loss, avg_focal_loss, avg_graph_loss,
            avg_graph_spec_src, avg_graph_edge_src,
            avg_graph_spec_tgt, avg_graph_edge_tgt,
            avg_graph_sym, avg_dyn_spec,
            avg_forbidden_violations, avg_required_violations
        ], device=device)

        dist.all_reduce(metrics_to_sync, op=dist.ReduceOp.SUM)
        metrics_to_sync /= world_size

        (avg_loss, avg_seg_loss, avg_target_seg_loss, avg_dice_loss,
         avg_ce_loss, avg_focal_loss, avg_graph_loss,
         avg_graph_spec_src, avg_graph_edge_src,
         avg_graph_spec_tgt, avg_graph_edge_tgt,
         avg_graph_sym, avg_dyn_spec,
         avg_forbidden_violations, avg_required_violations) = metrics_to_sync.tolist()

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

        # Graph losses
        if graph_loss is not None:
            writer.add_scalar('train/graph_total', avg_graph_loss, global_step)
            writer.add_scalar('train/graph_spec_src', avg_graph_spec_src, global_step)
            writer.add_scalar('train/graph_edge_src', avg_graph_edge_src, global_step)
            writer.add_scalar('train/graph_spec_tgt', avg_graph_spec_tgt, global_step)
            writer.add_scalar('train/graph_edge_tgt', avg_graph_edge_tgt, global_step)
            writer.add_scalar('train/graph_sym', avg_graph_sym, global_step)
            writer.add_scalar('train/graph_warmup', graph_loss.get_warmup_factor(), global_step)
            # NEW: Violation tracking
            writer.add_scalar('train/forbidden_violations', avg_forbidden_violations, global_step)
            writer.add_scalar('train/required_violations', avg_required_violations, global_step)

        # NEW: Dynamic spectral metrics with k tracking
        if in_dynamic_stage:
            writer.add_scalar('train/dyn_spec', avg_dyn_spec, global_step)
            writer.add_scalar('train/dyn_lambda', lambda_dyn_effective, global_step)
            writer.add_scalar('train/dyn_k', dyn_k_effective, global_step)
            writer.add_scalar('train/dyn_conflict_suppressions', dyn_conflict_suppressions, global_step)

    elapsed = time.time() - start_time

    if is_main:
        print(f"\nâœ“ Epoch {epoch} completed in {elapsed:.1f}s")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Segmentation loss: {avg_seg_loss:.4f}")
        print(f"    - Dice: {avg_dice_loss:.4f}")
        print(f"    - CE: {avg_ce_loss:.4f}")
        print(f"    - Focal: {avg_focal_loss:.4f}")
        if current_target_weight > 0:
            print(f"  Target segmentation loss: {avg_target_seg_loss:.4f} (weight: {current_target_weight:.3f})")

        # Print dual-branch losses
        if graph_loss is not None:
            warmup = graph_loss.get_warmup_factor()
            print(f"  PRIOR BRANCH (graph loss): {avg_graph_loss:.4f} (warmup: {warmup:.3f})")
            if hasattr(graph_loss, 'graph_align_mode'):
                mode = graph_loss.graph_align_mode
                if mode in ['src_only', 'joint']:
                    print(f"    SOURCE alignment:")
                    print(f"      - Spectral: {avg_graph_spec_src:.4f}")
                    print(f"      - Edge: {avg_graph_edge_src:.4f}")
                if mode in ['tgt_only', 'joint']:
                    print(f"    TARGET regularization:")
                    print(f"      - Spectral: {avg_graph_spec_tgt:.4f}")
                    print(f"      - Edge: {avg_graph_edge_tgt:.4f}")
                print(f"    - Symmetry: {avg_graph_sym:.4f}")
            # NEW: Print violation statistics
            print(f"  Structural violations:")
            print(f"    - Forbidden edges present: {avg_forbidden_violations:.1f}")
            print(f"    - Required edges missing: {avg_required_violations:.1f}")

        if in_dynamic_stage:
            print(
                f"  DYNAMIC BRANCH (spectral): {avg_dyn_spec:.4f} (Î»={lambda_dyn_effective:.3f}, k={dyn_k_effective})")
            if dyn_conflict_suppressions > 0:
                print(f"    Conflict suppressions: {dyn_conflict_suppressions}")

    # Clear cache at epoch end
    torch.cuda.empty_cache()

    # Return metrics with dual-branch info
    return {
        'loss': avg_loss,
        'seg_loss': avg_seg_loss,
        'target_seg_loss': avg_target_seg_loss,
        'dice_loss': avg_dice_loss,
        'ce_loss': avg_ce_loss,
        'focal_loss': avg_focal_loss,
        'graph_loss': avg_graph_loss,
        'graph_spec_src': avg_graph_spec_src,
        'graph_edge_src': avg_graph_edge_src,
        'graph_spec_tgt': avg_graph_spec_tgt,
        'graph_edge_tgt': avg_graph_edge_tgt,
        'graph_sym': avg_graph_sym,
        'dyn_spec': avg_dyn_spec,  # NEW
        'dyn_lambda': lambda_dyn_effective,  # NEW
        'dyn_k': dyn_k_effective,  # NEW
        'dyn_conflicts': dyn_conflict_suppressions,  # NEW
        'forbidden_violations': avg_forbidden_violations,  # NEW
        'required_violations': avg_required_violations,  # NEW
        # Legacy keys
        'graph_spec': (avg_graph_spec_src + avg_graph_spec_tgt) / 2 if graph_loss else 0,
        'graph_edge': (avg_graph_edge_src + avg_graph_edge_tgt) / 2 if graph_loss else 0,
        'graph_struct': 0,
        'lr': optimizer.param_groups[0]['lr'],
        'grad_stats': grad_stats_before,
        'weight_stats': weight_stats,
        'activation_stats': activation_stats,
        'system_stats': final_system_stats,
        'epoch_time': elapsed
    }


def sliding_window_inference_with_tta(
        inputs,
        roi_size,
        sw_batch_size,
        predictor,
        overlap=0.5,
        use_tta=True,
        mode="gaussian",
        sigma_scale=0.125,
        padding_mode="constant",
        cval=0.0,
        laterality_pairs=None
):
    """Enhanced sliding window inference with Test Time Augmentation (TTA) and laterality-aware channel swapping"""

    if laterality_pairs is None:
        laterality_pairs = []

    def flip_tensor(x, axis):
        """Flip tensor along specified axis"""
        return torch.flip(x, dims=[axis])

    # Base prediction without TTA
    base_pred = sliding_window_inference(
        inputs,
        roi_size,
        sw_batch_size,
        predictor,
        overlap=overlap,
        mode=mode,
        sigma_scale=sigma_scale,
        padding_mode=padding_mode,
        cval=cval
    )

    if not use_tta:
        return base_pred

    # TTA with 3D flips (7 additional augmentations)
    predictions = [base_pred]

    # Single axis flips
    for axis in [2, 3, 4]:  # x, y, z axes (2 corresponds to x/left-right axis)
        flipped_input = flip_tensor(inputs, axis)
        pred = sliding_window_inference(
            flipped_input,
            roi_size,
            sw_batch_size,
            predictor,
            overlap=overlap,
            mode=mode,
            sigma_scale=sigma_scale,
            padding_mode=padding_mode,
            cval=cval
        )
        pred = flip_tensor(pred, axis)

        # Swap channels for laterality if x-axis flip
        if axis == 2 and laterality_pairs:
            pred = _swap_channels_for_laterality(pred, laterality_pairs)

        predictions.append(pred)

    # Double axis flips
    for axes in [(2, 3), (2, 4), (3, 4)]:
        flipped_input = inputs
        for axis in axes:
            flipped_input = flip_tensor(flipped_input, axis)

        pred = sliding_window_inference(
            flipped_input,
            roi_size,
            sw_batch_size,
            predictor,
            overlap=overlap,
            mode=mode,
            sigma_scale=sigma_scale,
            padding_mode=padding_mode,
            cval=cval
        )

        for axis in reversed(axes):
            pred = flip_tensor(pred, axis)

        # Swap channels if x-axis was involved
        if 2 in axes and laterality_pairs:
            pred = _swap_channels_for_laterality(pred, laterality_pairs)

        predictions.append(pred)

    # Triple axis flip
    flipped_input = inputs
    for axis in [2, 3, 4]:
        flipped_input = flip_tensor(flipped_input, axis)

    pred = sliding_window_inference(
        flipped_input,
        roi_size,
        sw_batch_size,
        predictor,
        overlap=overlap,
        mode=mode,
        sigma_scale=sigma_scale,
        padding_mode=padding_mode,
        cval=cval
    )

    for axis in [4, 3, 2]:
        pred = flip_tensor(pred, axis)

    # Triple axis includes x-axis, so swap channels
    if laterality_pairs:
        pred = _swap_channels_for_laterality(pred, laterality_pairs)

    predictions.append(pred)

    # Average all predictions
    final_pred = torch.stack(predictions).mean(dim=0)

    return final_pred


def val_epoch_simplified(model, loader, epoch, writer, args, dice_monitor=None,
                         device=None, is_distributed=False, world_size=1, rank=0):
    """Validate one epoch with enhanced TTA, overlap, laterality-aware processing, and CROSS-DOMAIN graph metrics"""

    # Import graph metric computation (FIXED: use cross-domain version)
    compute_graph_metrics = None
    if getattr(args, 'prior_adj_npy', None) and os.path.exists(args.prior_adj_npy):
        try:
            from graph_prior_loss import compute_validation_graph_metrics
            compute_graph_metrics = compute_validation_graph_metrics
        except ImportError:
            if rank == 0:
                print("âš ï¸ Graph metrics module not available for validation")

    is_main = (not is_distributed) or rank == 0
    model.eval()

    # Extract actual model from DDP if needed
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model

    # Load laterality pairs for TTA
    laterality_pairs = _load_lr_pairs_for_val(args)
    if is_main and laterality_pairs:
        print(f"  ðŸ“‹ Using {len(laterality_pairs)} laterality pairs for TTA")

    # Load required/forbidden edges for validation metrics
    required_edges = []
    forbidden_edges = []
    if getattr(args, 'prior_required_json', None) and os.path.exists(args.prior_required_json):
        with open(args.prior_required_json, 'r') as f:
            data = json.load(f)
            required_edges = [(int(i), int(j)) for i, j in data['required']]
    if getattr(args, 'prior_forbidden_json', None) and os.path.exists(args.prior_forbidden_json):
        with open(args.prior_forbidden_json, 'r') as f:
            data = json.load(f)
            forbidden_edges = [(int(i), int(j)) for i, j in data['forbidden']]

    # Get source prior path for cross-domain metrics
    src_prior_adj_path = getattr(args, 'src_prior_adj_npy', None)

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
    post_label = AsDiscrete(to_onehot=actual_model.num_classes)

    total_loss = 0
    num_steps = len(loader)

    # Component loss tracking
    dice_loss_total = 0
    ce_loss_total = 0
    focal_loss_total = 0

    # Graph metric accumulators (FIXED: cross-domain)
    all_adjacency_errors = []
    all_adjacency_errors_src = []  # NEW: source alignment errors
    all_symmetry_scores = []
    all_structural_violations = []

    if is_main:
        print(f"\nðŸ“Š Validation - Epoch {epoch}")

    # Enhanced inference settings
    use_tta = getattr(args, 'use_tta', True)
    infer_overlap = getattr(args, 'infer_overlap', 0.7)

    if is_main:
        print(f"  ðŸ”§ Enhanced Inference Settings:")
        print(f"    - TTA: {'Enabled (8 augmentations)' if use_tta else 'Disabled'}")
        if use_tta and laterality_pairs:
            print(f"    - Laterality-aware channel swapping: Enabled")
        print(f"    - Sliding Window Overlap: {infer_overlap}")
        print(f"    - Gaussian Blending: Enabled")
        if compute_graph_metrics:
            print(f"    - Graph metrics computation: Enabled")
            if src_prior_adj_path:
                print(f"    - Cross-domain metrics: Enabled")
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

    with torch.no_grad():
        for i, batch_data in enumerate(pbar):
            data = batch_data["image"].to(device, non_blocking=True)
            target = batch_data["label"].to(device, non_blocking=True)

            # Fix label dimensions
            if len(target.shape) == 5 and target.shape[1] == 1:
                target = target.squeeze(1)

            # Enhanced sliding window inference with TTA and laterality awareness
            roi_size = (args.roi_x, args.roi_y, args.roi_z)
            sw_batch_size = args.sw_batch_size

            # Time the inference
            infer_start = time.time()

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    output = sliding_window_inference_with_tta(
                        data,
                        roi_size,
                        sw_batch_size,
                        model,
                        overlap=infer_overlap,
                        use_tta=use_tta,
                        mode="gaussian",
                        sigma_scale=0.125,
                        padding_mode="constant",
                        cval=0.0,
                        laterality_pairs=laterality_pairs
                    )
            else:
                output = sliding_window_inference_with_tta(
                    data,
                    roi_size,
                    sw_batch_size,
                    model,
                    overlap=infer_overlap,
                    use_tta=use_tta,
                    mode="gaussian",
                    sigma_scale=0.125,
                    padding_mode="constant",
                    cval=0.0,
                    laterality_pairs=laterality_pairs
                )

            infer_time = time.time() - infer_start
            total_inference_time += infer_time
            inference_times.append(infer_time)

            # Compute loss
            loss, loss_components = seg_criterion(output, target)
            total_loss += loss.item()
            dice_loss_total += loss_components['dice']
            ce_loss_total += loss_components['ce']
            focal_loss_total += loss_components['focal']

            # Compute CROSS-DOMAIN graph validation metrics (FIXED)
            if compute_graph_metrics and i < 10:  # Compute for first 10 batches to save time
                graph_metrics = compute_graph_metrics(
                    pred=F.softmax(output, dim=1),  # Use probabilities
                    target=target,
                    prior_adj_path=args.prior_adj_npy,  # Target domain prior
                    src_prior_adj_path=src_prior_adj_path,  # Source domain prior (NEW)
                    lr_pairs=laterality_pairs,
                    required_edges=required_edges,
                    forbidden_edges=forbidden_edges
                )

                all_adjacency_errors.append(graph_metrics['adjacency_errors'])

                # Store source alignment errors separately if available
                if 'mean_abs_error_src' in graph_metrics['adjacency_errors']:
                    all_adjacency_errors_src.append({
                        'mean_abs_error_src': graph_metrics['adjacency_errors']['mean_abs_error_src'],
                        'spectral_distance_src': graph_metrics['adjacency_errors'].get('spectral_distance_src', 0)
                    })

                all_symmetry_scores.extend(graph_metrics['symmetry_scores'])
                all_structural_violations.append(graph_metrics['structural_violations'])

            # For metrics computation
            target_for_metrics = target.clone()
            target_for_metrics[target == -1] = 0

            if len(target_for_metrics.shape) == 4:
                target_for_metrics = target_for_metrics.unsqueeze(1)

            # Apply one-hot encoding
            output_one_hot = [post_pred(i) for i in decollate_batch(output)]
            target_one_hot = [post_label(i) for i in decollate_batch(target_for_metrics)]

            # Apply mask to one-hot tensors
            for j in range(len(output_one_hot)):
                mask = (decollate_batch(target.unsqueeze(1))[j] != -1).float()
                output_one_hot[j] = output_one_hot[j] * mask
                target_one_hot[j] = target_one_hot[j] * mask

            # Compute dice
            dice_metric(y_pred=output_one_hot, y=target_one_hot)
            dice_metric_per_class(y_pred=output_one_hot, y=target_one_hot)

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

    # Aggregate graph metrics (FIXED: include cross-domain)
    aggregated_adjacency_errors = {
        'mean_abs_error': 0.0,
        'max_error': 0.0,
        'spectral_distance': 0.0,
        'vs_gt_error': 0.0
    }

    aggregated_adjacency_errors_src = {
        'mean_abs_error_src': 0.0,
        'spectral_distance_src': 0.0
    }

    aggregated_structural_violations = {
        'required_missing': 0,
        'forbidden_present': 0,
        'containment_violated': 0,
        'exclusivity_violated': 0
    }

    if all_adjacency_errors:
        for key in aggregated_adjacency_errors:
            values = [d[key] for d in all_adjacency_errors if key in d]
            if values:
                aggregated_adjacency_errors[key] = np.mean(values)

    if all_adjacency_errors_src:
        for key in aggregated_adjacency_errors_src:
            values = [d[key] for d in all_adjacency_errors_src if key in d]
            if values:
                aggregated_adjacency_errors_src[key] = np.mean(values)

    if all_structural_violations:
        for key in aggregated_structural_violations:
            values = [d[key] for d in all_structural_violations if key in d]
            if values:
                aggregated_structural_violations[key] = sum(values)  # Total count

    # Synchronize metrics across processes if distributed
    if is_distributed:
        # Convert metrics to tensors for synchronization
        metrics_tensor = torch.tensor([
            dice_score, avg_loss, avg_dice_loss, avg_ce_loss,
            avg_focal_loss
        ], device=device)

        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        metrics_tensor /= world_size

        (dice_score, avg_loss, avg_dice_loss, avg_ce_loss,
         avg_focal_loss) = metrics_tensor.tolist()

        # Synchronize per-class dice scores
        if dice_per_class is not None:
            dist.all_reduce(dice_per_class, op=dist.ReduceOp.SUM)
            dice_per_class /= world_size

        # Synchronize graph metrics if available
        if all_adjacency_errors:
            graph_metrics_list = [
                aggregated_adjacency_errors['mean_abs_error'],
                aggregated_adjacency_errors['max_error'],
                aggregated_adjacency_errors['spectral_distance'],
                aggregated_adjacency_errors['vs_gt_error'],
                float(aggregated_structural_violations['required_missing']),
                float(aggregated_structural_violations['forbidden_present'])
            ]

            # Add source alignment metrics if available
            if all_adjacency_errors_src:
                graph_metrics_list.extend([
                    aggregated_adjacency_errors_src['mean_abs_error_src'],
                    aggregated_adjacency_errors_src['spectral_distance_src']
                ])

            graph_metrics_tensor = torch.tensor(graph_metrics_list, device=device)
            dist.all_reduce(graph_metrics_tensor, op=dist.ReduceOp.SUM)
            graph_metrics_tensor /= world_size

            # Unpack synchronized values
            idx = 0
            aggregated_adjacency_errors['mean_abs_error'] = graph_metrics_tensor[idx].item();
            idx += 1
            aggregated_adjacency_errors['max_error'] = graph_metrics_tensor[idx].item();
            idx += 1
            aggregated_adjacency_errors['spectral_distance'] = graph_metrics_tensor[idx].item();
            idx += 1
            aggregated_adjacency_errors['vs_gt_error'] = graph_metrics_tensor[idx].item();
            idx += 1
            aggregated_structural_violations['required_missing'] = int(graph_metrics_tensor[idx].item());
            idx += 1
            aggregated_structural_violations['forbidden_present'] = int(graph_metrics_tensor[idx].item());
            idx += 1

            if all_adjacency_errors_src and idx < len(graph_metrics_tensor):
                aggregated_adjacency_errors_src['mean_abs_error_src'] = graph_metrics_tensor[idx].item();
                idx += 1
                aggregated_adjacency_errors_src['spectral_distance_src'] = graph_metrics_tensor[idx].item();
                idx += 1

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
        print(f"\nðŸ“Š Validation Results:")
        print(f"  Overall Dice Score: {dice_score:.4f}")
        print(f"  Worst 10 Classes Avg Dice: {worst_10_dice_avg:.4f}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"    - Dice: {avg_dice_loss:.4f}")
        print(f"    - CE: {avg_ce_loss:.4f}")
        print(f"    - Focal: {avg_focal_loss:.4f}")

        # Print CROSS-DOMAIN graph metrics (FIXED)
        if all_adjacency_errors:
            print(f"\nðŸ§  Graph Metrics:")

            # Target domain alignment
            print(f"  TARGET Domain Adjacency:")
            print(f"    - Mean absolute error: {aggregated_adjacency_errors['mean_abs_error']:.4f}")
            print(f"    - Max error: {aggregated_adjacency_errors['max_error']:.4f}")
            print(f"    - Spectral distance: {aggregated_adjacency_errors['spectral_distance']:.4f}")
            print(f"    - vs GT error: {aggregated_adjacency_errors['vs_gt_error']:.4f}")

            # Source domain alignment (if available)
            if all_adjacency_errors_src and aggregated_adjacency_errors_src['mean_abs_error_src'] > 0:
                print(f"  SOURCE Domain Alignment:")
                print(f"    - Mean absolute error: {aggregated_adjacency_errors_src['mean_abs_error_src']:.4f}")
                print(f"    - Spectral distance: {aggregated_adjacency_errors_src['spectral_distance_src']:.4f}")

            print(f"  Structural Violations:")
            print(f"    - Required edges missing: {int(aggregated_structural_violations['required_missing'])}")
            print(f"    - Forbidden edges present: {int(aggregated_structural_violations['forbidden_present'])}")

            if all_symmetry_scores:
                print(f"  Symmetry Scores:")
                print(f"    - Mean: {np.mean(all_symmetry_scores):.3f}")
                print(f"    - Min: {np.min(all_symmetry_scores):.3f}")
                print(f"    - Max: {np.max(all_symmetry_scores):.3f}")

        print(f"\nâ±ï¸ Performance Metrics:")
        print(f"  Total validation time: {total_val_time:.1f}s")
        print(f"  Average inference time per sample: {avg_inference_time:.2f}s")
        print(f"  Total samples: {num_steps}")
        if use_tta:
            print(f"  TTA overhead: ~8x base inference time")

        # Print per-class dice scores
        if dice_per_class is not None and dice_per_class.numel() > 0:
            dice_per_class_np = dice_per_class.cpu().numpy()
            print(f"\nðŸ“Š Per-class Dice Scores:")

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

        # Log CROSS-DOMAIN graph metrics (FIXED)
        if all_adjacency_errors:
            # Target domain metrics
            writer.add_scalar('val/graph_adj_error_tgt', aggregated_adjacency_errors['mean_abs_error'], epoch)
            writer.add_scalar('val/graph_spectral_dist_tgt', aggregated_adjacency_errors['spectral_distance'], epoch)

            # Source domain metrics (if available)
            if all_adjacency_errors_src and aggregated_adjacency_errors_src['mean_abs_error_src'] > 0:
                writer.add_scalar('val/graph_adj_error_src', aggregated_adjacency_errors_src['mean_abs_error_src'],
                                  epoch)
                writer.add_scalar('val/graph_spectral_dist_src',
                                  aggregated_adjacency_errors_src['spectral_distance_src'], epoch)

            writer.add_scalar('val/graph_req_missing', aggregated_structural_violations['required_missing'], epoch)
            writer.add_scalar('val/graph_forb_present', aggregated_structural_violations['forbidden_present'], epoch)

            if all_symmetry_scores:
                writer.add_scalar('val/graph_symmetry', np.mean(all_symmetry_scores), epoch)

        if dice_per_class is not None and dice_per_class.numel() > 0:
            for c in range(actual_model.num_classes):
                writer.add_scalar(f'val/dice_class_{c}', dice_per_class_np[c], epoch)

    if is_main:
        elapsed = time.time() - start_time
        print(f"\nâœ“ Validation completed in {elapsed:.1f}s")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Dice Score: {dice_score:.4f}")
        print(f"  Worst 10 Classes Avg: {worst_10_dice_avg:.4f}")

    # Return metrics (including CROSS-DOMAIN graph metrics for DiceMonitor)
    val_metrics = {
        'loss': avg_loss,
        'dice': dice_score,
        'dice_loss': avg_dice_loss,
        'ce_loss': avg_ce_loss,
        'focal_loss': avg_focal_loss,
        'dice_per_class': dice_per_class_np.tolist() if dice_per_class is not None else None,
        'sample_dice_scores': sample_dice_scores,
        'class_prediction_counts': class_prediction_counts.tolist(),
        'class_label_counts': class_label_counts.tolist(),
        'val_time': total_val_time,
        'worst_10_dice_avg': worst_10_dice_avg,
        'avg_inference_time': avg_inference_time,
        'total_inference_time': total_inference_time,
        'tta_enabled': use_tta,
        'overlap_ratio': infer_overlap,
        'laterality_aware': len(laterality_pairs) > 0,
        # Add CROSS-DOMAIN graph metrics for DiceMonitor
        'adjacency_errors': aggregated_adjacency_errors if all_adjacency_errors else None,
        'adjacency_errors_src': aggregated_adjacency_errors_src if all_adjacency_errors_src else None,  # NEW
        'symmetry_scores': all_symmetry_scores if all_symmetry_scores else [],
        'structural_violations': aggregated_structural_violations if all_structural_violations else None
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
        'use_topk_dice': getattr(args, 'use_topk_dice', True),
        'topk_ratio': getattr(args, 'topk_ratio', 0.3),
        'dice_history': dice_history if dice_history is not None else {},
    }

    # Add additional info if provided
    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, filepath)
    print(f"ðŸ’¾ Checkpoint saved: {filepath}")