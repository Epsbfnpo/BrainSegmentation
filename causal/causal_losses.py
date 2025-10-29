import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def _ensure_labels_dim(labels: torch.Tensor) -> torch.Tensor:
    if labels.dim() == 5 and labels.shape[1] == 1:
        return labels.squeeze(1)
    return labels


def compute_per_sample_segmentation_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_config: Dict[str, object],
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Compute per-sample segmentation loss components to support causal regularizers."""
    labels = _ensure_labels_dim(labels)
    device = logits.device
    num_classes = int(loss_config.get('num_classes', logits.shape[1]))
    class_weights = loss_config.get('class_weights')
    if class_weights is not None and not isinstance(class_weights, torch.Tensor):
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    if isinstance(class_weights, torch.Tensor):
        class_weights = class_weights.to(device=device, dtype=logits.dtype)

    mask = (labels != -1)
    mask_float = mask.unsqueeze(1).to(dtype=logits.dtype)
    valid_counts = mask.view(mask.shape[0], -1).sum(dim=1).clamp(min=1)

    per_sample_losses = {
        'total': torch.zeros(logits.shape[0], device=device, dtype=logits.dtype),
        'dice': torch.zeros(logits.shape[0], device=device, dtype=logits.dtype),
        'ce': torch.zeros(logits.shape[0], device=device, dtype=logits.dtype),
        'focal': torch.zeros(logits.shape[0], device=device, dtype=logits.dtype),
    }

    # Cross entropy component
    if float(loss_config.get('ce_weight', 0.0)) > 0:
        ce_map = F.cross_entropy(
            logits,
            labels.long(),
            weight=class_weights,
            ignore_index=-1,
            reduction='none',
        )
        ce_weight_map = mask.to(dtype=logits.dtype)
        ce_loss = (ce_map * ce_weight_map).view(logits.shape[0], -1).sum(dim=1)
        ce_loss = ce_loss / valid_counts
        per_sample_losses['ce'] = ce_loss
        per_sample_losses['total'] = per_sample_losses['total'] + ce_loss * float(loss_config.get('ce_weight', 0.0))

    # Dice component
    if float(loss_config.get('dice_weight', 0.0)) > 0:
        probs = torch.softmax(logits, dim=1)
        probs = probs * mask_float
        clamped_labels = torch.clamp(labels, min=0, max=num_classes - 1)
        target_one_hot = F.one_hot(clamped_labels.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).to(dtype=logits.dtype)
        target_one_hot = target_one_hot * mask_float

        intersection = (probs * target_one_hot).sum(dim=(1, 2, 3, 4))
        if bool(loss_config.get('dice_squared', True)):
            probs_sq = (probs ** 2).sum(dim=(1, 2, 3, 4))
            target_sq = (target_one_hot ** 2).sum(dim=(1, 2, 3, 4))
        else:
            probs_sq = probs.sum(dim=(1, 2, 3, 4))
            target_sq = target_one_hot.sum(dim=(1, 2, 3, 4))
        dice_loss = 1.0 - (2.0 * intersection + eps) / (probs_sq + target_sq + eps)
        per_sample_losses['dice'] = dice_loss
        per_sample_losses['total'] = per_sample_losses['total'] + dice_loss * float(loss_config.get('dice_weight', 0.0))

    # Focal component
    if float(loss_config.get('focal_weight', 0.0)) > 0:
        gamma = float(loss_config.get('focal_gamma', 2.0))
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        clamped_labels = torch.clamp(labels, min=0, max=num_classes - 1)
        target_one_hot = F.one_hot(clamped_labels.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).to(dtype=logits.dtype)
        target_one_hot = target_one_hot * mask_float

        log_pt = (target_one_hot * log_probs).sum(dim=1)
        log_pt = torch.clamp(log_pt, min=-20.0, max=2.0)
        pt = torch.exp(log_pt)
        focal_factor = torch.pow(torch.clamp(1.0 - pt, min=0.0, max=1.0), gamma)

        if class_weights is not None:
            class_weights_map = (target_one_hot * class_weights.view(1, -1, 1, 1, 1)).sum(dim=1)
        else:
            class_weights_map = 1.0

        focal_map = -class_weights_map * focal_factor * log_pt
        focal_loss = (focal_map * mask.to(dtype=logits.dtype)).view(logits.shape[0], -1).sum(dim=1)
        focal_loss = focal_loss / valid_counts
        per_sample_losses['focal'] = focal_loss
        per_sample_losses['total'] = per_sample_losses['total'] + focal_loss * float(loss_config.get('focal_weight', 0.0))

    return per_sample_losses


def compute_age_bin_indices(
    ages: torch.Tensor,
    bin_edges: torch.Tensor,
) -> torch.Tensor:
    """Discretize ages into integer bin indices."""
    if ages.dim() == 2 and ages.shape[1] == 1:
        ages = ages.squeeze(1)
    bin_edges = bin_edges.to(device=ages.device, dtype=ages.dtype)
    indices = torch.bucketize(ages, bin_edges, right=False) - 1
    max_valid = bin_edges.numel() - 2
    indices = indices.clamp(min=0, max=max_valid)
    return indices.long()


def compute_age_balance_weights(
    ages: torch.Tensor,
    bin_edges: torch.Tensor,
    source_hist: torch.Tensor,
    target_hist: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    bins = compute_age_bin_indices(ages, bin_edges)
    source_hist = source_hist.to(device=ages.device, dtype=ages.dtype)
    target_hist = target_hist.to(device=ages.device, dtype=ages.dtype)
    weights = target_hist[bins] / torch.clamp(source_hist[bins], min=eps)
    return weights


def compute_conditional_vrex_loss(
    domain_losses: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    min_count: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute conditional variance risk across domains (V-REx)."""
    device = next(iter(domain_losses.values()))[0].device
    all_bins = torch.cat([info[1] for info in domain_losses.values() if info[1].numel() > 0])
    if all_bins.numel() == 0:
        return torch.zeros((), device=device)
    unique_bins = all_bins.unique(sorted=True)
    var_terms = []
    for b in unique_bins:
        env_means = []
        for losses, bins in domain_losses.values():
            mask = bins == b
            if mask.sum() >= min_count:
                env_means.append(losses[mask].mean())
        if len(env_means) > 1:
            stacked = torch.stack(env_means)
            variance = stacked.var(unbiased=False) + eps
            var_terms.append(variance)
    if not var_terms:
        return torch.zeros((), device=device)
    return torch.stack(var_terms).mean()


def compute_laplacian_invariance_loss(
    domain_probs: Dict[str, torch.Tensor],
    domain_bins: Dict[str, torch.Tensor],
    adjacency_fn,
    laplacian_fn,
    pool_kernel: int = 3,
    pool_stride: int = 2,
    temperature: float = 1.0,
    min_count: int = 1,
    restricted_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Laplacian residual invariance across domains conditioned on age."""
    devices = [probs.device for probs in domain_probs.values()]
    device = devices[0] if devices else torch.device('cpu')

    all_bins = torch.cat([bins for bins in domain_bins.values() if bins.numel() > 0])
    if all_bins.numel() == 0:
        return torch.zeros((), device=device)
    unique_bins = all_bins.unique(sorted=True)
    loss_terms = []

    pool_kwargs = {
        'kernel_size': pool_kernel,
        'stride': pool_stride,
        'temperature': temperature,
    }

    for b in unique_bins:
        domain_adjs = {}
        aggregated_adj = []
        for domain, probs in domain_probs.items():
            bins = domain_bins.get(domain)
            if bins is None:
                continue
            mask = bins == b
            if mask.sum() >= min_count:
                subset_probs = probs[mask]
                if subset_probs.numel() == 0:
                    continue
                adj = adjacency_fn(
                    subset_probs,
                    restricted_mask=restricted_mask,
                    **pool_kwargs,
                )
                domain_adjs[domain] = adj
                aggregated_adj.append(adj)
        if len(domain_adjs) < 2:
            continue
        expectation = torch.stack(aggregated_adj).mean(dim=0)
        for domain, adj in domain_adjs.items():
            residual_adj = adj - expectation
            residual_lap = laplacian_fn(residual_adj, normalized=True)
            domain_adjs[domain] = residual_lap
        domains = list(domain_adjs.keys())
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                diff = domain_adjs[domains[i]] - domain_adjs[domains[j]]
                loss_terms.append(torch.mean(diff ** 2))

    if not loss_terms:
        return torch.zeros((), device=device)
    return torch.stack(loss_terms).mean()


def generate_counterfactuals(
    images: torch.Tensor,
    intensity_scale: float = 0.25,
    intensity_shift: float = 0.15,
    noise_std: float = 0.03,
    bias_field_strength: float = 0.1,
    clip_min: float = -3.0,
    clip_max: float = 3.0,
) -> torch.Tensor:
    """Generate counterfactual style perturbations without altering anatomy."""
    device = images.device
    cf = images.clone()

    if intensity_scale > 0:
        scale = 1.0 + (torch.rand(cf.shape[0], 1, 1, 1, 1, device=device) - 0.5) * 2.0 * intensity_scale
        cf = cf * scale
    if intensity_shift > 0:
        shift = (torch.rand(cf.shape[0], 1, 1, 1, 1, device=device) - 0.5) * 2.0 * intensity_shift
        cf = cf + shift
    if noise_std > 0:
        noise = torch.randn_like(cf) * noise_std
        cf = cf + noise
    if bias_field_strength > 0:
        coarse = torch.randn(cf.shape[0], 1, 6, 6, 6, device=device)
        bias = F.interpolate(coarse, size=cf.shape[2:], mode='trilinear', align_corners=False)
        bias = bias / (bias.std(dim=(2, 3, 4), keepdim=True).clamp(min=1e-6))
        cf = cf * (1.0 + bias_field_strength * bias)

    cf = torch.clamp(cf, min=clip_min, max=clip_max)
    return cf


def compute_counterfactual_consistency_loss(
    logits_ref: torch.Tensor,
    logits_cf: torch.Tensor,
    confidence_threshold: float = 0.6,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Encourage consistency between original and counterfactual predictions."""
    probs_ref = torch.softmax(logits_ref, dim=1)
    probs_cf = torch.softmax(logits_cf, dim=1)

    confidence = probs_ref.max(dim=1).values
    weight_mask = (confidence >= confidence_threshold).to(dtype=probs_ref.dtype).unsqueeze(1)
    if weight_mask.sum() == 0:
        weight_mask = torch.ones_like(weight_mask)

    intersection = (probs_ref * probs_cf * weight_mask).sum(dim=(1, 2, 3, 4))
    denom = ((probs_ref ** 2 + probs_cf ** 2) * weight_mask).sum(dim=(1, 2, 3, 4))
    dice_loss = 1.0 - (intersection * 2.0 + eps) / (denom + eps)
    return dice_loss.mean()
