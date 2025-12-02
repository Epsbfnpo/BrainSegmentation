"""SHOT core utilities: parameter freezing and entropy-based loss."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def shot_loss(logits: torch.Tensor, *, diversity_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the SHOT objective.

    SHOT encourages confident predictions (low entropy) while maintaining
    class diversity across the batch (high entropy of the batch-mean logits).

    Args:
        logits: Network outputs of shape ``(B, C, ...)``.
        diversity_weight: Weighting factor ``β`` for the diversity term.

    Returns:
        total_loss: ``L_ent - β * L_div`` to be **minimised**.
        ent_loss: Mean pixel-wise entropy.
        div_loss: Batch-wise diversity term (entropy of the mean probabilities).
    """

    # Probabilities for each class
    probs = torch.softmax(logits, dim=1)

    # Pixel-wise entropy averaged over batch and spatial dims
    ent_loss = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1).mean()

    # Diversity: entropy of the mean probability over the entire batch volume
    mean_prob = probs.mean(dim=(0,) + tuple(range(2, probs.dim())))
    div_loss = -(mean_prob * torch.log(mean_prob.clamp_min(1e-6))).sum()

    total_loss = ent_loss - diversity_weight * div_loss
    return total_loss, ent_loss, div_loss


def configure_model_for_shot(model: nn.Module, *, verbose: bool = True) -> List[nn.Parameter]:
    """Freeze the segmentation head while leaving the feature extractor trainable.

    The function returns the parameters that should be optimised (i.e. all
    non-head parameters). It also supports models wrapped in ``DistributedDataParallel``.
    """

    real_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    params_to_update: List[nn.Parameter] = []

    frozen_param_ids = set()
    head = getattr(getattr(real_model, "backbone", real_model), "out", None)
    if head is not None:
        for param in head.parameters():
            param.requires_grad = False
            frozen_param_ids.add(id(param))

    for name, param in real_model.named_parameters():
        if id(param) in frozen_param_ids:
            if verbose:
                print(f"[SHOT] Frozen head parameter: {name}")
            continue
        param.requires_grad = True
        params_to_update.append(param)

    if verbose:
        print(f"[SHOT] Trainable parameters: {len(params_to_update)}")
        print(f"[SHOT] Frozen head parameters: {len(frozen_param_ids)}")

    return params_to_update
