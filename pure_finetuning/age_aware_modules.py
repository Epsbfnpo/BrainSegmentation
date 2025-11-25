from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist


class SimplifiedDAUnetModule(nn.Module):
    """Thin wrapper around a backbone network with a stable interface.

    The original project supported age conditioning and prior-driven class
    weighting. For the pure fine-tuning baseline we intentionally drop those
    features and expose only the raw backbone. This keeps the forward signature
    compatible with existing training and evaluation utilities while ensuring
    no external priors influence optimisation.
    """

    def __init__(self,
                 backbone: nn.Module,
                 num_classes: int,
                 *,
                 foreground_only: bool = True,
                 debug_mode: bool = False):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.foreground_only = foreground_only
        self.debug_mode = debug_mode
        self.class_weights = None

        is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        if is_main:
            print("✅ SimplifiedDAUnetModule initialised (baseline mode)")
            print(f"  Classes: {num_classes}")
            print(f"  Foreground-only remap: {foreground_only}")
            print("  Using uniform class weights")

    def forward(self, x: torch.Tensor, age: torch.Tensor | None = None) -> torch.Tensor:  # noqa: D401
        return self.backbone(x)

    def get_class_weights(self):
        return self.class_weights

