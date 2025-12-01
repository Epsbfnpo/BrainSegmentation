import torch
import torch.nn as nn
import torch.jit


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits.

    Returns entropy per-sample/per-voxel; caller can reduce as needed.
    """
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def configure_model_for_tent(model: nn.Module):
    """Prepare model for TENT adaptation.

    1. Enable train mode to allow statistics updates where applicable.
    2. Freeze all parameters by default.
    3. Unfreeze only affine parameters (weight/bias) of normalization layers.
    """
    model.train()
    model.requires_grad_(False)

    params = []
    names = []

    real_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )

    print("TENT Configuration: Scanning for Normalization layers...")

    for nm, m in real_model.named_modules():
        if isinstance(
            m,
            (
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LayerNorm,
                nn.GroupNorm,
                nn.InstanceNorm3d,
            ),
        ):
            m.requires_grad_(True)

            if hasattr(m, "track_running_stats"):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

            for np_name, p in m.named_parameters():
                if np_name in {"weight", "bias"}:
                    params.append(p)
                    names.append(f"{nm}.{np_name}")

    print(f"TENT Configuration: Found {len(params)} affine parameters to update.")
    return params, names
