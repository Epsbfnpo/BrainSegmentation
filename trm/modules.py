import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETRWrapper(nn.Module):
    """Thin wrapper around MONAI SwinUNETR to match pretrained checkpoint keys."""

    def __init__(self, args):
        super().__init__()
        self.backbone = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=not args.no_swin_checkpoint,
        )

    def forward(self, x):
        return self.backbone(x)


def load_pretrained_weights(model: nn.Module, checkpoint_path: str) -> None:
    """Load weights from a checkpoint produced by the source model."""

    if not checkpoint_path:
        return
    print(f"📦 Loading weights from {checkpoint_path}")
    # Allow full checkpoint contents because source checkpoints include argparse.Namespace metadata.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    new_state = {}
    for key, value in state_dict.items():
        cleaned = key
        if cleaned.startswith("module."):
            cleaned = cleaned[7:]
        if not cleaned.startswith("backbone."):
            cleaned = f"backbone.{cleaned}"
        new_state[cleaned] = value

    msg = model.load_state_dict(new_state, strict=False)
    print(f"✅ Weights loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
