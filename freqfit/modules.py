import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETRWrapper(nn.Module):
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
    if not checkpoint_path: return
    print(f"📦 Loading weights from {checkpoint_path}")
    # Use weights_only=False to avoid errors with complex checkpoints (e.g. containing args)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    new_state = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        if not k.startswith("backbone."): k = f"backbone.{k}"
        new_state[k] = v

    msg = model.load_state_dict(new_state, strict=False)
    print(f"✅ Weights loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")