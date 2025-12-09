"""
Model definition for supervised fine-tuning
Loads SSL pretrained weights and adds classification head
"""
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
import os
from typing import Optional, Dict


def load_ssl_pretrained_weights(model: nn.Module, checkpoint_path: str, strict: bool = False):
    """Load weights from SSL pretrained model

    Args:
        model: Target model to load weights into
        checkpoint_path: Path to SSL checkpoint
        strict: Whether to enforce strict loading (default: False)
    """
    print(f"\n📥 Loading SSL pretrained weights from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SSL pretrained model not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        pretrained_state_dict = checkpoint['model']
    else:
        pretrained_state_dict = checkpoint

    # Get current model state dict
    model_state_dict = model.state_dict()

    # Filter and adapt pretrained weights
    adapted_state_dict = {}
    skipped_keys = []
    shape_mismatches = []

    for key, value in pretrained_state_dict.items():
        # Remove module. prefix if present
        if key.startswith('module.'):
            key = key[7:]

        # Check if key exists in current model
        if key in model_state_dict:
            # Check shape compatibility
            if value.shape == model_state_dict[key].shape:
                adapted_state_dict[key] = value
            else:
                # Special handling for output layers
                if 'out' in key or 'head' in key or 'final' in key:
                    # This is expected for classification layers
                    shape_mismatches.append((key, value.shape, model_state_dict[key].shape))
                else:
                    # Unexpected shape mismatch
                    print(f"  ⚠️  Shape mismatch for {key}: {value.shape} vs {model_state_dict[key].shape}")
                    shape_mismatches.append((key, value.shape, model_state_dict[key].shape))
        else:
            skipped_keys.append(key)

    # Load adapted weights
    model.load_state_dict(adapted_state_dict, strict=False)

    # Report loading statistics
    total_params = len(model_state_dict)
    loaded_params = len(adapted_state_dict)
    print(f"\n✓ Loaded {loaded_params}/{total_params} parameters ({loaded_params / total_params * 100:.1f}%)")

    if shape_mismatches:
        print(f"  Shape mismatches (expected for output layers): {len(shape_mismatches)}")
        for key, old_shape, new_shape in shape_mismatches[:5]:  # Show first 5
            print(f"    - {key}: {old_shape} → {new_shape}")

    if skipped_keys and len(skipped_keys) < 10:
        print(f"  Skipped keys: {skipped_keys}")

    # Initialize unloaded parameters
    unloaded_keys = set(model_state_dict.keys()) - set(adapted_state_dict.keys())
    if unloaded_keys:
        print(f"\n  Randomly initialized parameters: {len(unloaded_keys)}")
        # Show some examples
        examples = list(unloaded_keys)[:5]
        for key in examples:
            print(f"    - {key}")
        if len(unloaded_keys) > 5:
            print(f"    ... and {len(unloaded_keys) - 5} more")


class SegmentationHead(nn.Module):
    """Custom segmentation head with optional dropout"""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.dropout(x)
        return self.conv(x)


def create_supervised_model(args) -> nn.Module:
    """Create model for supervised fine-tuning

    Args:
        args: Arguments containing model configuration

    Returns:
        Model ready for supervised training
    """
    print("\n🏗️  Creating SwinUNETR model for supervised training...")
    print(f"  Output channels: {args.out_channels} (87 brain regions)")
    print(f"  Foreground-only mode: {args.foreground_only}")

    # Create base model
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,  # 87 classes for foreground-only
        feature_size=args.feature_size,
        drop_rate=0.0,  # We'll use dropout in the head instead
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,  # Enable gradient checkpointing
    )

    # Replace the output layer with custom head if dropout is specified
    if hasattr(args, 'dropout_rate') and args.dropout_rate > 0:
        print(f"  Adding dropout ({args.dropout_rate}) to segmentation head")

        # Get the in_channels of the original output layer
        # For SwinUNETR, the output layer is typically model.out
        if hasattr(model, 'out') and hasattr(model.out, 'conv'):
            # Standard SwinUNETR structure
            in_features = model.out.conv.in_channels
        elif hasattr(model.out, 'in_channels'):
            in_features = model.out.in_channels
        else:
            # Fallback: use feature_size * 8 (typical for SwinUNETR)
            in_features = args.feature_size * 8
            print(f"  ⚠️  Could not detect in_channels, using {in_features}")

        model.out = SegmentationHead(
            in_channels=in_features,
            out_channels=args.out_channels,
            dropout_rate=args.dropout_rate
        )

    # Load SSL pretrained weights
    if args.pretrained_model:
        load_ssl_pretrained_weights(model, args.pretrained_model, strict=False)
    else:
        print("  ⚠️  No pretrained model specified, training from scratch")

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 ** 2:.1f} MB (fp32)")

    return model


class ModelEMA:
    """Exponential Moving Average of model parameters

    Helps stabilize training and can improve final performance
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_val = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_val.clone()

    def apply_shadow(self):
        """Apply shadow parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}


def freeze_encoder(model: nn.Module, freeze_ratio: float = 0.5):
    """Optionally freeze early layers of the encoder

    Args:
        model: SwinUNETR model
        freeze_ratio: Fraction of encoder layers to freeze (0.0 = none, 1.0 = all)
    """
    if freeze_ratio <= 0:
        return

    print(f"\n❄️  Freezing {freeze_ratio * 100:.0f}% of encoder layers")

    # Get all encoder parameters
    encoder_params = []
    for name, param in model.named_parameters():
        if 'swinViT' in name:  # Encoder parameters
            encoder_params.append((name, param))

    # Freeze specified fraction
    num_to_freeze = int(len(encoder_params) * freeze_ratio)
    for i, (name, param) in enumerate(encoder_params[:num_to_freeze]):
        param.requires_grad = False

    # Report
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Remaining trainable: {trainable_params:,}")