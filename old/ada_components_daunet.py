import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import json
import os


class SimplifiedDAUnetModule(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes: int = 88, roi_size: Tuple[int, int, int] = (96, 96, 96),
                 foreground_only: bool = False, class_prior_path: str = None, enhanced_class_weights: bool = True):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.foreground_only = foreground_only
        self.enhanced_class_weights = enhanced_class_weights
        self.class_weights = self._load_class_weights(class_prior_path)
        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main:
            print(f"âœ… Simplified DAUnet Module initialized")
            print(f"  Number of classes: {num_classes}")
            print(f"  ROI size: {roi_size}")
            print(f"  Foreground only: {foreground_only}")
            print(f"  Enhanced class weights: {enhanced_class_weights}")
            if self.class_weights is not None:
                print(
                    f"  Class weights loaded: shape={self.class_weights.shape}, min={self.class_weights.min():.3f}, max={self.class_weights.max():.3f}")

    def _load_class_weights(self, class_prior_path: str) -> Optional[torch.Tensor]:
        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if class_prior_path is None or not os.path.exists(class_prior_path):
            if is_main:
                print("  âš ï¸  No class prior file provided, using uniform weights")
            return None
        if is_main:
            print(f"  Loading class weights from: {class_prior_path}")
        with open(class_prior_path, 'r') as f:
            prior_data = json.load(f)
        class_ratios = np.array(prior_data['class_ratios'])
        if self.foreground_only:
            class_ratios = class_ratios[1:]
            if is_main:
                print(f"  Foreground-only mode: Using {len(class_ratios)} foreground class ratios")
                print(f"  Model expects {self.num_classes} classes")
            if len(class_ratios) != self.num_classes:
                if is_main:
                    print(
                        f"  âš ï¸  WARNING: Mismatch! Prior has {len(class_ratios)} classes, model expects {self.num_classes}")
                if self.num_classes == 87 and len(class_ratios) == 88:
                    class_ratios = class_ratios[:87]
        epsilon = 1e-7
        class_weights = 1.0 / (class_ratios + epsilon)
        if self.enhanced_class_weights:
            if is_main:
                print("  Using enhanced class weighting strategy")
            class_weights = class_weights / class_weights.mean()
            class_weights = np.log1p(class_weights)
            small_class_threshold = 0.001
            small_classes = class_ratios < small_class_threshold
            num_small_classes = small_classes.sum()
            if num_small_classes > 0 and is_main:
                print(f"  Found {num_small_classes} small classes (< 0.1% of voxels)")
                class_weights[small_classes] *= 2.0
                small_indices = np.where(small_classes)[0]
                for idx in small_indices[:5]:
                    if self.foreground_only:
                        original_idx = idx + 1
                    else:
                        original_idx = idx
                    print(
                        f"    Class {idx} (orig {original_idx}): ratio={class_ratios[idx]:.6f}, weight={class_weights[idx]:.3f}")
            class_weights = np.clip(class_weights, 0.1, 20.0)
            class_weights = class_weights / class_weights.mean()
        else:
            if is_main:
                print("  Using standard class weighting strategy")
            class_weights = class_weights / class_weights.mean()
            class_weights = np.sqrt(class_weights)
            class_weights = np.clip(class_weights, 0.1, 10.0)
        if is_main:
            print(f"  Final class weights shape: {class_weights.shape}")
            print(
                f"  Weight statistics: min={class_weights.min():.3f}, max={class_weights.max():.3f}, mean={class_weights.mean():.3f}, std={class_weights.std():.3f}")
            print("  Weight distribution:")
            print(f"    Weights < 0.5: {(class_weights < 0.5).sum()} classes")
            print(f"    Weights 0.5-1.0: {((class_weights >= 0.5) & (class_weights < 1.0)).sum()} classes")
            print(f"    Weights 1.0-2.0: {((class_weights >= 1.0) & (class_weights < 2.0)).sum()} classes")
            print(f"    Weights 2.0-5.0: {((class_weights >= 2.0) & (class_weights < 5.0)).sum()} classes")
            print(f"    Weights > 5.0: {(class_weights >= 5.0).sum()} classes")
        return torch.FloatTensor(class_weights).cuda()

    def get_small_class_indices(self, top_k: int = 20) -> Optional[np.ndarray]:
        if self.class_weights is None:
            return None
        weights_np = self.class_weights.cpu().numpy()
        small_class_indices = np.argsort(weights_np)[-top_k:]
        return small_class_indices

    def forward(self, x: torch.Tensor):
        return self.base_model(x)

    def compute_losses(self, source_images: torch.Tensor, source_labels: torch.Tensor, seg_criterion: nn.Module,
                       step: int = 0) -> Dict[str, torch.Tensor]:
        losses = {}
        with torch.no_grad():
            unique_labels = torch.unique(source_labels)
            min_label = unique_labels.min().item()
            max_label = unique_labels.max().item()
            is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if step == 0 and is_main and max_label >= self.num_classes:
                print(f"\nðŸ” Label Check (step {step}):")
                print(f"  Unique labels count: {len(unique_labels)}")
                print(f"  Range: [{min_label}, {max_label}]")
                print(f"  Model expects: -1 (ignored) and 0 to {self.num_classes - 1}")
            if max_label >= self.num_classes:
                if is_main:
                    print(f"  âš ï¸  Clamping labels from {max_label} to {self.num_classes - 1}")
                valid_mask = source_labels >= 0
                source_labels = torch.where(valid_mask, torch.clamp(source_labels, min=0, max=self.num_classes - 1),
                                            source_labels)
        if len(source_labels.shape) == 4:
            source_labels = source_labels.unsqueeze(1)
        source_seg = self.forward(source_images)
        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if step == 0 and is_main:
            print(f"  Model output shape: {source_seg.shape}")
            print(f"  Labels shape: {source_labels.shape}")
        try:
            seg_loss_output = seg_criterion(source_seg, source_labels)
        except RuntimeError as e:
            if is_main:
                print(f"\nâŒ Loss computation failed!")
                print(f"  Error: {str(e)}")
                with torch.no_grad():
                    print(f"  Label stats: min={source_labels.min()}, max={source_labels.max()}")
                    print(f"  Unique labels: {torch.unique(source_labels).cpu().tolist()}")
            raise
        if isinstance(seg_loss_output, tuple):
            seg_loss, seg_loss_components = seg_loss_output
        else:
            seg_loss = seg_loss_output
            seg_loss_components = None
        losses['seg_loss'] = seg_loss
        if seg_loss_components:
            losses['seg_loss_components'] = seg_loss_components
        losses['total'] = seg_loss

        # Return logits as well to avoid redundant forward passes
        losses['logits'] = source_seg

        return losses