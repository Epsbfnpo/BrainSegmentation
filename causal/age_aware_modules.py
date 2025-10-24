import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import json
import os


class AgeConditionedModule(nn.Module):
    """Age conditioning module using FiLM (Feature-wise Linear Modulation)"""

    def __init__(self, age_embed_dim=64, num_features=None):
        super().__init__()
        self.age_embed = nn.Sequential(
            nn.Linear(1, age_embed_dim),
            nn.ReLU(),
            nn.Linear(age_embed_dim, age_embed_dim * 2),
            nn.ReLU()
        )

        if num_features is not None:
            # Generate scale and shift parameters
            self.gamma = nn.Linear(age_embed_dim * 2, num_features)
            self.beta = nn.Linear(age_embed_dim * 2, num_features)
            nn.init.zeros_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)
            self.register_buffer("strength", torch.tensor(0.0))

    def set_strength(self, s: float):
        s = float(max(0.0, min(1.0, s)))
        self.strength = torch.as_tensor(s, device=self.strength.device)

    def forward(self, x, age):
        """
        Apply age conditioning to features
        Args:
            x: (B, C, ...) feature tensor
            age: (B, 1) age tensor
        """
        age_feat = self.age_embed(age)  # (B, age_embed_dim * 2)

        if hasattr(self, 'gamma'):
            gamma_raw = self.gamma(age_feat).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            beta_raw = self.beta(age_feat).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # 有界（避免把 logits 推爆），并用 strength 做门控，初期~0，逐步拉到1
            gamma = 0.1 * torch.tanh(gamma_raw) * self.strength
            beta = 0.1 * torch.tanh(beta_raw) * self.strength

            return x * (1 + gamma) + beta
        else:
            return age_feat


class VolumePredictor(nn.Module):
    """Predict expected volume for each class given age"""

    def __init__(self, num_classes, age_embed_dim=64):
        super().__init__()
        self.num_classes = num_classes

        # Age encoder
        self.age_encoder = nn.Sequential(
            nn.Linear(1, age_embed_dim),
            nn.ReLU(),
            nn.Linear(age_embed_dim, age_embed_dim * 2),
            nn.ReLU()
        )

        # Volume predictor for each class
        self.volume_mean = nn.Linear(age_embed_dim * 2, num_classes)
        self.volume_std = nn.Linear(age_embed_dim * 2, num_classes)

    def forward(self, age):
        """
        Predict volume statistics given age
        Args:
            age: (B, 1) age tensor
        Returns:
            mean: (B, C) expected volume mean for each class
            std: (B, C) expected volume std for each class
        """
        age_feat = self.age_encoder(age)
        mean = self.volume_mean(age_feat)
        std = F.softplus(self.volume_std(age_feat)) + 0.01  # Ensure positive
        return mean, std


class ShapeEncoder(nn.Module):
    """Encode shape priors given age"""

    def __init__(self, num_classes, shape_dim=128, age_embed_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.shape_dim = shape_dim

        # Age encoder
        self.age_encoder = nn.Sequential(
            nn.Linear(1, age_embed_dim),
            nn.ReLU(),
            nn.Linear(age_embed_dim, age_embed_dim * 2),
            nn.ReLU()
        )

        # Shape template generator for each class
        self.shape_generator = nn.Linear(age_embed_dim * 2, num_classes * shape_dim)

    def forward(self, age):
        """
        Generate shape templates given age
        Args:
            age: (B, 1) age tensor
        Returns:
            shapes: (B, C, shape_dim) shape features for each class
        """
        age_feat = self.age_encoder(age)
        shapes = self.shape_generator(age_feat)
        B = age.shape[0]
        shapes = shapes.view(B, self.num_classes, self.shape_dim)
        return shapes


class SimplifiedDAUnetModule(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes: int = 87,
                 roi_size: Tuple[int, int, int] = (96, 96, 96),
                 foreground_only: bool = False, class_prior_path: str = None,
                 enhanced_class_weights: bool = True,
                 use_age_conditioning: bool = True,
                 age_embed_dim: int = 64,
                 volume_statistics_path: str = None,
                 shape_prior_path: str = None,
                 debug_mode: bool = False,
                 debug_step_limit: int = 2):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.foreground_only = foreground_only
        self.enhanced_class_weights = enhanced_class_weights
        self.use_age_conditioning = use_age_conditioning
        self.class_weights = self._load_class_weights(class_prior_path)
        self.debug_mode = bool(debug_mode)
        self.debug_step_limit = max(1, int(debug_step_limit)) if self.debug_mode else 0
        self._debug_logged_steps = set()

        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main:
            print(f"✅ Simplified DAUnet Module initialized")
            print(f"  Number of classes: {num_classes}")
            print(f"  ROI size: {roi_size}")
            print(f"  Foreground only: {foreground_only}")
            print(f"  Enhanced class weights: {enhanced_class_weights}")
            print(f"  Age conditioning: {use_age_conditioning}")
            if self.class_weights is not None:
                print(
                    f"  Class weights loaded: shape={self.class_weights.shape}, min={self.class_weights.min():.3f}, max={self.class_weights.max():.3f}")
            if self.debug_mode:
                print(f"  🐞 Debug mode enabled for SimplifiedDAUnetModule (first {self.debug_step_limit} steps per epoch)")

        # Age conditioning modules
        if use_age_conditioning:
            # Age conditioner for features
            self.age_conditioner = AgeConditionedModule(age_embed_dim, self.num_classes)

            # Volume predictor
            self.volume_predictor = VolumePredictor(num_classes, age_embed_dim)

            # Shape encoder
            self.shape_encoder = ShapeEncoder(num_classes, shape_dim=128, age_embed_dim=age_embed_dim)

            # Age prediction head (auxiliary task)
            self.age_predictor = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(self.num_classes, age_embed_dim),
                nn.ReLU(),
                nn.Linear(age_embed_dim, 1)
            )
            # Default: rely on GraphPriorLoss for volume prior, disable internal volume loss
            self.enable_internal_volume_loss = False

            # Load volume statistics if available
            self.volume_statistics = self._load_volume_statistics(volume_statistics_path)

            if is_main:
                print(f"  ✨ Age conditioning modules initialized")
                print(f"    - Feature conditioner")
                print(f"    - Volume predictor")
                print(f"    - Shape encoder")
                print(f"    - Age predictor (auxiliary)")

    def _load_class_weights(self, class_prior_path: str) -> Optional[torch.Tensor]:
        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if class_prior_path is None or not os.path.exists(class_prior_path):
            if is_main:
                print("  ⚠️  No class prior file provided, using uniform weights")
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
                        f"  ⚠️  WARNING: Mismatch! Prior has {len(class_ratios)} classes, model expects {self.num_classes}")
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

    def _load_volume_statistics(self, volume_statistics_path: str) -> Optional[Dict]:
        """Load precomputed volume statistics per age"""
        if volume_statistics_path is None or not os.path.exists(volume_statistics_path):
            return None
        with open(volume_statistics_path, 'r') as f:
            return json.load(f)

    def get_small_class_indices(self, top_k: int = 20) -> Optional[np.ndarray]:
        if self.class_weights is None:
            return None
        weights_np = self.class_weights.cpu().numpy()
        small_class_indices = np.argsort(weights_np)[-top_k:]
        return small_class_indices

    def forward(self, x: torch.Tensor, age: Optional[torch.Tensor] = None):
        """
        Forward pass with optional age conditioning
        Args:
            x: (B, C, X, Y, Z) input image
            age: (B, 1) age tensor
        """
        # Get base features
        features = self.base_model(x)

        # Apply age conditioning if available
        if self.use_age_conditioning and age is not None:
            features = self.age_conditioner(features, age)

        return features

    def compute_losses(self, source_images: torch.Tensor, source_labels: torch.Tensor,
                       seg_criterion: nn.Module, source_ages: Optional[torch.Tensor] = None,
                       step: int = 0) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        debug_active = self.debug_mode and (step < self.debug_step_limit) and is_main

        with torch.no_grad():
            unique_labels = torch.unique(source_labels)
            min_label = unique_labels.min().item()
            max_label = unique_labels.max().item()
            if step == 0 and is_main and max_label >= self.num_classes:
                print(f"\n🔍 Label Check (step {step}):")
                print(f"  Unique labels count: {len(unique_labels)}")
                print(f"  Range: [{min_label}, {max_label}]")
                print(f"  Model expects: -1 (ignored) and 0 to {self.num_classes - 1}")
            if max_label >= self.num_classes:
                if is_main:
                    print(f"  ⚠️  Clamping labels from {max_label} to {self.num_classes - 1}")
                valid_mask = source_labels >= 0
                source_labels = torch.where(valid_mask, torch.clamp(source_labels, min=0, max=self.num_classes - 1),
                                            source_labels)
            if debug_active and step not in self._debug_logged_steps:
                print(f"\n🐞 [SimplifiedDAUnet][Step {step}] Input diagnostics")
                print(f"  Source images shape: {tuple(source_images.shape)}, dtype={source_images.dtype}")
                print(f"  Source labels shape: {tuple(source_labels.shape)}, dtype={source_labels.dtype}")
                print(f"  Unique labels (after clamp): count={len(unique_labels)}, min={min_label}, max={max_label}")
                neg_count = int((source_labels == -1).sum().item())
                print(f"  Ignored voxels (-1): {neg_count}")
                img_view = source_images.detach()
                print(f"  Image stats: min={img_view.min():.4f}, max={img_view.max():.4f}, mean={img_view.mean():.4f}, std={img_view.std():.4f}")
                if source_ages is not None:
                    ages_np = source_ages.detach().cpu().flatten()
                    print(f"  Age stats: min={ages_np.min():.2f}, max={ages_np.max():.2f}, mean={ages_np.mean():.2f}")

        if len(source_labels.shape) == 4:
            source_labels = source_labels.unsqueeze(1)

        # Forward with age conditioning
        source_seg = self.forward(source_images, source_ages)

        if step == 0 and is_main:
            print(f"  Model output shape: {source_seg.shape}")
            print(f"  Labels shape: {source_labels.shape}")
            if source_ages is not None:
                print(f"  Age range in batch: [{source_ages.min():.1f}, {source_ages.max():.1f}]")

        if debug_active and step not in self._debug_logged_steps:
            with torch.no_grad():
                logits = source_seg.detach()
                print(f"  Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}, std={logits.std():.4f}")
                if torch.isnan(logits).any():
                    print(f"  ⚠️  NaNs detected in logits! count={int(torch.isnan(logits).sum().item())}")
                probs = torch.softmax(logits, dim=1)
                probs_sum = probs.sum(dim=(2, 3, 4))
                row = probs_sum[0]
                topk = torch.topk(row, k=min(5, row.numel()))
                print(f"  Top-{topk.indices.numel()} class probability masses (sample 0):")
                for idx in topk.indices.cpu().tolist():
                    print(f"    Class {idx}: mass={row[idx].item():.5f}")

        try:
            seg_loss_output = seg_criterion(source_seg, source_labels)
        except RuntimeError as e:
            if is_main:
                print(f"\n⌠Loss computation failed!")
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
            if debug_active and step not in self._debug_logged_steps:
                print("  Segmentation loss components:")
                for key, value in seg_loss_components.items():
                    print(f"    {key}: {float(value):.6f}")

        if debug_active and step not in self._debug_logged_steps:
            print(f"  Segmentation loss (combined): {seg_loss.item():.6f}")

        # Age prediction loss (auxiliary task)
        if self.use_age_conditioning and source_ages is not None and getattr(self, "enable_internal_volume_loss", False):
            predicted_age = self.age_predictor(source_seg)
            age_loss = F.smooth_l1_loss(predicted_age, source_ages)
            losses['age_loss'] = age_loss * 0.01  # Small weight for auxiliary task
            if debug_active and step not in self._debug_logged_steps:
                print(f"  Age auxiliary loss: {losses['age_loss'].item():.6f}")

        # Volume consistency loss
        if self.use_age_conditioning and source_ages is not None and getattr(self, "enable_internal_volume_loss", False):
            # Get predicted volumes from softmax probabilities
            probs = F.softmax(source_seg, dim=1)
            predicted_volumes = probs.sum(dim=(2, 3, 4))  # (B, C)

            # Get expected volumes given age
            expected_mean, expected_std = self.volume_predictor(source_ages)

            # Compute volume loss (KL divergence style)
            volume_loss = F.smooth_l1_loss(predicted_volumes, expected_mean)
            losses['volume_loss'] = volume_loss * 0.1
            if debug_active and step not in self._debug_logged_steps:
                print(f"  Internal volume loss: {losses['volume_loss'].item():.6f}")

        # Total loss
        total_loss = seg_loss
        if 'age_loss' in losses:
            total_loss = total_loss + losses['age_loss']
        if 'volume_loss' in losses:
            total_loss = total_loss + losses['volume_loss']

        losses['total'] = total_loss
        losses['logits'] = source_seg

        if debug_active and step not in self._debug_logged_steps:
            print(f"  Total loss: {total_loss.item():.6f}")
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated(source_seg.device) / (1024 ** 3)
                mem_max = torch.cuda.max_memory_allocated(source_seg.device) / (1024 ** 3)
                print(f"  CUDA memory: alloc={mem_alloc:.2f} GB, max_alloc={mem_max:.2f} GB")
            self._debug_logged_steps.add(step)

        return losses