"""
Data transforms for supervised fine-tuning on dHCP dataset
FIXED: Correct foreground mask, parameterized rotation, safer LR swap
"""
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Spacingd, Orientationd,
    RandRotated,
    RandGaussianNoised, RandGaussianSmoothd,
    RandScaleIntensityd, RandShiftIntensityd,
    RandAdjustContrastd, RandGibbsNoised,
    ToTensord, CastToTyped,
    Transform, MapTransform,
    RandBiasFieldd, AdjustContrastd
)
from monai.data import MetaTensor
from typing import Dict, List, Tuple, Optional
import random
import json
import os


class ForegroundPercentileNormalizationd(MapTransform):
    """Percentile normalization using only foreground voxels

    FIXED: Use label >= 0 for foreground mask (0-86 are brain regions, -1 is background)
    """

    def __init__(self, keys, label_key="label", lower=1, upper=99, b_min=0.0, b_max=1.0):
        super().__init__(keys)
        self.label_key = label_key
        self.lower = lower
        self.upper = upper
        self.b_min = b_min
        self.b_max = b_max

    def __call__(self, data):
        d = dict(data)

        # Get foreground mask from label
        label = d[self.label_key]
        if isinstance(label, MetaTensor):
            label_np = label.array
        else:
            label_np = np.asarray(label)

        # FIXED: Create foreground mask (brain tissue: 0-86, background: -1)
        foreground_mask = label_np >= 0  # Changed from > 0

        for key in self.key_iterator(d):
            if key == self.label_key or key == 'label_original':
                continue

            image = d[key]

            if isinstance(image, MetaTensor):
                image_np = image.array
                image_meta = image.meta
            else:
                image_np = np.asarray(image)
                image_meta = None

            image_np = image_np.astype(np.float32)

            # Check for NaN/Inf
            if np.isnan(image_np).any() or np.isinf(image_np).any():
                print(f"⚠️ WARNING: NaN or Inf found in image before normalization")
                image_np = np.nan_to_num(image_np, nan=0.0, posinf=0.0, neginf=0.0)

            # Calculate percentiles only on foreground voxels
            if foreground_mask.any():
                foreground_voxels = image_np[foreground_mask]

                if len(foreground_voxels) > 0:
                    lower_percentile = np.percentile(foreground_voxels, self.lower)
                    upper_percentile = np.percentile(foreground_voxels, self.upper)

                    # Clip and normalize
                    image_clipped = np.clip(image_np, lower_percentile, upper_percentile)

                    if upper_percentile > lower_percentile:
                        image_norm = (image_clipped - lower_percentile) / (upper_percentile - lower_percentile)
                        image_norm = image_norm * (self.b_max - self.b_min) + self.b_min
                    else:
                        image_norm = np.ones_like(image_np) * ((self.b_max + self.b_min) / 2)

                    # Preserve background as 0
                    image_norm[~foreground_mask] = 0
                else:
                    image_norm = np.zeros_like(image_np)
            else:
                print(f"⚠️ WARNING: No foreground voxels found")
                image_norm = np.zeros_like(image_np)

            image_norm = image_norm.astype(np.float32)

            if image_meta is not None:
                d[key] = MetaTensor(image_norm, meta=image_meta)
            else:
                d[key] = image_norm

        return d


class ConvertLabelsForForegroundOnlyd(MapTransform):
    """Convert labels for foreground-only training with robust handling

    Maps regions 1-87 to 0-86, background to -1 (ignore_index)
    """

    def __init__(self, keys, num_classes=87):
        super().__init__(keys)
        self.num_classes = num_classes

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            label = d[key]

            if isinstance(label, MetaTensor):
                label_np = label.array
                label_meta = label.meta
            else:
                label_np = np.asarray(label)
                label_meta = None

            # Robust conversion to integer (handle potential float labels)
            label_np = np.rint(label_np).astype(np.int32)

            # Store original labels
            if key == "label":
                if isinstance(label, MetaTensor):
                    d["label_original"] = MetaTensor(label_np.copy(), meta=label_meta.copy())
                else:
                    d["label_original"] = label_np.copy()

            # Convert for foreground-only mode
            converted_label = np.full_like(label_np, -1, dtype=np.int32)  # Default to ignore

            # Map brain regions 1-87 to 0-86
            for region in range(1, 88):
                converted_label[label_np == region] = region - 1

            # Verify conversion
            unique_values = np.unique(converted_label)
            valid_range = set(range(-1, self.num_classes))
            invalid_values = set(unique_values) - valid_range
            if invalid_values:
                print(f"⚠️ WARNING: Invalid label values after conversion: {invalid_values}")
                # Clip to valid range
                converted_label = np.clip(converted_label, -1, self.num_classes - 1)

            if label_meta is not None:
                d[key] = MetaTensor(converted_label, meta=label_meta)
            else:
                d[key] = converted_label

        return d


class LRSymmetricFlipd(MapTransform):
    """Left-right flip with corresponding label swapping for brain symmetry

    FIXED: Disable flip if swap file missing, validate numbering
    """

    def __init__(self, keys, swap_pairs_file, prob=0.5, enabled=True):
        super().__init__(keys)
        self.prob = prob
        self.enabled = enabled

        if not enabled:
            self.swap_map = {}
            return

        # Load swap pairs
        if swap_pairs_file and os.path.exists(swap_pairs_file):
            with open(swap_pairs_file, 'r') as f:
                swap_pairs = json.load(f)

            # Validate that all values are in 1-87 range (1-based)
            all_values = []
            for left, right in swap_pairs:
                all_values.extend([left, right])

            min_val, max_val = min(all_values), max(all_values)
            if min_val < 1 or max_val > 87:
                raise ValueError(f"LR swap pairs contain invalid values: min={min_val}, max={max_val}, expected 1-87")

            # Convert to mapping dict (both directions) with 0-based indexing
            self.swap_map = {}
            for left, right in swap_pairs:
                # Adjust for 0-based indexing after conversion
                self.swap_map[left - 1] = right - 1
                self.swap_map[right - 1] = left - 1

            print(f"✓ Loaded {len(swap_pairs)} LR swap pairs (1-based: {min_val}-{max_val})")
        else:
            # FIXED: Disable flipping if swap file missing
            print(f"⚠️ WARNING: LR swap file not found: {swap_pairs_file}")
            print(f"   Disabling LR flip augmentation to avoid inconsistency")
            self.enabled = False
            self.swap_map = {}

    def __call__(self, data):
        d = dict(data)

        # Skip if disabled or random check fails
        if not self.enabled or random.random() > self.prob:
            return d

        for key in self.key_iterator(d):
            img = d[key]

            if isinstance(img, MetaTensor):
                img_array = img.array
                meta_dict = img.meta.copy() if hasattr(img, 'meta') else {}
            else:
                img_array = np.asarray(img)
                meta_dict = {}

            # Apply LR flip (axis 0 is typically the L-R axis after RAS orientation)
            if key == "image":
                # For 4D image (C, H, W, D), flip axis 1 (first spatial dim)
                if img_array.ndim == 4:
                    img_array = np.flip(img_array, axis=1).copy()
                else:
                    img_array = np.flip(img_array, axis=0).copy()
            elif key in ["label", "label_original"] and self.swap_map:
                # For labels, also swap corresponding left-right pairs
                if img_array.ndim == 3:
                    img_array = np.flip(img_array, axis=0).copy()

                    # Apply label swapping
                    swapped = img_array.copy()
                    for old_label, new_label in self.swap_map.items():
                        mask = img_array == old_label
                        if mask.any():
                            swapped[mask] = new_label
                    img_array = swapped
            else:
                # Simple flip for other keys
                if img_array.ndim == 3:
                    img_array = np.flip(img_array, axis=0).copy()

            if isinstance(img, MetaTensor):
                d[key] = MetaTensor(img_array, meta=meta_dict)
            else:
                d[key] = torch.from_numpy(img_array) if not isinstance(img_array, torch.Tensor) else img_array

        return d


class RandCropByLabelClassesd(MapTransform):
    """Random crop centered on different label classes based on class ratios"""

    def __init__(self, keys, label_key, roi_size, num_classes, class_ratios=None, num_samples=1):
        super().__init__(keys)
        self.label_key = label_key
        self.roi_size = roi_size
        self.num_classes = num_classes
        self.num_samples = num_samples

        # Set sampling probabilities based on class ratios
        if class_ratios is not None:
            # Inverse frequency with smoothing
            weights = np.array([1.0 / (r + 1e-6) for r in class_ratios[1:88]])  # Skip background
            # Apply sqrt for stability
            weights = np.sqrt(weights)
            # Normalize
            self.class_probs = weights / weights.sum()
        else:
            # Uniform sampling
            self.class_probs = np.ones(num_classes) / num_classes

    def __call__(self, data):
        d = dict(data)

        label = d[self.label_key]
        if isinstance(label, MetaTensor):
            label_np = label.array
        else:
            label_np = np.asarray(label)

        # Find available classes in this sample
        unique_classes = np.unique(label_np)
        unique_classes = unique_classes[unique_classes >= 0]  # Exclude background (-1)

        if len(unique_classes) == 0:
            # No valid classes, do center crop
            return self._center_crop(d)

        # Sample a class based on probabilities
        # Filter probabilities to only available classes
        available_probs = self.class_probs[unique_classes]
        available_probs = available_probs / available_probs.sum()

        selected_class = np.random.choice(unique_classes, p=available_probs)

        # Find center of mass for selected class
        class_mask = (label_np == selected_class)
        if not class_mask.any():
            return self._center_crop(d)

        # Get bounding box of the class
        coords = np.where(class_mask)
        center = [int(np.mean(c)) for c in coords]

        # Calculate crop boundaries
        crop_start = []
        crop_end = []

        for i, (c, s, dim) in enumerate(zip(center, self.roi_size, label_np.shape)):
            # Add random offset to avoid always centering perfectly
            offset = np.random.randint(-s//4, s//4 + 1)
            start = c + offset - s // 2
            start = max(0, min(start, dim - s))
            crop_start.append(start)
            crop_end.append(start + s)

        # Apply crop to all keys
        for key in self.key_iterator(d):
            img = d[key]

            if isinstance(img, MetaTensor):
                img_array = img.array
                meta_dict = img.meta.copy()
            else:
                img_array = np.asarray(img)
                meta_dict = {}

            # Handle different dimensions
            if key == "image" and img_array.ndim == 4:
                cropped = img_array[:,
                                   crop_start[0]:crop_end[0],
                                   crop_start[1]:crop_end[1],
                                   crop_start[2]:crop_end[2]]
            else:
                cropped = img_array[crop_start[0]:crop_end[0],
                                   crop_start[1]:crop_end[1],
                                   crop_start[2]:crop_end[2]]

            if isinstance(img, MetaTensor):
                d[key] = MetaTensor(cropped, meta=meta_dict)
            else:
                d[key] = cropped

        return d

    def _center_crop(self, data):
        """Fallback center crop"""
        d = dict(data)

        for key in self.key_iterator(d):
            img = d[key]

            if isinstance(img, MetaTensor):
                img_array = img.array
            else:
                img_array = np.asarray(img)

            if key == "image" and img_array.ndim == 4:
                spatial_shape = img_array.shape[1:]
            else:
                spatial_shape = img_array.shape

            crop_start = [(s - r) // 2 for s, r in zip(spatial_shape, self.roi_size)]
            crop_end = [s + r for s, r in zip(crop_start, self.roi_size)]

            if key == "image" and img_array.ndim == 4:
                cropped = img_array[:,
                                   crop_start[0]:crop_end[0],
                                   crop_start[1]:crop_end[1],
                                   crop_start[2]:crop_end[2]]
            else:
                cropped = img_array[crop_start[0]:crop_end[0],
                                   crop_start[1]:crop_end[1],
                                   crop_start[2]:crop_end[2]]

            d[key] = cropped

        return d


class CustomSpatialPadd(MapTransform):
    """Custom spatial padding that handles -1 ignore index"""

    def __init__(self, keys, spatial_size, mode="constant", constant_values=0):
        super().__init__(keys)
        self.spatial_size = spatial_size
        self.mode = mode
        self.constant_values = constant_values

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            if key not in d:
                continue

            img = d[key]

            if isinstance(img, MetaTensor):
                img_array = img.array
                meta_dict = img.meta.copy() if hasattr(img, 'meta') else {}
            else:
                img_array = np.asarray(img)
                meta_dict = {}

            if torch.is_tensor(img_array):
                img_array = img_array.numpy()

            # Determine padding based on data type
            if key == "image":
                if img_array.ndim == 3:
                    img_array = np.expand_dims(img_array, axis=0)

                current_shape = img_array.shape[1:]
                pad_width = [(0, 0)]  # No padding for channel

                for i in range(3):
                    if current_shape[i] < self.spatial_size[i]:
                        pad_before = (self.spatial_size[i] - current_shape[i]) // 2
                        pad_after = self.spatial_size[i] - current_shape[i] - pad_before
                        pad_width.append((pad_before, pad_after))
                    else:
                        pad_width.append((0, 0))

                pad_value = self.constant_values

            elif key in ["label", "label_original"]:
                if img_array.ndim == 4 and img_array.shape[0] == 1:
                    img_array = img_array.squeeze(0)

                current_shape = img_array.shape
                pad_width = []

                for i in range(3):
                    if current_shape[i] < self.spatial_size[i]:
                        pad_before = (self.spatial_size[i] - current_shape[i]) // 2
                        pad_after = self.spatial_size[i] - current_shape[i] - pad_before
                        pad_width.append((pad_before, pad_after))
                    else:
                        pad_width.append((0, 0))

                # Use -1 for label padding (ignore index)
                pad_value = -1

            # Apply padding if needed
            if any(p[0] > 0 or p[1] > 0 for p in pad_width):
                img_array = np.pad(img_array, pad_width, mode=self.mode, constant_values=pad_value)

            if isinstance(img, MetaTensor):
                d[key] = MetaTensor(img_array, meta=meta_dict)
            else:
                d[key] = torch.from_numpy(img_array) if not isinstance(img_array, torch.Tensor) else img_array

        return d


def get_supervised_transforms(
    args,
    mode: str = 'train',
    target_spacing: List[float] = None
) -> Compose:
    """Get transforms for supervised training on registered dHCP data

    FIXED: Remove validation center crop, parameterized rotation
    """

    if target_spacing is None:
        target_spacing = args.target_spacing

    roi_size = [args.roi_x, args.roi_y, args.roi_z]

    # Get rotation angle from args
    max_rotation_angle = args.max_rotation_angle if hasattr(args, 'max_rotation_angle') else 0.1

    # Base transforms
    base_transforms = [
        # Load data
        LoadImaged(keys=["image", "label"], image_only=False),

        # Ensure channel first
        EnsureChannelFirstd(keys=["image", "label"]),

        # Standardize orientation to RAS
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # Apply spacing
        Spacingd(
            keys=["image", "label"],
            pixdim=target_spacing,
            mode=("bilinear", "nearest"),
        ),

        # Convert labels to foreground-only format FIRST
        ConvertLabelsForForegroundOnlyd(keys=["label"], num_classes=args.out_channels),

        # Foreground-aware intensity normalization (FIXED: >= 0 for foreground)
        ForegroundPercentileNormalizationd(
            keys=["image"],
            label_key="label",
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0
        ),
    ]

    if mode == 'train':
        # Load class prior if available
        class_ratios = None
        if hasattr(args, 'class_prior_json') and os.path.exists(args.class_prior_json):
            with open(args.class_prior_json, 'r') as f:
                prior_data = json.load(f)
                class_ratios = prior_data.get('class_ratios', None)

        # LR swap file path from args or disable if not available
        enable_lr_flip = True
        lr_swap_file = None
        if hasattr(args, 'laterality_pairs_json'):
            lr_swap_file = args.laterality_pairs_json
            if not os.path.exists(lr_swap_file):
                print(f"⚠️ LR swap file not found, disabling LR flip: {lr_swap_file}")
                enable_lr_flip = False
        else:
            # Try default location
            lr_swap_file = os.path.join(os.path.dirname(__file__), 'dhcp_lr_swap.json')
            if not os.path.exists(lr_swap_file):
                print(f"⚠️ No LR swap file found, disabling LR flip")
                enable_lr_flip = False

        transforms = base_transforms + [
            # Padding
            CustomSpatialPadd(
                keys=["image", "label", "label_original"],
                spatial_size=roi_size,
                mode="constant"
            ),

            # Class-aware cropping (prioritize rare classes)
            RandCropByLabelClassesd(
                keys=["image", "label", "label_original"],
                label_key="label",
                roi_size=roi_size,
                num_classes=args.out_channels,
                class_ratios=class_ratios
            ),
        ]

        # Add LR flip only if swap file available
        if enable_lr_flip:
            transforms.append(
                LRSymmetricFlipd(
                    keys=["image", "label", "label_original"],
                    swap_pairs_file=lr_swap_file,
                    prob=0.5,
                    enabled=True
                )
            )

        # Continue with other augmentations
        transforms.extend([

            # Parameterized rotation for registered data
            RandRotated(
                keys=["image", "label", "label_original"],
                range_x=max_rotation_angle,
                range_y=max_rotation_angle,
                range_z=max_rotation_angle,
                prob=0.2,
                mode=("bilinear", "nearest", "nearest"),
            ),

            # Enhanced intensity augmentations
            RandGaussianNoised(
                keys=["image"],
                std=0.02,
                prob=0.4,
            ),

            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
                prob=0.3,
            ),

            RandScaleIntensityd(
                keys=["image"],
                factors=0.2,
                prob=0.4,
            ),

            RandShiftIntensityd(
                keys=["image"],
                offsets=0.1,
                prob=0.4,
            ),

            RandAdjustContrastd(
                keys=["image"],
                gamma=(0.6, 1.4),
                prob=0.4,
            ),

            # MRI-specific augmentations
            RandBiasFieldd(
                keys=["image"],
                degree=3,
                coeff_range=(0.0, 0.02),
                prob=0.3,
            ),

            # Ensure correct types
            CastToTyped(keys=["image"], dtype=torch.float32),
            CastToTyped(keys=["label"], dtype=torch.long),
            CastToTyped(keys=["label_original"], dtype=torch.long),
        ])
    else:
        # Validation transforms
        # FIXED: Remove center crop to enable sliding window inference
        transforms = base_transforms + [
            # Padding only (no cropping for validation)
            CustomSpatialPadd(
                keys=["image", "label", "label_original"],
                spatial_size=roi_size,
                mode="constant"
            ),

            # Ensure correct types
            CastToTyped(keys=["image"], dtype=torch.float32),
            CastToTyped(keys=["label"], dtype=torch.long),
            CastToTyped(keys=["label_original"], dtype=torch.long),
        ]

    return Compose(transforms)


def get_post_transforms(args) -> Compose:
    """Get post-processing transforms for predictions"""
    from monai.transforms import (
        EnsureTyped, AsDiscreted,
        KeepLargestConnectedComponentd, FillHolesd
    )

    transforms = [
        EnsureTyped(keys="pred"),
        AsDiscreted(keys="pred", argmax=True),
    ]

    # Add connected component analysis for specific regions prone to false positives
    if hasattr(args, 'use_post_processing') and args.use_post_processing:
        # These regions often benefit from connected component filtering
        # (adjust based on your dataset characteristics)
        transforms.extend([
            KeepLargestConnectedComponentd(
                keys="pred",
                applied_labels=list(range(10, 30)),  # Example: small subcortical structures
            ),
            FillHolesd(
                keys="pred",
                applied_labels=list(range(0, 10)),  # Example: larger cortical regions
            ),
        ])

    return Compose(transforms)