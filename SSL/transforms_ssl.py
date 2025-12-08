"""
Data transforms for SSL pretraining on dHCP dataset
Includes resolution adjustment and appropriate preprocessing
"""
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Spacingd, SpatialPadd, CenterSpatialCropd,
    RandSpatialCropd, RandFlipd, RandRotated,
    RandGaussianNoised, RandGaussianSmoothd,
    RandScaleIntensityd, RandShiftIntensityd,
    RandAdjustContrastd, ToTensord,
    Transform, MapTransform, ScaleIntensityRanged
)
from monai.data import MetaTensor
from typing import Dict, List, Tuple, Optional


class PercentileNormalizationd(MapTransform):
    """Percentile normalization as used in the domain adaptation code"""

    def __init__(self, keys, lower=1, upper=99, b_min=0.0, b_max=1.0):
        super().__init__(keys)
        self.lower = lower
        self.upper = upper
        self.b_min = b_min
        self.b_max = b_max

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key]

            # Handle MetaTensor
            if isinstance(image, MetaTensor):
                image_np = image.array
                image_meta = image.meta
            else:
                image_np = np.asarray(image)
                image_meta = None

            # Ensure float type
            image_np = image_np.astype(np.float32)

            # Check for NaN or Inf in input
            if np.isnan(image_np).any() or np.isinf(image_np).any():
                print(f"⚠️  WARNING: NaN or Inf found in image before normalization")
                # Replace NaN/Inf with zeros
                image_np = np.nan_to_num(image_np, nan=0.0, posinf=0.0, neginf=0.0)

            # Get non-zero voxels for percentile calculation
            non_zero_mask = image_np > 0
            if non_zero_mask.any():
                non_zero_voxels = image_np[non_zero_mask]
                # Calculate percentiles on non-zero voxels
                lower_percentile = np.percentile(non_zero_voxels, self.lower)
                upper_percentile = np.percentile(non_zero_voxels, self.upper)

                # Check for valid percentiles
                if np.isnan(lower_percentile) or np.isnan(upper_percentile):
                    print(f"⚠️  WARNING: NaN percentiles computed, using min/max")
                    lower_percentile = np.min(non_zero_voxels)
                    upper_percentile = np.max(non_zero_voxels)

                # Clip and normalize
                image_clipped = np.clip(image_np, lower_percentile, upper_percentile)

                # Normalize to [b_min, b_max]
                if upper_percentile > lower_percentile:
                    image_norm = (image_clipped - lower_percentile) / (upper_percentile - lower_percentile)
                    image_norm = image_norm * (self.b_max - self.b_min) + self.b_min
                else:
                    # If all values are the same, set to middle of range
                    image_norm = np.ones_like(image_np) * ((self.b_max + self.b_min) / 2)

                # Preserve background as 0
                image_norm[~non_zero_mask] = 0

                # Final check for NaN
                if np.isnan(image_norm).any() or np.isinf(image_norm).any():
                    print(f"⚠️  WARNING: NaN or Inf after normalization, replacing with zeros")
                    image_norm = np.nan_to_num(image_norm, nan=0.0, posinf=1.0, neginf=0.0)

                # Ensure output is float32
                image_norm = image_norm.astype(np.float32)

                # Preserve type
                if image_meta is not None:
                    d[key] = MetaTensor(image_norm, meta=image_meta)
                else:
                    d[key] = image_norm
            else:
                # If all zeros, keep as zeros
                print(f"⚠️  WARNING: All-zero image encountered")
                if image_meta is not None:
                    d[key] = MetaTensor(np.zeros_like(image_np, dtype=np.float32), meta=image_meta)
                else:
                    d[key] = np.zeros_like(image_np, dtype=np.float32)

        return d


class IntensityClipd(MapTransform):
    """Clip intensity values to remove outliers"""

    def __init__(self, keys, percentile=99.5):
        super().__init__(keys)
        self.percentile = percentile

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key]

            if isinstance(image, MetaTensor):
                image_np = image.array
                image_meta = image.meta
            else:
                image_np = np.asarray(image)
                image_meta = None

            # Clip outliers
            non_zero_mask = image_np > 0
            if non_zero_mask.any():
                threshold = np.percentile(image_np[non_zero_mask], self.percentile)
                image_np = np.clip(image_np, 0, threshold)

            if image_meta is not None:
                d[key] = MetaTensor(image_np, meta=image_meta)
            else:
                d[key] = image_np

        return d


def get_ssl_transforms(
        args,
        mode: str = 'train',
        target_spacing: List[float] = [0.8, 0.8, 0.8]
) -> Compose:
    """Get transforms for SSL pretraining

    Args:
        args: Arguments containing ROI size and other parameters
        mode: 'train' or 'val'
        target_spacing: Target voxel spacing in mm
    """

    # Base transforms - always applied
    base_transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),

        # Resolution standardization (from 0.5mm to 0.8mm)
        Spacingd(
            keys=["image"],
            pixdim=target_spacing,
            mode="bilinear",
        ),

        # Intensity preprocessing
        IntensityClipd(keys=["image"], percentile=99.5),
        PercentileNormalizationd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0
        ),
    ]

    if mode == 'train':
        # Training transforms with augmentation
        transforms = base_transforms + [
            # Spatial transforms
            RandSpatialCropd(
                keys=["image"],
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
                random_center=True,
                random_size=False,
            ),

            # Spatial augmentations (moderate for SSL)
            RandFlipd(
                keys=["image"],
                spatial_axis=[0],
                prob=0.5,
            ),
            RandFlipd(
                keys=["image"],
                spatial_axis=[1],
                prob=0.5,
            ),
            RandFlipd(
                keys=["image"],
                spatial_axis=[2],
                prob=0.5,
            ),

            # Small rotations
            RandRotated(
                keys=["image"],
                range_x=0.2,  # Reduced from 0.5
                range_y=0.2,
                range_z=0.2,
                prob=0.3,
                mode="bilinear",
            ),

            # Intensity augmentations (mild for SSL)
            RandScaleIntensityd(
                keys=["image"],
                factors=0.1,  # Reduced from 0.3
                prob=0.3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.05,  # Reduced from 0.1
                prob=0.3,
            ),

            # Mild contrast adjustment
            RandAdjustContrastd(
                keys=["image"],
                prob=0.2,
                gamma=(0.9, 1.1)  # Reduced range
            ),

            # Mild noise
            RandGaussianNoised(
                keys=["image"],
                prob=0.2,
                mean=0.0,
                std=0.02,  # Reduced from 0.05
            ),

            # Mild smoothing
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
                prob=0.2,
            ),

            ToTensord(keys=["image"]),
        ]
    else:
        # Validation transforms - no augmentation
        transforms = base_transforms + [
            # Pad if needed
            SpatialPadd(
                keys=["image"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                mode="constant",
                constant_values=0,
            ),
            # Center crop
            CenterSpatialCropd(
                keys=["image"],
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
            ),
            ToTensord(keys=["image"]),
        ]

    return Compose(transforms)


class SSLAugmentation:
    """Additional augmentations specific to SSL tasks"""

    @staticmethod
    def get_strong_augmentation(args) -> Compose:
        """Get strong augmentation for contrastive learning"""
        return Compose([
            RandSpatialCropd(
                keys=["image"],
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
                random_center=True,
                random_size=False,
            ),
            RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.8),
            RandRotated(
                keys=["image"],
                range_x=0.4,
                range_y=0.4,
                range_z=0.4,
                prob=0.5,
                mode="bilinear",
            ),
            RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
            ToTensord(keys=["image"]),
        ])