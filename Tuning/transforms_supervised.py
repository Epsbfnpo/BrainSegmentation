"""
Data transforms for supervised fine-tuning (AMOS CT adaptation)
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Spacingd, Orientationd,
    RandRotated, RandZoomd, RandFlipd,
    RandGaussianNoised, RandGaussianSmoothd,
    RandAdjustContrastd,
    ScaleIntensityRanged, SpatialPadd,
    ToTensord, MapTransform
)
from monai.data import MetaTensor


class RandCropByLabelClassesd(MapTransform):
    """Random crop centered on different label classes based on class ratios"""

    def __init__(self, keys, label_key, spatial_size, num_classes, class_ratios=None, num_samples=1):
        super().__init__(keys)
        self.label_key = label_key
        self.roi_size = spatial_size
        self.num_classes = num_classes
        self.num_samples = num_samples

        # Set sampling probabilities based on class ratios
        if class_ratios is not None:
            # Inverse frequency with smoothing (skip background)
            ratios = np.asarray(class_ratios[1:self.num_classes], dtype=float)
            weights = np.array([1.0 / (r + 1e-6) for r in ratios])
            weights = np.sqrt(weights)  # Apply sqrt for stability
            self.class_probs = weights / weights.sum()
        else:
            # Uniform sampling
            self.class_probs = np.ones(self.num_classes, dtype=float) / float(self.num_classes)

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
        unique_classes = unique_classes.astype(int, copy=False)

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
                cropped = img_array[
                    :,
                    crop_start[0]:crop_end[0],
                    crop_start[1]:crop_end[1],
                    crop_start[2]:crop_end[2]
                ]
            else:
                cropped = img_array[
                    crop_start[0]:crop_end[0],
                    crop_start[1]:crop_end[1],
                    crop_start[2]:crop_end[2]
                ]

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
                cropped = img_array[
                    :,
                    crop_start[0]:crop_end[0],
                    crop_start[1]:crop_end[1],
                    crop_start[2]:crop_end[2]
                ]
            else:
                cropped = img_array[
                    crop_start[0]:crop_end[0],
                    crop_start[1]:crop_end[1],
                    crop_start[2]:crop_end[2]
                ]

            d[key] = cropped

        return d


def get_supervised_transforms(args, mode: str = None):
    """
    Get transforms for supervised fine-tuning on AMOS CT
    ADAPTED: CT Normalization, No LR-Swap
    """

    keys = ["image", "label"]

    # 1. 基础加载与几何校正
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),

        # [修改] 适配腹部 CT 的 Spacing (建议 1.5mm)
        # 你的运行脚本里会传入 --target_spacing 1.5 1.5 1.5
        Spacingd(
            keys=keys,
            pixdim=args.target_spacing,
            mode=("bilinear", "nearest")
        ),

        # [修改] 确保方向一致
        Orientationd(keys=keys, axcodes="RAS"),

        # [修改] CT 归一化 (替代原来的 PercentileNormalization)
        # 范围 [-175, 250] 是软组织/腹部常用窗宽，映射到 [0, 1]
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175.0,
            a_max=250.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        # 填充以防止随机裁剪出错 (96x96x96)
        SpatialPadd(
            keys=keys,
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            mode="constant"
        ),
    ]

    current_mode = mode or getattr(args, "mode", "train")

    # 2. 训练阶段的数据增强
    if current_mode == "train":
        transforms.extend([
            # [修改] 使用普通的随机裁剪 (RandCropByLabelClassesd 非常好，保留)
            # 它会保证采样的 patch 里包含标签，不会只采到背景
            RandCropByLabelClassesd(
                keys=keys,
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                num_classes=args.num_classes,
                num_samples=1,
            ),

            # [修改] 移除带有 lr_swap 逻辑的翻转，改为普通翻转
            # 腹部是不对称的，左右翻转(axis 0)要慎用，除非你想做增强
            # 这里我们只保留旋转和缩放，更加安全
            RandRotated(
                keys=keys,
                range_x=0.3, range_y=0.3, range_z=0.3,
                prob=0.3,
                mode=["bilinear", "nearest"],
            ),

            RandZoomd(
                keys=keys,
                min_zoom=0.85, max_zoom=1.15,
                prob=0.2,
                mode=["bilinear", "nearest"],
            ),

            # 强度增强 (保留，增加鲁棒性)
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
            RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.7, 1.3)),
            RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.5, 1.0)),
        ])

    # 3. 格式转换
    transforms.append(ToTensord(keys=keys))

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
