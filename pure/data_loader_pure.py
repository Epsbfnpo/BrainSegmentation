"""Utility dataloaders for pure fine-tuning on the target dataset."""

from __future__ import annotations

import json
from typing import List, Sequence, Tuple

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
)


def _load_split(split_json: str) -> Tuple[List[dict], List[dict]]:
    with open(split_json, "r") as f:
        data = json.load(f)

    train_list = list(data.get("training", []))
    val_list = list(data.get("validation", []))

    if not train_list:
        raise ValueError(f"No training samples found in {split_json}")
    if not val_list:
        # fall back to training list for validation if dedicated split missing
        val_list = train_list

    return train_list, val_list


def _compose_transforms(
    roi_size: Sequence[int],
    *,
    is_training: bool,
    apply_spacing: bool,
    target_spacing: Sequence[float],
    apply_orientation: bool,
) -> Compose:
    transforms: List = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
    ]

    if apply_orientation:
        transforms.append(Orientationd(keys=["image", "label"], axcodes="RAS"))
    if apply_spacing:
        transforms.append(
            Spacingd(keys=["image", "label"], pixdim=tuple(target_spacing), mode=("bilinear", "nearest"))
        )

    if is_training:
        transforms.extend(
            [
                RandSpatialCropd(keys=["image", "label"], roi_size=tuple(roi_size), random_center=True, random_size=False),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
                EnsureTyped(keys=["image", "label"]),
                ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        transforms.extend(
            [
                CenterSpatialCropd(keys=["image", "label"], roi_size=tuple(roi_size)),
                EnsureTyped(keys=["image", "label"]),
                ToTensord(keys=["image", "label"]),
            ]
        )

    return Compose(transforms)


def create_dataloaders(
    split_json: str,
    *,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    cache_rate: float,
    cache_num_workers: int,
    roi_size: Sequence[int],
    apply_spacing: bool,
    target_spacing: Sequence[float],
    apply_orientation: bool,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders for pure fine-tuning."""

    train_list, val_list = _load_split(split_json)

    train_transforms = _compose_transforms(
        roi_size,
        is_training=True,
        apply_spacing=apply_spacing,
        target_spacing=target_spacing,
        apply_orientation=apply_orientation,
    )
    val_transforms = _compose_transforms(
        roi_size,
        is_training=False,
        apply_spacing=apply_spacing,
        target_spacing=target_spacing,
        apply_orientation=apply_orientation,
    )

    if cache_rate > 0:
        train_ds = CacheDataset(
            data=train_list,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=cache_num_workers,
        )
    else:
        train_ds = Dataset(data=train_list, transform=train_transforms)

    val_ds = Dataset(data=val_list, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


__all__ = ["create_dataloaders"]
