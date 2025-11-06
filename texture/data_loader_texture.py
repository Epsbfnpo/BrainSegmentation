"""Dataloaders for texture-focused domain adaptation."""

from __future__ import annotations

import json
from typing import Iterable, List, Sequence, Tuple

import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    RandSpatialCropd,
    RandZoomd,
    Spacingd,
    ToTensord,
)

from .texture_transforms import TextureStatsd

__all__ = ["create_texture_dataloaders"]


def _load_split(split_json: str) -> Tuple[List[dict], List[dict]]:
    with open(split_json, "r") as f:
        data = json.load(f)

    train_list = list(data.get("training", []))
    val_list = list(data.get("validation", []))

    if not train_list:
        raise ValueError(f"No training samples found in {split_json}")
    if not val_list:
        val_list = train_list

    return train_list, val_list


def _inject_domain(samples: Iterable[dict], domain_index: int) -> List[dict]:
    enriched: List[dict] = []
    for sample in samples:
        enriched.append({**sample, "domain": int(domain_index)})
    return enriched


def _compose_transforms(
    roi_size: Sequence[int],
    *,
    is_training: bool,
    apply_spacing: bool,
    target_spacing: Sequence[float],
    apply_orientation: bool,
    texture_prefix: str = "texture_stats",
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
                RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=tuple(roi_size),
                    random_center=True,
                    random_size=False,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.5)),
                RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
                RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
                RandHistogramShiftd(keys=["image"], prob=0.2, num_control_points=4, range_start=-0.05, range_end=0.05),
                RandZoomd(keys=["image", "label"], prob=0.15, min_zoom=0.9, max_zoom=1.1, keep_size=True),
            ]
        )
    else:
        transforms.append(
            CenterSpatialCropd(keys=["image", "label"], roi_size=tuple(roi_size)),
        )

    transforms.append(TextureStatsd(keys=["image"], prefix=texture_prefix, mask_key="label"))
    transforms.append(
        Lambdad(keys=["domain"], func=lambda x: torch.tensor(int(x), dtype=torch.long))
    )
    transforms.append(ToTensord(keys=["image", "label"]))
    transforms.append(
        EnsureTyped(keys=["image", "label", texture_prefix], dtype=torch.float32, device=torch.device("cpu"))
    )

    return Compose(transforms)


def create_texture_dataloaders(
    source_split_json: str,
    target_split_json: str,
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
    source_domain_index: int = 0,
    target_domain_index: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    source_train, _ = _load_split(source_split_json)
    target_train, target_val = _load_split(target_split_json)

    train_samples = _inject_domain(source_train, source_domain_index) + _inject_domain(
        target_train, target_domain_index
    )
    val_samples = _inject_domain(target_val, target_domain_index)

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
            data=train_samples,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=cache_num_workers,
        )
    else:
        train_ds = Dataset(data=train_samples, transform=train_transforms)

    val_ds = Dataset(data=val_samples, transform=val_transforms)

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
