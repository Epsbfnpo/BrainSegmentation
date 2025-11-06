"""Dataloading utilities for the texture-focused adaptation pipeline."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    Spacingd,
    ToTensord,
)
from monai.transforms.transform import MapTransform


def _load_split(path: str) -> Tuple[List[dict], List[dict]]:
    with open(path, "r") as f:
        payload = json.load(f)

    train_records = list(payload.get("training", []))
    val_records = list(payload.get("validation", [])) or list(payload.get("training", []))

    if not train_records:
        raise ValueError(f"No training entries found in {path}")

    return train_records, val_records


def _annotate_domain(records: Iterable[dict], domain_id: int) -> List[dict]:
    annotated: List[dict] = []
    for sample in records:
        sample_copy = deepcopy(sample)
        sample_copy.setdefault("metadata", {})
        sample_copy["metadata"]["domain_id"] = int(domain_id)
        annotated.append(sample_copy)
    return annotated


class AddDomainId(MapTransform):
    """Attach the domain id stored in metadata to the data dict."""

    def __init__(self, keys: Sequence[str], domain_key: str = "metadata", field: str = "domain_id") -> None:
        super().__init__(keys)
        self.domain_key = domain_key
        self.field = field

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        metadata = d.get(self.domain_key, {})
        domain_val = metadata.get(self.field)
        if domain_val is None:
            raise KeyError(
                "Domain id missing from metadata. Ensure the split jsons were processed through "
                "_annotate_domain before creating datasets."
            )
        d["domain"] = torch.as_tensor([int(domain_val)], dtype=torch.long)
        return d


def _build_transforms(
    roi_size: Sequence[int],
    *,
    training: bool,
    apply_spacing: bool,
    target_spacing: Sequence[float],
    apply_orientation: bool,
    intensity_norm: bool,
) -> Compose:
    transforms: List = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
    ]

    if apply_orientation:
        transforms.append(Orientationd(keys=["image", "label"], axcodes="RAS"))
    if apply_spacing:
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(target_spacing),
                mode=("bilinear", "nearest"),
            )
        )

    if training:
        transforms.extend(
            [
                RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=tuple(roi_size),
                    random_center=True,
                    random_size=False,
                ),
                RandZoomd(keys=["image", "label"], prob=0.15, min_zoom=0.9, max_zoom=1.1, mode=("trilinear", "nearest")),
                RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.4)),
                RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.3),
                RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.02),
                RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
                RandGibbsNoised(keys=["image"], prob=0.1, alpha=(0.8, 1.2)),
                RandHistogramShiftd(keys=["image"], prob=0.2, num_control_points=5),
            ]
        )
    else:
        transforms.append(
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=tuple(roi_size),
                random_center=False,
                random_size=False,
            )
        )

    transforms.append(AddDomainId(keys=["image", "label", "metadata"]))

    if intensity_norm:
        transforms.append(EnsureTyped(keys=["image", "label", "domain"]))
    else:
        transforms.append(EnsureTyped(keys=["image", "label", "domain"]))
    transforms.append(ToTensord(keys=["image", "label", "domain"]))

    return Compose(transforms)


def create_texture_dataloaders(
    source_split_json: str,
    target_split_json: str,
    *,
    roi_size: Sequence[int],
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    cache_rate: float,
    cache_num_workers: int,
    apply_spacing: bool,
    target_spacing: Sequence[float],
    apply_orientation: bool,
    intensity_norm: bool = True,
    source_repeat: int = 1,
    target_repeat: int = 1,
) -> Dict[str, DataLoader]:
    """Create dataloaders for the texture pipeline.

    Returns a dictionary with ``train``, ``val_source`` and ``val_target`` loaders.
    """

    source_train, source_val = _load_split(source_split_json)
    target_train, target_val = _load_split(target_split_json)

    source_train = _annotate_domain(source_train, domain_id=0)
    target_train = _annotate_domain(target_train, domain_id=1)
    source_val = _annotate_domain(source_val, domain_id=0)
    target_val = _annotate_domain(target_val, domain_id=1)

    combined_train: List[dict] = []
    for _ in range(max(1, source_repeat)):
        combined_train.extend(deepcopy(source_train))
    for _ in range(max(1, target_repeat)):
        combined_train.extend(deepcopy(target_train))

    train_transforms = _build_transforms(
        roi_size,
        training=True,
        apply_spacing=apply_spacing,
        target_spacing=target_spacing,
        apply_orientation=apply_orientation,
        intensity_norm=intensity_norm,
    )
    eval_transforms = _build_transforms(
        roi_size,
        training=False,
        apply_spacing=apply_spacing,
        target_spacing=target_spacing,
        apply_orientation=apply_orientation,
        intensity_norm=intensity_norm,
    )

    if cache_rate > 0:
        train_ds = CacheDataset(
            data=combined_train,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=cache_num_workers,
        )
    else:
        train_ds = Dataset(data=combined_train, transform=train_transforms)

    source_val_ds = Dataset(data=source_val, transform=eval_transforms)
    target_val_ds = Dataset(data=target_val, transform=eval_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_source_loader = DataLoader(
        source_val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_target_loader = DataLoader(
        target_val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return {
        "train": train_loader,
        "val_source": val_source_loader,
        "val_target": val_target_loader,
    }


__all__ = ["create_texture_dataloaders"]
