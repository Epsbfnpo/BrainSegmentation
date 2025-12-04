from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset, MetaTensor
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Spacingd,
    SpatialPadd,
)
from torch.utils.data import DistributedSampler


class ExtractAged(MapTransform):
    """Extract scan age from metadata when available (defensive coding)."""

    def __init__(self, metadata_key: str = "metadata"):
        super().__init__(keys=None)
        self.metadata_key = metadata_key

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        metadata = d.get(self.metadata_key, {}) or {}
        age = None
        for key in ("scan_age", "PMA", "pma", "ga", "GA"):
            if key in metadata:
                try:
                    value = float(metadata[key])
                except (TypeError, ValueError):
                    continue
                if key in {"ga", "GA"} and "pna" in metadata:
                    try:
                        value += float(metadata.get("pna") or metadata.get("PNA"))
                    except (TypeError, ValueError):
                        pass
                age = value
                break
        if age is None:
            age = 40.0
        d["age"] = torch.tensor([age], dtype=torch.float32)
        return d


class PercentileNormalizationd(MapTransform):
    """Robust intensity normalisation using 1st and 99th percentiles."""

    def __init__(self, keys: Sequence[str], lower: float = 1.0, upper: float = 99.0):
        super().__init__(keys)
        self.lower = lower
        self.upper = upper

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key]
            if isinstance(image, (torch.Tensor, MetaTensor)):
                array = image.cpu().numpy()
                meta = image.meta if isinstance(image, MetaTensor) else None
            else:
                array = np.asarray(image)
                meta = None
            mask = array > 0
            if not mask.any():
                continue
            voxels = array[mask]
            lo = np.percentile(voxels, self.lower)
            hi = np.percentile(voxels, self.upper)
            if hi <= lo:
                norm = np.zeros_like(array)
            else:
                clipped = np.clip(array, lo, hi)
                norm = (clipped - lo) / (hi - lo)
            norm[~mask] = 0
            tensor = torch.as_tensor(norm, dtype=torch.float32)
            if meta is not None:
                d[key] = MetaTensor(tensor, meta=meta)
            else:
                d[key] = tensor
        return d


class RemapLabelsd(MapTransform):
    """Remap labels 1..87 -> 0..86 and mark background as -1."""

    def __init__(self, keys: Sequence[str]):
        super().__init__(keys)

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            if isinstance(label, (torch.Tensor, MetaTensor)):
                array = label.cpu().numpy()
                meta = label.meta if isinstance(label, MetaTensor) else None
            else:
                array = np.asarray(label)
                meta = None
            array = array.astype(np.int32)
            remapped = np.full_like(array, -1, dtype=np.int32)
            mask = array > 0
            remapped[mask] = array[mask] - 1
            tensor = torch.as_tensor(remapped, dtype=torch.float32)
            if meta is not None:
                d[key] = MetaTensor(tensor, meta=meta)
            else:
                d[key] = tensor
        return d


def _load_split(split_json: str) -> tuple[list[dict], list[dict]]:
    with open(split_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _convert(items: list[dict]) -> list[dict]:
        result: list[dict] = []
        for item in items:
            result.append(
                {
                    "image": item["image"][0] if isinstance(item.get("image"), list) else item["image"],
                    "label": item["label"],
                }
            )
        return result

    return _convert(data.get("training", [])), _convert(data.get("validation", []))


def get_loader(
    args,
    *,
    is_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    train_files, val_files = _load_split(args.split_json)
    spatial_size = (args.roi_x, args.roi_y, args.roi_z)

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=args.target_spacing, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            PercentileNormalizationd(keys=["image"]),
            RemapLabelsd(keys=["label"]),
            ExtractAged(),
            SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                num_classes=args.out_channels,
                num_samples=1,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label", "age"], track_meta=False),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=args.target_spacing, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            PercentileNormalizationd(keys=["image"]),
            RemapLabelsd(keys=["label"]),
            ExtractAged(),
            EnsureTyped(keys=["image", "label", "age"], track_meta=False),
        ]
    )

    cache_kwargs = {"cache_rate": args.cache_rate, "num_workers": args.cache_num_workers} if args.cache_rate > 0 else {}

    train_ds = CacheDataset(data=train_files, transform=train_transforms, **cache_kwargs)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, **cache_kwargs)

    train_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


__all__ = [
    "get_loader",
    "ExtractAged",
    "PercentileNormalizationd",
    "RemapLabelsd",
]
