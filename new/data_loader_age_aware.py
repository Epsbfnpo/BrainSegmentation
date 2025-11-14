from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (CenterSpatialCropd, Compose, EnsureChannelFirstd,
                              EnsureTyped, LoadImaged, MapTransform, Orientationd,
                              RandAdjustContrastd, RandBiasFieldd, RandFlipd, RandGaussianNoised,
                              RandGaussianSmoothd, RandHistogramShiftd, RandRotated, RandScaleIntensityd,
                              RandShiftIntensityd, RandSpatialCropd, RandZoomd, Spacingd,
                              SpatialPadd)
from torch.utils.data import DistributedSampler


class ExtractAged(MapTransform):
    """Extract scan age from metadata and store as a tensor."""

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
    """Normalise intensities based on foreground percentiles."""

    def __init__(self, keys: Sequence[str], lower: float = 1.0, upper: float = 99.0):
        super().__init__(keys)
        self.lower = lower
        self.upper = upper

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key]
            if isinstance(image, torch.Tensor):
                array = image.cpu().numpy()
                meta = None
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
            d[key] = tensor if meta is None else type(image)(tensor, meta=meta)
        return d


class RemapLabelsd(MapTransform):
    """Remap labels 1..87 -> 0..86 and mark background as -1."""

    def __init__(self, keys: Sequence[str]):
        super().__init__(keys)

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            if isinstance(label, torch.Tensor):
                array = label.cpu().numpy()
                meta = None
            else:
                array = np.asarray(label)
                meta = None
            array = array.astype(np.int32)
            remapped = np.full_like(array, -1, dtype=np.int32)
            mask = array > 0
            remapped[mask] = array[mask] - 1
            tensor = torch.as_tensor(remapped, dtype=torch.float32)
            d[key] = tensor if meta is None else type(label)(tensor, meta=meta)
        return d


def _load_split(json_path: str) -> Tuple[List[Dict], List[Dict]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    train = data.get("training") or data.get("train") or []
    val = data.get("validation") or data.get("val") or []
    if not val:
        raise RuntimeError("Split JSON must provide a validation set")
    return train, val


def _process_items(items: List[Dict]) -> List[Dict]:
    processed: List[Dict] = []
    for item in items:
        entry = {
            "image": item["image"][0] if isinstance(item.get("image"), list) else item.get("image"),
            "label": item.get("label"),
        }
        if "metadata" in item:
            entry["metadata"] = item["metadata"]
        processed.append(entry)
    return processed


def _augmentation_transforms(args) -> Compose:
    spatial_size = (args.roi_x, args.roi_y, args.roi_z)
    aug = [
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotated(keys=["image", "label"], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.3, mode=["bilinear", "nearest"]),
        RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.9, max_zoom=1.1, mode=["trilinear", "nearest"]),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.5, 1.0)),
        RandBiasFieldd(keys=["image"], prob=0.1),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.3), prob=0.3),
        RandHistogramShiftd(keys=["image"], prob=0.1),
        RandSpatialCropd(keys=["image", "label"], roi_size=spatial_size, random_center=True, random_size=False),
        EnsureTyped(keys=["image", "label", "age"], track_meta=False),
    ]
    return Compose(aug)


def _base_transforms(args, mode: str) -> Compose:
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ExtractAged(),
    ]
    if getattr(args, "apply_spacing", True):
        transforms.append(
            Spacingd(keys=["image", "label"], pixdim=args.target_spacing, mode=("bilinear", "nearest"))
        )
    if getattr(args, "apply_orientation", True):
        transforms.append(Orientationd(keys=["image", "label"], axcodes="RAS"))
    transforms.append(PercentileNormalizationd(keys=["image"]))
    if args.foreground_only:
        transforms.append(RemapLabelsd(keys=["label"]))
    spatial_size = (args.roi_x, args.roi_y, args.roi_z)
    transforms.extend([
        SpatialPadd(keys=["image", "label"], spatial_size=spatial_size, method="end"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size),
        EnsureTyped(keys=["image", "label", "age"], track_meta=False),
    ])
    if mode == "train":
        transforms.append(_augmentation_transforms(args))
    return Compose(transforms)


def get_target_dataloaders(args,
                           *,
                           is_distributed: bool = False,
                           world_size: int = 1,
                           rank: int = 0):
    train_items, val_items = _load_split(args.split_json)
    train_items = _process_items(train_items)
    val_items = _process_items(val_items)

    train_transform = _base_transforms(args, mode="train")
    val_transform = _base_transforms(args, mode="val")

    use_cache = args.cache_rate > 0
    DatasetCls = CacheDataset if use_cache else Dataset
    dataset_kwargs = {}
    if use_cache:
        dataset_kwargs = {
            "cache_rate": args.cache_rate,
            "num_workers": min(args.cache_num_workers, 4),
            "progress": (rank == 0),
        }

    train_dataset = DatasetCls(train_items, transform=train_transform, **dataset_kwargs)
    val_dataset = Dataset(val_items, transform=val_transform)

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    return train_loader, val_loader
