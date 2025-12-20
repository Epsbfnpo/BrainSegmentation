from __future__ import annotations
import math
import numpy as np
import torch
from typing import Dict, Optional, Sequence, Tuple
from monai.data import CacheDataset, DataLoader, Dataset, MetaTensor
from monai.transforms import (
    CenterSpatialCropd, Compose, CopyItemsd, EnsureChannelFirstd, EnsureTyped,
    LoadImaged, MapTransform, Orientationd, RandCropByLabelClassesd,
    RandFlipd, RandRotate90d, RandShiftIntensityd, Spacingd, SpatialPadd
)
from torch.utils.data import DistributedSampler, WeightedRandomSampler, Sampler


class PercentileNormalizationd(MapTransform):
    def __init__(self, keys, lower=1.0, upper=99.0):
        super().__init__(keys)
        self.lower = lower
        self.upper = upper

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            arr = img.cpu().numpy() if isinstance(img, MetaTensor) else np.asarray(img)
            mask = arr > 0
            if mask.any():
                lo, hi = np.percentile(arr[mask], [self.lower, self.upper])
                arr = np.clip(arr, lo, hi)
                arr = (arr - lo) / (hi - lo + 1e-8)
                arr[~mask] = 0
            d[key] = torch.as_tensor(arr).float()
        return d


class RemapLabelsd(MapTransform):
    def __init__(self, keys): super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            lbl = d[key]
            arr = lbl.cpu().numpy() if isinstance(lbl, MetaTensor) else np.asarray(lbl)
            arr = arr.astype(np.int32)
            # -1: Background, 0~86: Foreground
            arr = np.where(arr > 0, arr - 1, -1).astype(np.int32)
            d[key] = torch.as_tensor(arr).float()
        return d


def _load_split(json_path):
    import json
    with open(json_path) as f: data = json.load(f)

    def _proc(l): return [{'image': i['image'], 'label': i['label']} for i in l]

    return _proc(data['training']), _proc(data['validation'])


def get_target_dataloaders(args, is_distributed=False, rank=0, world_size=1):
    train_files, val_files = _load_split(args.split_json)
    roi_size = (args.roi_x, args.roi_y, args.roi_z)

    train_tx = Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=args.target_spacing, mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        PercentileNormalizationd(keys=["image"]),
        RemapLabelsd(keys=["label"]),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        RandCropByLabelClassesd(keys=["image", "label"], label_key="label", spatial_size=roi_size,
                                num_classes=args.out_channels, num_samples=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"], track_meta=False),
    ])

    val_tx = Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=args.target_spacing, mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        PercentileNormalizationd(keys=["image"]),
        RemapLabelsd(keys=["label"]),
        EnsureTyped(keys=["image", "label"], track_meta=False),
    ])

    sampler = DistributedSampler(train_files, num_replicas=world_size, rank=rank,
                                 shuffle=True) if is_distributed else None
    train_ds = CacheDataset(train_files, transform=train_tx, cache_rate=args.cache_rate,
                            num_workers=args.cache_num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(Dataset(val_files, transform=val_tx), batch_size=1, shuffle=False, num_workers=2,
                            pin_memory=True)
    return train_loader, val_loader
