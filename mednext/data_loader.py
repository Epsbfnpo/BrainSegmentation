from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

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
    RandGaussianNoised,
    RandRotated,
    RandZoomd,
    Spacingd,
    SpatialPadd,
)
from torch.utils.data import DistributedSampler, Sampler, WeightedRandomSampler


class DistributedWeightedSampler(Sampler[int]):
    def __init__(self, weights: torch.Tensor, *, num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 replacement: bool = True, seed: int = 0):
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D tensor")
        if weights.numel() == 0:
            raise ValueError("weights tensor must be non-empty")

        if num_replicas is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("num_replicas must be provided when not in a distributed context")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("rank must be provided when not in a distributed context")
            rank = torch.distributed.get_rank()

        self.weights = weights.to(dtype=torch.double)
        self.num_samples = int(np.ceil(self.weights.numel() / num_replicas))
        self.total_size = self.num_samples * num_replicas
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch)
        probs = self.weights.clone()
        total = float(probs.sum())
        if total <= 0:
            probs.fill_(1.0 / probs.numel())
        else:
            probs.div_(total)
        indices = torch.multinomial(probs, self.total_size, self.replacement, generator=g)
        indices = indices.view(self.num_replicas, self.num_samples)
        yield from indices[self.rank].tolist()

    def __len__(self):  # pragma: no cover - trivial
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class PercentileNormalizationd(MapTransform):
    def __init__(self, keys: Sequence[str], lower: float = 1.0, upper: float = 99.0):
        super().__init__(keys)
        self.lower = lower
        self.upper = upper

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key]
            if isinstance(image, MetaTensor):
                array = image.cpu().numpy()
                meta = image.meta
            else:
                array = np.asarray(image)
                meta = None
            mask = array > 0
            if mask.any():
                voxels = array[mask]
                lo = np.percentile(voxels, self.lower)
                hi = np.percentile(voxels, self.upper)
                clipped = np.clip(array, lo, hi)
                norm = (clipped - lo) / (hi - lo + 1e-8)
                norm[~mask] = 0
            else:
                norm = np.zeros_like(array)
            tensor = torch.as_tensor(norm, dtype=torch.float32)
            d[key] = MetaTensor(tensor, meta=meta) if meta is not None else tensor
        return d


class RemapLabelsd(MapTransform):
    def __init__(self, keys: Sequence[str]):
        super().__init__(keys)

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            if isinstance(label, MetaTensor):
                array = label.cpu().numpy()
                meta = label.meta
            else:
                array = np.asarray(label)
                meta = None
            array = array.astype(np.int32)
            remapped = np.full_like(array, -1, dtype=np.int32)
            mask = array > 0
            remapped[mask] = array[mask] - 1
            tensor = torch.as_tensor(remapped, dtype=torch.int64)
            d[key] = MetaTensor(tensor, meta=meta) if meta is not None else tensor
        return d


def _load_split(json_path: str) -> Tuple[List[Dict], List[Dict]]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Split JSON not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    train = data.get("training") or data.get("train") or []
    val = data.get("validation") or data.get("val") or []
    if not val:
        raise RuntimeError("Split JSON must provide a validation set")
    return train, val


def _prepare_transforms(args, *, mode: str) -> Compose:
    spatial_size = (args.roi_x, args.roi_y, args.roi_z)
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(0.8, 0.8, 0.8), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        PercentileNormalizationd(keys=["image"]),
        RemapLabelsd(keys=["label"]),
        SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
    ]

    if mode == "train":
        transforms.append(
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                num_classes=args.out_channels,
                num_samples=1,
            )
        )
        transforms.extend(
            [
                RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.3,
                            mode=["bilinear", "nearest"]),
                RandZoomd(keys=["image", "label"], min_zoom=0.85, max_zoom=1.15, prob=0.2,
                          mode=["trilinear", "nearest"]),
                RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
            ]
        )
    transforms.append(EnsureTyped(keys=["image", "label"], track_meta=False))
    return Compose(transforms)


def get_loader(args, *, is_distributed: bool = False, rank: int = 0, world_size: int = 1):
    train_items, val_items = _load_split(args.split_json)
    train_transforms = _prepare_transforms(args, mode="train")
    val_transforms = _prepare_transforms(args, mode="val")

    weights = torch.ones(len(train_items)).double()
    if is_distributed:
        sampler: Sampler[int] = DistributedWeightedSampler(weights, num_replicas=world_size, rank=rank)
    else:
        sampler = WeightedRandomSampler(weights, len(weights))

    train_ds = CacheDataset(train_items, transform=train_transforms, cache_rate=args.cache_rate, num_workers=4)
    val_ds = Dataset(val_items, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader
