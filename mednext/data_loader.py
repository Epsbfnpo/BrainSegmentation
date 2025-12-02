import json
import math
from typing import Dict, Optional, Sequence

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
from torch.utils.data import Sampler, WeightedRandomSampler


class DistributedWeightedSampler(Sampler[int]):
    """Weighted sampler compatible with DistributedDataParallel."""

    def __init__(
        self,
        weights: torch.Tensor,
        *,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True,
        seed: int = 0,
    ):
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D tensor")
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()

        self.weights = weights.to(dtype=torch.double)
        self.num_samples = int(math.ceil(self.weights.numel() / num_replicas))
        self.total_size = self.num_samples * num_replicas
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g)
        indices = indices.view(self.num_replicas, self.num_samples)
        yield from indices[self.rank].tolist()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


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
            meta = image.meta if isinstance(image, MetaTensor) else None
            array = image.cpu().numpy() if isinstance(image, (torch.Tensor, MetaTensor)) else np.asarray(image)
            mask = array > 0
            if mask.any():
                lo = np.percentile(array[mask], self.lower)
                hi = np.percentile(array[mask], self.upper)
                clipped = np.clip(array, lo, hi)
                norm = (clipped - lo) / (hi - lo + 1e-8)
                norm[~mask] = 0
            else:
                norm = array
            tensor = torch.as_tensor(norm, dtype=torch.float32)
            d[key] = MetaTensor(tensor, meta=meta) if meta is not None else tensor
        return d


class RemapLabelsd(MapTransform):
    """Remap labels 1..87 -> 0..86 and mark background as -1."""

    def __init__(self, keys: Sequence[str]):
        super().__init__(keys)

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            meta = label.meta if isinstance(label, MetaTensor) else None
            array = label.cpu().numpy() if isinstance(label, (torch.Tensor, MetaTensor)) else np.asarray(label)
            remapped = np.full_like(array, -1, dtype=np.int32)
            mask = array > 0
            remapped[mask] = array[mask] - 1
            tensor = torch.as_tensor(remapped, dtype=torch.int16)
            d[key] = MetaTensor(tensor, meta=meta) if meta is not None else tensor
        return d


def get_loader(args, is_distributed: bool = False, rank: int = 0, world_size: int = 1):
    with open(args.split_json, "r") as f:
        split = json.load(f)

    train_files = split["training"]
    val_files = split["validation"]

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(0.8, 0.8, 0.8), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            PercentileNormalizationd(keys=["image"]),
            RemapLabelsd(keys=["label"]),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                num_classes=args.out_channels,
                num_samples=1,
            ),
            RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.2, mode=["bilinear", "nearest"]),
            RandZoomd(keys=["image", "label"], min_zoom=0.85, max_zoom=1.15, prob=0.2, mode=["trilinear", "nearest"]),
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(0.8, 0.8, 0.8), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            PercentileNormalizationd(keys=["image"]),
            RemapLabelsd(keys=["label"]),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
    )

    weights = torch.ones(len(train_files), dtype=torch.double)

    if is_distributed:
        sampler = DistributedWeightedSampler(weights, num_replicas=world_size, rank=rank)
    else:
        sampler = WeightedRandomSampler(weights, len(weights))

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=4)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader
