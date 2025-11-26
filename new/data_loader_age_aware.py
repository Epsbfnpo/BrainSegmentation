from __future__ import annotations

import json
import os
import math
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset, MetaTensor
from monai.transforms import (CenterSpatialCropd, Compose, CopyItemsd, EnsureChannelFirstd,
                              EnsureTyped, LoadImaged, MapTransform, Orientationd,
                              RandAdjustContrastd, RandBiasFieldd, RandCropByLabelClassesd,
                              RandGaussianNoised, RandGaussianSmoothd, RandHistogramShiftd,
                              RandRotated, RandScaleIntensityd, RandShiftIntensityd, RandZoomd,
                              Spacingd, SpatialPadd, Randomizable)
from torch.utils.data import DistributedSampler, WeightedRandomSampler, Sampler

from age_aware_modules import prepare_class_ratios


class DistributedWeightedSampler(Sampler[int]):
    """Weighted sampler compatible with DistributedDataParallel.

    Each replica draws ``num_samples`` indices (with replacement) according to the
    provided weights. The sampled subsets are disjoint by construction, ensuring
    that the global sampling distribution matches the requested weights.
    """

    def __init__(self,
                 weights: torch.Tensor,
                 *,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 replacement: bool = True,
                 seed: int = 0):
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
        self.num_samples = int(math.ceil(self.weights.numel() / num_replicas))
        self.total_size = self.num_samples * num_replicas
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0

        self.generator = torch.Generator(device="cpu")

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

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


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


class RandFlipLateralityd(Randomizable, MapTransform):
    """Flip volumes along an axis and swap laterality label pairs when required."""

    def __init__(self, keys: Sequence[str], spatial_axis: int, prob: float = 0.5,
                 swap_label_pairs: Optional[Sequence[Tuple[int, int]]] = None):
        super().__init__(keys)
        self.spatial_axis = spatial_axis
        self.prob = float(prob)
        self.swap_label_pairs = list(swap_label_pairs or [])

    def randomize(self, data=None):
        self._do_transform = self.R.rand() < self.prob

    def _swap_labels(self, array: torch.Tensor) -> torch.Tensor:
        if not self.swap_label_pairs:
            return array
        result = array.clone()
        for a, b in self.swap_label_pairs:
            mask_a = result == a
            mask_b = result == b
            tmp = torch.zeros_like(result)
            tmp[mask_a] = b
            tmp[mask_b] = a
            result = torch.where(mask_a | mask_b, tmp, result)
        return result

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            arr = d[key]
            meta = None
            if isinstance(arr, MetaTensor):
                meta = arr.meta
                tensor = arr.as_tensor()
            else:
                tensor = torch.as_tensor(arr)

            axis = 1 + self.spatial_axis if tensor.ndim == 4 else self.spatial_axis
            flipped = torch.flip(tensor, dims=[axis])

            if key == "label" and self.spatial_axis == 0:
                flipped = self._swap_labels(flipped)

            if meta is not None:
                d[key] = MetaTensor(flipped, meta=meta)
            else:
                d[key] = flipped
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


def _load_test_split(json_path: str) -> List[Dict]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Split JSON not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    for key in ("testing", "test", "test_set"):
        items = data.get(key) or []
        if items:
            return _process_items(items)
    raise RuntimeError("Split JSON must provide a testing set")


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


def _load_lr_pairs(args) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    json_path = getattr(args, "laterality_pairs_json", None)
    if json_path and os.path.exists(json_path):
        with open(json_path, "r") as f:
            payload = json.load(f)
        for item in payload:
            a, b = item
            if args.foreground_only:
                a -= 1
                b -= 1
            pairs.append((int(a), int(b)))
    return pairs


def _augmentation_transforms(args,
                             *,
                             laterality_pairs: Optional[Sequence[Tuple[int, int]]] = None) -> Compose:
    spatial_size = (args.roi_x, args.roi_y, args.roi_z)
    aug = [
        RandFlipLateralityd(keys=["image", "label"], spatial_axis=0, prob=0.5, swap_label_pairs=laterality_pairs),
        RandFlipLateralityd(keys=["image", "label"], spatial_axis=1, prob=0.5, swap_label_pairs=[]),
        RandFlipLateralityd(keys=["image", "label"], spatial_axis=2, prob=0.5, swap_label_pairs=[]),
        RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.35, mode=["bilinear", "nearest"]),
        RandZoomd(keys=["image", "label"], prob=0.25, min_zoom=0.85, max_zoom=1.15, mode=["trilinear", "nearest"]),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.5, 1.0)),
        RandBiasFieldd(keys=["image"], prob=0.1),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.3), prob=0.3),
        RandHistogramShiftd(keys=["image"], prob=0.1),
    ]
    return Compose(aug)


def _base_transforms(args,
                     mode: str,
                     *,
                     class_crop_ratios: Optional[Sequence[float]] = None,
                     laterality_pairs: Optional[Sequence[Tuple[int, int]]] = None) -> Compose:
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
    transforms.append(SpatialPadd(keys=["image", "label"], spatial_size=spatial_size, method="end"))

    if mode == "train" and getattr(args, "use_label_crop", True):
        ratios = class_crop_ratios or [1.0] * args.out_channels
        transforms.append(
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                ratios=ratios,
                num_classes=args.out_channels,
                num_samples=max(1, int(getattr(args, "label_crop_samples", 1))),
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            )
        )
    else:
        transforms.append(CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size))

    transforms.append(EnsureTyped(keys=["image", "label", "age"], track_meta=False))

    if mode == "train":
        transforms.append(CopyItemsd(keys=["image", "label"], times=1, names=["image_clean", "label_clean"]))
        transforms.append(_augmentation_transforms(args, laterality_pairs=laterality_pairs))
        transforms.append(EnsureTyped(keys=["image", "label", "image_clean", "label_clean", "age"], track_meta=False))
    return Compose(transforms)


def _inference_transforms(args, *, keep_meta: bool = True) -> Compose:
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
    if getattr(args, "foreground_only", False):
        transforms.append(RemapLabelsd(keys=["label"]))
    transforms.append(EnsureTyped(keys=["image", "label", "age"], track_meta=keep_meta))
    return Compose(transforms)


def _load_class_ratios(args, *, is_main: bool) -> Optional[List[float]]:
    stats_path = getattr(args, "volume_stats", None)
    if not stats_path or not os.path.exists(stats_path):
        if is_main:
            print("⚠️  No volume_stats.json found; RandCropByLabelClassesd will use uniform ratios")
        return None
    with open(stats_path, "r") as f:
        prior_data = json.load(f)
    ratios = prepare_class_ratios(
        prior_data,
        expected_num_classes=args.out_channels,
        foreground_only=args.foreground_only,
        is_main=is_main,
        context="Target volume prior",
    )
    ratios = ratios.astype(np.float64)
    if ratios.sum() <= 0:
        return None
    ratios = ratios / ratios.sum()
    return ratios.tolist()


def _compute_sample_weights(items: List[Dict],
                             args,
                             *,
                             class_ratios: Optional[Sequence[float]] = None,
                             is_main: bool = False) -> Optional[torch.Tensor]:
    if not getattr(args, "enable_weighted_sampling", False):
        return None
    if not class_ratios:
        if is_main:
            print("⚠️  Weighted sampling requested but no class ratios available; skipping")
        return None

    inv = 1.0 / (np.asarray(class_ratios, dtype=np.float64) + 1e-6)
    weights = []
    for item in items:
        label_path = item.get("label")
        if not label_path or not os.path.exists(label_path):
            weights.append(1.0)
            continue
        try:
            label_obj = nib.load(label_path)
            label = np.asarray(label_obj.dataobj).astype(np.int16)
        except Exception:
            weights.append(1.0)
            continue
        label = label.astype(np.int32)
        if args.foreground_only:
            label = np.where(label > 0, label - 1, -1)
        unique = np.unique(label)
        unique = unique[(unique >= 0) & (unique < inv.shape[0])]
        if unique.size == 0:
            weights.append(1.0)
        else:
            weights.append(float(inv[unique].mean()))
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        return None
    weights = weights / weights.sum() * len(weights)
    if is_main:
        print("✅ Weighted sampling enabled for target dataset")
        print(f"   Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    return torch.as_tensor(weights, dtype=torch.double)


def get_target_dataloaders(args,
                           *,
                           is_distributed: bool = False,
                           world_size: int = 1,
                           rank: int = 0):
    train_items, val_items = _load_split(args.split_json)
    train_items = _process_items(train_items)
    val_items = _process_items(val_items)

    is_main = (not is_distributed) or rank == 0
    class_ratios = _load_class_ratios(args, is_main=is_main)
    laterality_pairs = _load_lr_pairs(args)

    train_transform = _base_transforms(
        args,
        mode="train",
        class_crop_ratios=class_ratios,
        laterality_pairs=laterality_pairs,
    )
    val_transform = _base_transforms(
        args,
        mode="val",
        class_crop_ratios=None,
        laterality_pairs=None,
    )

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

    sample_weights = _compute_sample_weights(train_items, args, class_ratios=class_ratios, is_main=is_main)

    if is_distributed:
        if sample_weights is not None:
            train_sampler = DistributedWeightedSampler(
                sample_weights,
                num_replicas=world_size,
                rank=rank,
            )
            if is_main:
                print("✅ Distributed weighted sampling enabled for target dataset")
                print(f"   Local samples per replica: {len(train_sampler)}")
        else:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        sample_weights = None
    else:
        if sample_weights is not None:
            train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
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


def get_target_test_loader(args, *, keep_meta: bool = True):
    test_items = _load_test_split(args.split_json)
    test_transform = _inference_transforms(args, keep_meta=keep_meta)
    test_dataset = Dataset(test_items, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )
    return test_loader
