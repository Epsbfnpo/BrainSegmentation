import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import (CenterSpatialCropd, Compose, EnsureChannelFirstd,
                              EnsureTyped, LoadImaged, MapTransform, Orientationd,
                              Randomizable, RandAdjustContrastd, RandBiasFieldd,
                              RandGaussianNoised, RandGaussianSmoothd, RandHistogramShiftd,
                              RandRotated, RandScaleIntensityd, RandShiftIntensityd,
                              RandZoomd, Spacingd, SpatialPadd)
from torch.utils.data import Dataset


class ExtractAged(MapTransform):
    """Extract scan age from nested metadata."""

    def __init__(self, metadata_key: str = "metadata"):
        super().__init__(keys=None)
        self.metadata_key = metadata_key

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        metadata = d.get(self.metadata_key, {}) or {}
        age = None
        for key in ("scan_age", "PMA", "pma", "ga", "GA"):
            if key not in metadata:
                continue
            try:
                value = float(metadata[key])
            except (TypeError, ValueError):
                continue
            if key in {"ga", "GA"}:
                for pn_key in ("pna", "PNA"):
                    if pn_key in metadata:
                        try:
                            value += float(metadata[pn_key])
                        except (TypeError, ValueError):
                            pass
                        break
            age = value
            break
        if age is None:
            age = 40.0
        d["age"] = torch.tensor([age], dtype=torch.float32)
        return d


class PercentileNormalizationd(MapTransform):
    """Normalise intensities inside the brain mask via percentiles."""

    def __init__(self, keys: Sequence[str], lower: float = 1.0, upper: float = 99.0):
        super().__init__(keys)
        self.lower = lower
        self.upper = upper

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key]
            meta = None
            if isinstance(image, MetaTensor):
                array = image.as_tensor().cpu().numpy()
                meta = image.meta
            elif isinstance(image, torch.Tensor):
                array = image.cpu().numpy()
            else:
                array = np.asarray(image)
            mask = array > 0
            if not np.any(mask):
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
    """Remap Draw-EM style labels to foreground-only indices."""

    def __init__(self, keys: Sequence[str]):
        super().__init__(keys)

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            meta = None
            if isinstance(label, MetaTensor):
                array = label.as_tensor().cpu().numpy()
                meta = label.meta
            elif isinstance(label, torch.Tensor):
                array = label.cpu().numpy()
            else:
                array = np.asarray(label)
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
    """Flip a 3D volume and swap paired labels when needed."""

    def __init__(self, keys: Sequence[str], spatial_axis: int, prob: float = 0.5,
                 swap_label_pairs: Optional[Sequence[Tuple[int, int]]] = None):
        super().__init__(keys)
        self.spatial_axis = int(spatial_axis)
        self.prob = float(prob)
        self.swap_label_pairs = list(swap_label_pairs or [])
        self._do_transform = False

    def randomize(self, data=None):
        self._do_transform = self.R.rand() < self.prob

    def _swap_labels(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.swap_label_pairs:
            return tensor
        result = tensor.clone()
        for a, b in self.swap_label_pairs:
            mask_a = result == a
            mask_b = result == b
            result[mask_a] = b
            result[mask_b] = a
        return result

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        for key in self.key_iterator(d):
            tensor = d[key]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            flipped = torch.flip(tensor, dims=[self.spatial_axis + 1])
            if key == "label":
                flipped = self._swap_labels(flipped)
            d[key] = flipped
        return d


def _load_split_items(split_json: str, subset: str) -> List[Dict]:
    path = Path(split_json)
    if not path.exists():
        raise FileNotFoundError(f"Split JSON not found: {split_json}")
    payload = json.loads(path.read_text())
    if subset not in payload:
        raise KeyError(f"Subset '{subset}' not present in {split_json}")
    return list(payload[subset])


def _load_laterality_pairs(json_path: Optional[str], foreground_only: bool) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if not json_path:
        return pairs
    path = Path(json_path)
    if not path.exists():
        return pairs
    data = json.loads(path.read_text())
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        a, b = int(item[0]), int(item[1])
        if foreground_only:
            a -= 1
            b -= 1
        pairs.append((a, b))
    return pairs


@dataclass
class DatasetConfig:
    split_json: str
    subset: str
    roi_size: Tuple[int, int, int]
    target_spacing: Tuple[float, float, float]
    apply_spacing: bool = True
    apply_orientation: bool = True
    foreground_only: bool = True
    laterality_pairs_json: Optional[str] = None


def _base_transforms(cfg: DatasetConfig, mode: str) -> Compose:
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ExtractAged(),
    ]
    if cfg.apply_spacing:
        transforms.append(
            Spacingd(keys=["image", "label"], pixdim=cfg.target_spacing, mode=("bilinear", "nearest"))
        )
    if cfg.apply_orientation:
        transforms.append(Orientationd(keys=["image", "label"], axcodes="RAS"))
    transforms.append(PercentileNormalizationd(keys=["image"]))
    if cfg.foreground_only:
        transforms.append(RemapLabelsd(keys=["label"]))
    spatial_size = cfg.roi_size
    transforms.append(SpatialPadd(keys=["image", "label"], spatial_size=spatial_size, method="end"))
    transforms.append(CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size))
    transforms.append(EnsureTyped(keys=["image", "label", "age"], track_meta=False))

    if mode == "train":
        laterality_pairs = _load_laterality_pairs(cfg.laterality_pairs_json, cfg.foreground_only)
        aug = [
            RandFlipLateralityd(keys=["image", "label"], spatial_axis=0, prob=0.5,
                                swap_label_pairs=laterality_pairs),
            RandFlipLateralityd(keys=["image", "label"], spatial_axis=1, prob=0.5,
                                swap_label_pairs=[]),
            RandFlipLateralityd(keys=["image", "label"], spatial_axis=2, prob=0.5,
                                swap_label_pairs=[]),
            RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3,
                        prob=0.35, mode=["bilinear", "nearest"]),
            RandZoomd(keys=["image", "label"], prob=0.25, min_zoom=0.85, max_zoom=1.15,
                      mode=["trilinear", "nearest"]),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
            RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.5, 1.0)),
            RandBiasFieldd(keys=["image"], prob=0.1),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.3), prob=0.3),
            RandHistogramShiftd(keys=["image"], prob=0.1),
            EnsureTyped(keys=["image", "label", "age"], track_meta=False),
        ]
        transforms.extend(aug)
    return Compose(transforms)


def _derive_case_id(item: Dict) -> str:
    metadata = item.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("subject_id"):
        return str(metadata["subject_id"])
    image_path = item.get("image") or "case"
    return Path(image_path).stem


class PPREMOVolumeDataset(Dataset):
    """Return whole preprocessed volumes for validation/test."""

    def __init__(self, cfg: DatasetConfig, mode: str = "val"):
        self.items = _load_split_items(cfg.split_json, cfg.subset)
        self.case_ids = [_derive_case_id(item) for item in self.items]
        self.transform = _base_transforms(cfg, mode)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict:
        item = dict(self.items[index])
        data = self.transform(item)
        image = data["image"].float()
        label = data.get("label")
        if label is not None:
            label = label.long()
        return {
            "image": image,
            "label": label,
            "case_id": self.case_ids[index],
        }


class PPREMOSliceDataset(Dataset):
    """Return axial slices sampled from preprocessed volumes."""

    def __init__(self, cfg: DatasetConfig, slices_per_volume: Optional[int] = None):
        self.items = _load_split_items(cfg.split_json, cfg.subset)
        self.case_ids = [_derive_case_id(item) for item in self.items]
        self.transform = _base_transforms(cfg, mode="train")
        self.roi_size = cfg.roi_size
        depth = slices_per_volume or self.roi_size[2]
        self.slice_mappings: List[Tuple[int, int]] = []
        for case_idx in range(len(self.items)):
            for slice_idx in range(depth):
                self.slice_mappings.append((case_idx, slice_idx))

    def __len__(self) -> int:
        return len(self.slice_mappings)

    def __getitem__(self, index: int) -> Dict:
        vol_idx, slice_idx = self.slice_mappings[index]
        item = dict(self.items[vol_idx])
        data = self.transform(item)
        image = data["image"].float()
        label = data["label"].long()
        if image.ndim != 4:
            raise RuntimeError(f"Expected 4D tensor (C, X, Y, Z), got {image.shape}")
        depth = image.shape[3]
        slice_idx = int(np.clip(slice_idx, 0, depth - 1))
        slice_image = image[:, :, :, slice_idx]
        slice_label = label[:, :, :, slice_idx].squeeze(0)
        return {
            "image": slice_image,
            "label": slice_label,
            "idx": f"{self.case_ids[vol_idx]}_z{slice_idx}",
        }


def create_target_datasets(split_json: str,
                           *,
                           roi_size: Tuple[int, int, int],
                           target_spacing: Tuple[float, float, float],
                           apply_spacing: bool,
                           apply_orientation: bool,
                           foreground_only: bool,
                           laterality_pairs_json: Optional[str],
                           slices_per_volume: Optional[int] = None,
                           val_subset: str = "validation"):
    train_cfg = DatasetConfig(
        split_json=split_json,
        subset="training",
        roi_size=roi_size,
        target_spacing=target_spacing,
        apply_spacing=apply_spacing,
        apply_orientation=apply_orientation,
        foreground_only=foreground_only,
        laterality_pairs_json=laterality_pairs_json,
    )
    val_cfg = DatasetConfig(
        split_json=split_json,
        subset=val_subset,
        roi_size=roi_size,
        target_spacing=target_spacing,
        apply_spacing=apply_spacing,
        apply_orientation=apply_orientation,
        foreground_only=foreground_only,
        laterality_pairs_json=laterality_pairs_json,
    )
    train_dataset = PPREMOSliceDataset(train_cfg, slices_per_volume=slices_per_volume)
    val_dataset = PPREMOVolumeDataset(val_cfg, mode="val")
    return train_dataset, val_dataset
