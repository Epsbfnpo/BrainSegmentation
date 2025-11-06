"""Custom MONAI transforms for texture-focused training."""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_tensor import MetaTensor
from monai.transforms import MapTransform, Transform

__all__ = [
    "TextureStatsd",
    "RemapLabelsd",
    "AddDomainLabeld",
    "stack_texture_features",
    "RandomHistogramShiftd",
]


class TextureStatsd(MapTransform):
    """Compute handcrafted texture statistics for the given image keys."""

    def __init__(
        self,
        keys: Iterable[str],
        *,
        prefix: str = "texture_stats",
        mask_key: str | None = "label",
        channel_wise: bool = False,
        use_log: bool = True,
    ) -> None:
        super().__init__(keys)
        self.prefix = prefix
        self.mask_key = mask_key
        self.channel_wise = channel_wise
        self.use_log = use_log

    def _prepare_array(self, image: NdarrayOrTensor) -> np.ndarray:
        if isinstance(image, MetaTensor):
            array = image.array
        elif torch.is_tensor(image):
            array = image.detach().cpu().numpy()
        else:
            array = np.asarray(image)
        array = np.nan_to_num(array.astype(np.float32), copy=False)
        return array

    def _compute_features(self, array: np.ndarray, *, mask: np.ndarray | None) -> List[float]:
        if array.ndim == 4:  # (C, H, W, D)
            if self.channel_wise:
                features: List[float] = []
                for c in range(array.shape[0]):
                    features.extend(self._compute_features(array[c], mask=mask))
                return features
            array = array[0]
        elif array.ndim == 3:
            array = array
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {array.shape}")

        working = array.copy()
        if self.use_log:
            working = np.where(working > 0, np.log1p(working), working)

        if mask is not None:
            mask = mask.astype(bool)
            if mask.shape != working.shape:
                mask = mask.squeeze()
            region = working[mask]
            if region.size == 0:
                region = working.ravel()
        else:
            region = working.ravel()

        region = region.astype(np.float32)
        if region.size == 0:
            region = np.zeros(1, dtype=np.float32)

        features: List[float] = []

        # Intensity distribution statistics
        features.append(float(region.mean()))
        features.append(float(region.std()))
        features.append(float(region.min()))
        features.append(float(region.max()))
        percentiles = [5, 25, 50, 75, 95]
        features.extend(float(np.percentile(region, p)) for p in percentiles)
        features.append(float(np.median(region)))
        features.append(float(np.percentile(region, 99) - np.percentile(region, 1)))  # contrast range

        # Gradient-based descriptors
        gradients = np.gradient(working)
        grad_mag = np.sqrt(sum(g ** 2 for g in gradients))
        if mask is not None:
            grad_region = grad_mag[mask]
        else:
            grad_region = grad_mag.ravel()
        features.append(float(grad_region.mean()))
        features.append(float(grad_region.std()))

        # Laplacian approximation via divergence of gradients
        laplacian = sum(np.gradient(g)[i] for i, g in enumerate(gradients))
        if mask is not None:
            lap_region = laplacian[mask]
        else:
            lap_region = laplacian.ravel()
        features.append(float(np.mean(lap_region ** 2)))  # laplacian energy

        # Frequency energy (low vs high)
        spectrum = np.abs(np.fft.fftn(working))
        spectrum = np.fft.fftshift(spectrum)
        center = tuple(s // 2 for s in spectrum.shape)
        radius = min(center)
        if radius > 0:
            grid = np.stack(
                np.meshgrid(
                    *[np.arange(s) - c for s, c in zip(spectrum.shape, center)], indexing="ij"
                )
            )
            dist = np.sqrt(np.sum(grid ** 2, axis=0))
            low_mask = dist <= radius * 0.25
            high_mask = dist >= radius * 0.6
            low_energy = float(np.mean(spectrum[low_mask]))
            high_energy = float(np.mean(spectrum[high_mask]))
        else:
            low_energy = float(np.mean(spectrum))
            high_energy = float(np.mean(spectrum))
        features.extend([low_energy, high_energy])
        features.append(float(high_energy / (low_energy + 1e-6)))

        # Local variance at multiple scales via down-sampling
        scales = [1, 2, 4]
        for scale in scales:
            if scale == 1:
                scaled = working
            else:
                scaled = working[::scale, ::scale, ::scale]
            features.append(float(np.var(scaled)))

        return features

    def __call__(self, data):
        d = dict(data)
        mask = None
        if self.mask_key and self.mask_key in d:
            mask_array = self._prepare_array(d[self.mask_key])
            if mask_array.ndim == 4:
                mask_array = mask_array[0]
            if mask_array.min() < 0:
                mask = mask_array >= 0
            else:
                mask = mask_array > 0

        feature_vectors: List[float] = []
        for key in self.key_iterator(d):
            array = self._prepare_array(d[key])
            feature_vectors.extend(self._compute_features(array, mask=mask))

        d[self.prefix] = torch.tensor(feature_vectors, dtype=torch.float32)
        d[f"{self.prefix}_dim"] = torch.tensor([len(feature_vectors)], dtype=torch.int64)
        return d


class RemapLabelsd(MapTransform):
    """Remap labels to foreground-only indices with background = -1."""

    def __init__(
        self,
        keys,
        num_classes: int,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = int(num_classes)
        self._has_warned = False

    def _remap(self, label: np.ndarray) -> np.ndarray:
        data = label.astype(np.int32, copy=False)
        foreground_mask = data > 0
        remapped = data - 1
        remapped[~foreground_mask] = -1
        remapped = np.clip(remapped, -1, self.num_classes - 1)
        return remapped.astype(np.int32, copy=False)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            meta = None
            if isinstance(label, MetaTensor):
                meta = label.meta
                array = label.array
            else:
                array = np.asarray(label)
            remapped = self._remap(array)
            if not self._has_warned:
                unique_vals = np.unique(remapped)
                print(
                    "[RemapLabelsd] foreground-only remapping active: "
                    f"min={unique_vals.min()}, max={unique_vals.max()}, "
                    f"classes={len(unique_vals)}"
                )
                self._has_warned = True
            if meta is not None:
                d[key] = MetaTensor(remapped, meta=meta)
            else:
                d[key] = remapped
        return d


class AddDomainLabeld(Transform):
    """Attach a domain index to the sample."""

    def __init__(self, domain_index: int) -> None:
        self.domain_index = int(domain_index)

    def __call__(self, data):
        d = dict(data)
        d["domain"] = torch.tensor(self.domain_index, dtype=torch.long)
        return d


def stack_texture_features(batch: Sequence[dict], key: str = "texture_stats") -> torch.Tensor:
    """Utility to stack variable-length feature vectors in a batch."""

    features = [sample[key] for sample in batch]
    if not features:
        raise ValueError("Empty batch encountered when stacking texture features")
    return torch.stack(features, dim=0)


class RandomHistogramShiftd(MapTransform):
    """Histogram shift augmentation that is compatible across MONAI versions."""

    def __init__(
        self,
        keys: Iterable[str],
        prob: float = 0.1,
        num_control_points: int = 4,
        shift_range: Tuple[float, float] = (-0.05, 0.05),
        channel_wise: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if num_control_points < 2:
            raise ValueError("num_control_points must be >= 2")
        self.prob = float(prob)
        self.num_control_points = int(num_control_points)
        self.shift_range = (float(shift_range[0]), float(shift_range[1]))
        self.channel_wise = channel_wise

    def _prepare_array(self, image: NdarrayOrTensor) -> Tuple[np.ndarray, str]:
        if isinstance(image, MetaTensor):
            return np.asarray(image.array), "meta"
        if torch.is_tensor(image):
            return image.detach().cpu().numpy(), "tensor"
        return np.asarray(image), "array"

    def _to_original_type(self, array: np.ndarray, original, kind: str):
        if kind == "meta":
            orig_array = np.asarray(original.array)
            cast_array = array.astype(orig_array.dtype, copy=False)
            meta = dict(getattr(original, "meta", {}))
            output = MetaTensor(cast_array, meta=meta)
            if hasattr(original, "affine"):
                output.affine = original.affine
            if hasattr(original, "applied_operations"):
                output.applied_operations = list(original.applied_operations)
            return output
        if kind == "tensor":
            return torch.as_tensor(array, dtype=original.dtype)
        return array.astype(np.float32, copy=False)

    def _generate_mapping(self) -> Tuple[np.ndarray, np.ndarray]:
        ctrl_x = np.linspace(0.0, 1.0, self.num_control_points, dtype=np.float32)
        offsets = np.random.uniform(self.shift_range[0], self.shift_range[1], size=ctrl_x.shape).astype(np.float32)
        offsets[0] = 0.0
        offsets[-1] = 0.0
        ctrl_y = np.clip(ctrl_x + offsets, 0.0, 1.0)
        return ctrl_x, ctrl_y

    def _apply_shift(self, array: np.ndarray) -> np.ndarray:
        original_shape = array.shape
        if array.ndim == 4 and self.channel_wise:
            shifted_channels: List[np.ndarray] = []
            for channel in array:
                shifted_channels.append(self._apply_shift(channel))
            return np.stack(shifted_channels, axis=0)

        flat = array.astype(np.float32, copy=False)
        arr_min = flat.min()
        arr_max = flat.max()
        if not np.isfinite(arr_min) or not np.isfinite(arr_max) or arr_max <= arr_min + 1e-6:
            return flat.reshape(original_shape)

        ctrl_x, ctrl_y = self._generate_mapping()
        scaled = (flat - arr_min) / (arr_max - arr_min)
        shifted = np.interp(scaled, ctrl_x, ctrl_y)
        shifted = shifted * (arr_max - arr_min) + arr_min
        return shifted.reshape(original_shape)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if random.random() >= self.prob:
                continue
            original = d[key]
            array, kind = self._prepare_array(original)
            shifted = self._apply_shift(array)
            d[key] = self._to_original_type(shifted, original, kind)
        return d
