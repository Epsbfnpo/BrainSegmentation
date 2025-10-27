"""Utility to generate age-conditioned shape templates from dataset splits."""

import argparse
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, zoom

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore

try:  # pragma: no cover - optional GPU acceleration
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cupy_distance_transform_edt
except ImportError:  # pragma: no cover - cupy is optional
    cp = None  # type: ignore
    cupy_distance_transform_edt = None  # type: ignore

LABEL_KEYS = (
    "label",
    "label_path",
    "seg",
    "segmentation",
    "label_relpath",
)

ROOT_KEYS = (
    "dataset_root",
    "root",
    "root_dir",
    "label_root",
)

AGE_KEYS = (
    "age",
    "age_weeks",
    "scan_age",
    "scan_age_weeks",
    "pma",
    "PMA",
    "postmenstrual_age",
    "post_menstrual_age",
    "postnatal_age",
    "postnatal_age_weeks",
    "postnatal_age_days",
    "gestational_age",
    "gestational_age_weeks",
)


def _coerce_age(value) -> Optional[float]:
    """Best-effort conversion of mixed-format age metadata to weeks."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and value:
        return _coerce_age(value[0])
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        match = re.search(r"(\d+)\s*\+\s*(\d+)", text)
        if match:
            weeks = float(match.group(1))
            days = float(match.group(2))
            return weeks + days / 7.0
        cleaned = re.sub(r"[^0-9.+-]", "", text)
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _extract_age(item: Dict) -> Optional[float]:
    """Extract scan age (weeks) from common fields in the sample dict."""

    for key in AGE_KEYS:
        if key in item:
            age = _coerce_age(item[key])
            if age is not None:
                return age

    ga = _coerce_age(item.get("ga")) or _coerce_age(item.get("GA")) or _coerce_age(item.get("gestational_age"))
    pna = _coerce_age(item.get("pna")) or _coerce_age(item.get("PNA")) or _coerce_age(item.get("postnatal_age"))
    if ga is not None and pna is not None:
        return ga + pna

    for key in ("metadata", "meta", "attributes", "info"):
        meta = item.get(key)
        if not isinstance(meta, dict):
            continue
        age = _extract_age(meta)
        if age is not None:
            return age

    return None


def _load_split(json_path: str) -> Tuple[List[Dict], Optional[str]]:
    """Load a split JSON and return (samples, default_root)."""

    with open(json_path, "r") as f:
        data = json.load(f)

    default_root: Optional[str] = None
    for key in ROOT_KEYS:
        root_val = data.get(key)
        if isinstance(root_val, str) and root_val:
            default_root = root_val
            break

    samples: List[Dict] = []

    def _extend(value):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    samples.append(item)

    for key in ("training", "validation", "testing", "test", "train", "val"):
        _extend(data.get(key, []))

    for maybe_splits in (data.get("splits"), data.get("data")):
        if isinstance(maybe_splits, dict):
            for value in maybe_splits.values():
                _extend(value)

    if isinstance(data, list):
        _extend(data)

    return samples, default_root


def _resolve_label_path(sample: Dict,
                        json_path: str,
                        default_root: Optional[str],
                        cli_root: Optional[str]) -> Optional[str]:
    """Resolve the absolute label path using multiple fallbacks."""

    candidate = None
    for key in LABEL_KEYS:
        value = sample.get(key)
        if value:
            candidate = value
            break
    if candidate is None:
        return None

    candidate_path = Path(candidate)
    if candidate_path.is_absolute() and candidate_path.exists():
        return str(candidate_path)

    roots: List[Optional[str]] = []
    for key in ROOT_KEYS:
        value = sample.get(key)
        if isinstance(value, str):
            roots.append(value)
    roots.extend([default_root, cli_root, os.environ.get("DATA_ROOT")])
    roots.append(str(Path(json_path).parent))

    for root in roots:
        if not root:
            continue
        resolved = Path(root) / candidate
        if resolved.exists():
            return str(resolved)

    if roots:
        fallback_root = roots[0] or str(Path(json_path).parent)
        return str(Path(fallback_root) / candidate)
    return str(candidate_path)


def _iter_samples(split_paths: Iterable[str],
                  data_root: Optional[str]) -> Iterator[Tuple[str, Optional[float]]]:
    for path in split_paths:
        samples, default_root = _load_split(path)
        for sample in samples:
            label_path = _resolve_label_path(sample, path, default_root, data_root)
            if not label_path:
                continue
            yield label_path, _extract_age(sample)


def _compute_signed_distance_cpu(mask: np.ndarray,
                                 spacing: Tuple[float, float, float]) -> np.ndarray:
    inside = distance_transform_edt(mask, sampling=spacing)
    outside = distance_transform_edt(~mask, sampling=spacing)
    sdf = inside - outside
    return sdf.astype(np.float32, copy=False)


def _compute_signed_distance_gpu(cp_label,
                                 spacing: Tuple[float, float, float],
                                 cls: int) -> Optional[np.ndarray]:
    if cp is None or cupy_distance_transform_edt is None:  # pragma: no cover - guarded at runtime
        raise RuntimeError("CuPy is not available but CUDA device was requested.")

    mask = cp.equal(cp_label, cls)
    if not bool(cp.any(mask)):
        return None

    inside = cupy_distance_transform_edt(mask, sampling=spacing)
    outside = cupy_distance_transform_edt(cp.logical_not(mask), sampling=spacing)
    sdf = (inside - outside).astype(cp.float32)
    result = cp.asnumpy(sdf)
    cp.get_default_memory_pool().free_all_blocks()
    return result


def _ensure_bucket(stats: Dict[int, Dict[str, np.ndarray]],
                   age_bin: int,
                   num_classes: int,
                   volume_shape: Tuple[int, ...],
                   dtype: np.dtype = np.float32) -> Dict[str, np.ndarray]:
    bucket = stats.get(age_bin)
    if bucket is None:
        bucket = {
            "count": 0,
            "sum": np.zeros((num_classes, *volume_shape), dtype=dtype),
            "sum_sq": np.zeros((num_classes, *volume_shape), dtype=dtype),
        }
        stats[age_bin] = bucket
    return bucket


def _accumulate_sdf(bucket: Dict[str, np.ndarray], cls_index: int, sdf: np.ndarray) -> None:
    bucket_sum = bucket["sum"][cls_index]
    bucket_sq = bucket["sum_sq"][cls_index]
    bucket_sum += sdf
    bucket_sq += sdf * sdf


def _resample_label_to_shape(label: np.ndarray,
                             target_shape: Tuple[int, int, int]) -> np.ndarray:
    if label.shape == target_shape:
        return label

    zoom_factors = tuple(t / float(s) for t, s in zip(target_shape, label.shape))
    resampled = zoom(
        label.astype(np.float32, copy=False),
        zoom=zoom_factors,
        order=0,
        mode="nearest",
    )
    return resampled.astype(np.int32, copy=False)


def _adjust_spacing(spacing: Tuple[float, float, float],
                    original_shape: Tuple[int, int, int],
                    target_shape: Tuple[int, int, int]) -> Tuple[float, float, float]:
    if original_shape == target_shape:
        return spacing
    physical = np.array(spacing) * np.array(original_shape)
    adjusted = physical / np.array(target_shape)
    return tuple(float(v) for v in adjusted)


def _parse_shape_arg(shape: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not shape:
        return None
    parts = [p.strip() for p in re.split(r"[xX,]", shape) if p.strip()]
    if len(parts) != 3:
        raise ValueError(
            "--target-shape must specify exactly three integers separated by 'x' or ','"
        )
    return tuple(int(p) for p in parts)


def build_shape_templates(split_paths: Iterable[str],
                          output_path: str,
                          num_classes: int,
                          age_bin_width: float,
                          data_root: Optional[str] = None,
                          progress: bool = True,
                          workers: int = 1,
                          device: str = "cpu",
                          target_shape: Optional[Tuple[int, int, int]] = None) -> Dict[str, torch.Tensor]:
    stats: Dict[int, Dict[str, np.ndarray]] = {}
    reference_shape: Optional[Tuple[int, int, int]] = tuple(target_shape) if target_shape else None

    samples = list(_iter_samples(split_paths, data_root=data_root))
    if not samples:
        raise RuntimeError("No samples found in provided split files.")

    iterator = samples
    if progress and tqdm is not None:
        iterator = tqdm(samples, desc="Accumulating SDTs", unit="label")

    device = device.lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'")
    if device == "cuda" and (cp is None or cupy_distance_transform_edt is None):
        raise RuntimeError("CuPy is required for CUDA acceleration but is not installed.")

    workers = max(1, int(workers)) if device == "cpu" else 1

    for label_path, age in iterator:
        if not os.path.exists(label_path):
            if tqdm is not None:
                tqdm.write(f"⚠️  Skipping missing label: {label_path}")
            else:
                print(f"⚠️  Skipping missing label: {label_path}")
            continue

        label_img = nib.load(label_path)
        label = label_img.get_fdata().astype(np.int32)
        spacing = tuple(float(v) for v in label_img.header.get_zooms()[:3])
        original_shape = tuple(int(x) for x in label.shape)

        if reference_shape is None:
            reference_shape = original_shape
        if original_shape != reference_shape:
            label = _resample_label_to_shape(label, reference_shape)
            spacing = _adjust_spacing(spacing, original_shape, reference_shape)

        age_bin = -1 if age is None else int(math.floor(age / age_bin_width) * age_bin_width)
        bucket = _ensure_bucket(stats, age_bin, num_classes, reference_shape)
        bucket["count"] += 1
        present = np.unique(label)
        present = present[(present >= 1) & (present <= num_classes)]
        if not isinstance(present, np.ndarray):
            present = np.asarray(present)

        if device == "cuda":
            cp_label = cp.asarray(label)
            for cls in present:
                sdf = _compute_signed_distance_gpu(cp_label, spacing, int(cls))
                if sdf is None:
                    continue
                _accumulate_sdf(bucket, int(cls) - 1, sdf)
            cp.get_default_memory_pool().free_all_blocks()
        else:
            def _cpu_job(cls_id: int) -> Tuple[int, Optional[np.ndarray]]:
                cls_mask = label == cls_id
                if not np.any(cls_mask):
                    return cls_id, None
                sdf_local = _compute_signed_distance_cpu(cls_mask, spacing)
                return cls_id, sdf_local

            if workers > 1 and present.size > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for cls_id, sdf in pool.map(_cpu_job, [int(c) for c in present]):
                        if sdf is None:
                            continue
                        _accumulate_sdf(bucket, cls_id - 1, sdf)
            else:
                for cls in present:
                    cls_mask = label == int(cls)
                    if not np.any(cls_mask):
                        continue
                    sdf = _compute_signed_distance_cpu(cls_mask, spacing)
                    _accumulate_sdf(bucket, int(cls) - 1, sdf)

    if not stats:
        raise RuntimeError("No valid labels were processed; cannot build templates.")

    templates: Dict[str, torch.Tensor] = {}
    stddevs: Dict[str, torch.Tensor] = {}

    for age_bin, bucket in stats.items():
        count = float(bucket["count"])
        sum_sdf = bucket["sum"]
        sum_sq = bucket["sum_sq"]
        mean = sum_sdf / count
        var = np.maximum(sum_sq / count - mean ** 2, 0.0)
        std = np.sqrt(var, dtype=np.float32).astype(np.float32, copy=False)
        key = "unknown_age" if age_bin < 0 else f"age_{age_bin:02d}w"
        templates[key] = torch.from_numpy(mean.astype(np.float32, copy=False))
        stddevs[key] = torch.from_numpy(std)

    payload = {"mean": templates, "std": stddevs, "num_classes": num_classes}
    torch.save(payload, output_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build age-aware shape templates from split JSONs")
    parser.add_argument(
        "--split", dest="splits", action="append", required=True,
        help="Path to a split JSON file. Can be specified multiple times.",
    )
    parser.add_argument(
        "--num-classes", type=int, default=87,
        help="Number of foreground classes (default: 87).",
    )
    parser.add_argument(
        "--age-bin-width", type=float, default=2.0,
        help="Age bin width in weeks (default: 2.0).",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Optional root directory to prepend to relative label paths.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Destination .pt file to store the templates.",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable tqdm progress bar output.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel CPU workers per volume (only for CPU backend).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Execution device for distance transforms (default: cpu).",
    )
    parser.add_argument(
        "--target-shape", type=str, default=None,
        help="Optional 'XxYxZ' shape to which all labels will be resampled before computing SDTs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_shape = _parse_shape_arg(args.target_shape)
    payload = build_shape_templates(
        split_paths=args.splits,
        output_path=args.output,
        num_classes=args.num_classes,
        age_bin_width=args.age_bin_width,
        data_root=args.data_root,
        progress=not args.no_progress,
        workers=args.workers,
        device=args.device,
        target_shape=target_shape,
    )
    print(f"✅ Saved shape templates for {len(payload['mean'])} age buckets to {args.output}")


if __name__ == "__main__":
    main()
