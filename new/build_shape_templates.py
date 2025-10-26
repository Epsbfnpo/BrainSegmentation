"""Utility to generate age-conditioned shape templates from dataset splits."""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore

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


def _compute_signed_distance(mask: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
    inside = distance_transform_edt(mask, sampling=spacing)
    outside = distance_transform_edt(~mask, sampling=spacing)
    return inside - outside


def _ensure_bucket(stats: Dict[int, Dict[str, np.ndarray]],
                   age_bin: int,
                   num_classes: int,
                   volume_shape: Tuple[int, ...],
                   dtype: np.dtype = np.float32) -> Dict[str, np.ndarray]:
    bucket = stats.get(age_bin)
    if bucket is None:
        bucket = {
            "count": 0,
            "mean": np.zeros((num_classes, *volume_shape), dtype=dtype),
            "m2": np.zeros((num_classes, *volume_shape), dtype=np.float32),
        }
        stats[age_bin] = bucket
    return bucket


def _update_bucket(bucket: Dict[str, np.ndarray],
                   cls_index: int,
                   sdf: np.ndarray,
                   count: int) -> None:
    mean = bucket["mean"][cls_index]
    m2 = bucket["m2"][cls_index]

    delta = sdf - mean
    mean += delta / count
    m2 += delta * (sdf - mean)


def build_shape_templates(split_paths: Iterable[str],
                          output_path: str,
                          num_classes: int,
                          age_bin_width: float,
                          data_root: Optional[str] = None,
                          progress: bool = True) -> Dict[str, torch.Tensor]:
    stats: Dict[int, Dict[str, np.ndarray]] = {}
    reference_shape: Optional[Tuple[int, ...]] = None

    samples = list(_iter_samples(split_paths, data_root=data_root))
    if not samples:
        raise RuntimeError("No samples found in provided split files.")

    iterator = samples
    if progress and tqdm is not None:
        iterator = tqdm(samples, desc="Accumulating SDTs", unit="label")

    zero_sdf_cache: Dict[Tuple[int, ...], np.ndarray] = {}

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

        if reference_shape is None:
            reference_shape = label.shape
        elif label.shape != reference_shape:
            raise ValueError(
                f"All labels must share the same shape. Got {label.shape} vs {reference_shape} from {label_path}."
            )

        age_bin = -1 if age is None else int(math.floor(age / age_bin_width) * age_bin_width)
        bucket = _ensure_bucket(stats, age_bin, num_classes, label.shape)
        bucket["count"] += 1
        count = bucket["count"]

        zero_key = label.shape
        zero_sdf = zero_sdf_cache.get(zero_key)
        if zero_sdf is None:
            zero_sdf = np.zeros(label.shape, dtype=np.float32)
            zero_sdf_cache[zero_key] = zero_sdf

        for cls in range(1, num_classes + 1):
            mask = label == cls
            if np.any(mask):
                sdf = _compute_signed_distance(mask, spacing).astype(np.float32, copy=False)
            else:
                sdf = zero_sdf
            _update_bucket(bucket, cls - 1, sdf, count)

    if not stats:
        raise RuntimeError("No valid labels were processed; cannot build templates.")

    templates: Dict[str, torch.Tensor] = {}
    stddevs: Dict[str, torch.Tensor] = {}

    for age_bin, bucket in stats.items():
        mean = bucket["mean"]
        count = float(bucket["count"])
        m2 = bucket["m2"]
        denom = max(count - 1.0, 1.0)
        std = np.sqrt(np.maximum(m2 / denom, 0.0)).astype(np.float32)
        key = "unknown_age" if age_bin < 0 else f"age_{age_bin:02d}w"
        templates[key] = torch.from_numpy(mean)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_shape_templates(
        split_paths=args.splits,
        output_path=args.output,
        num_classes=args.num_classes,
        age_bin_width=args.age_bin_width,
        data_root=args.data_root,
        progress=not args.no_progress,
    )
    print(f"✅ Saved shape templates for {len(payload['mean'])} age buckets to {args.output}")


if __name__ == "__main__":
    main()
