#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target-only prior generator for volume / shape / structural priors.

This script builds age-conditioned priors using only the target dataset.
It produces the following files in <out_root>/target/:

  - class_map.json
  - volume_stats.json
  - sdf_templates.npz
  - adjacency_prior.npz
  - R_mask.npy
  - R_meta.json

The implementation follows the specification outlined in the project
instructions.  All statistics are computed using labels that have been
remapped to foreground-only indices (original labels 1..87 -> 0..86).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from scipy.ndimage import zoom
from tqdm import tqdm

NUM_CLASSES = 87  # foreground classes only (0..86)


# -----------------------------------------------------------------------------
# utility helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_shape(text: str) -> Sequence[int]:
    match = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)\s*$", text)
    if not match:
        raise ValueError(f"Invalid shape specification: {text}")
    return tuple(int(match.group(i)) for i in (1, 2, 3))


def age_to_bin(age: float, width: float) -> int:
    return int(round(age / width))


def load_split_target(json_path: str) -> List[Dict[str, object]]:
    with open(json_path, "r") as f:
        data = json.load(f)

    samples: List[Dict[str, object]] = []
    for key in ("training", "train", "validation", "val", "test", "testing"):
        for item in data.get(key, []):
            if not isinstance(item, dict):
                continue
            label_path = item.get("label")
            metadata = item.get("metadata", {}) or {}
            age = metadata.get("scan_age")
            if label_path is None or age is None:
                continue
            samples.append({
                "label": label_path,
                "age": float(age),
            })

    if not samples:
        raise RuntimeError(
            "Split JSON does not contain any entries with label paths and metadata.scan_age"
        )
    return samples


def remap_foreground_only(label: np.ndarray) -> np.ndarray:
    arr = label.astype(np.int32, copy=False)
    remapped = np.where(arr > 0, arr - 1, -1).astype(np.int16)
    return remapped


def voxel_fraction_per_class(lbl_fg: np.ndarray) -> np.ndarray:
    mask = lbl_fg >= 0
    total = int(mask.sum())
    fractions = np.zeros((NUM_CLASSES,), dtype=np.float64)
    if total == 0:
        return fractions
    values = lbl_fg[mask]
    counts = np.bincount(values, minlength=NUM_CLASSES)
    fractions = counts.astype(np.float64) / float(total)
    return fractions


def adjacency_counts(lbl_fg: np.ndarray) -> np.ndarray:
    counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    mask = lbl_fg >= 0
    volume = lbl_fg
    for axis in (0, 1, 2):
        for shift in (-1, 1):
            rolled = np.roll(volume, shift=shift, axis=axis)
            valid = mask & (rolled >= 0) & (volume != rolled)
            if not np.any(valid):
                continue
            src = volume[valid]
            dst = rolled[valid]
            for i, j in zip(src, dst):
                counts[i, j] += 1
                counts[j, i] += 1
    np.fill_diagonal(counts, 0)
    return counts


def resample_label_nn(label: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    factors = [t / s for t, s in zip(target_shape, label.shape)]
    if any(f <= 0 for f in factors):
        raise ValueError(f"Invalid resample factors from {label.shape} to {target_shape}")
    out = zoom(label, zoom=factors, order=0)
    return out.astype(np.int16)


def compute_sdf(label: np.ndarray, cls: int, band: float) -> Optional[np.ndarray]:
    mask = label == cls
    if not mask.any():
        return None
    d_in = edt(mask)
    d_out = edt(~mask)
    sdf = d_out - d_in
    if band > 0:
        sdf = np.clip(sdf, -band, band)
    return sdf.astype(np.float32)


# -----------------------------------------------------------------------------
# worker functions
# -----------------------------------------------------------------------------

def worker_basic(sample: Dict[str, object], age_bin: int) -> Dict[str, object]:
    img = nib.load(sample["label"])
    label = np.asarray(img.dataobj, dtype=np.int16)
    lbl_fg = remap_foreground_only(label)
    fractions = voxel_fraction_per_class(lbl_fg)
    adj = adjacency_counts(lbl_fg)
    return {"age_bin": age_bin, "fractions": fractions, "adj": adj}


def worker_sdf(chunk: Sequence[Dict[str, object]], temp_dir: str, template_shape: Sequence[int],
               band: float, bin_width: float) -> List[str]:
    sums: Dict[int, np.ndarray] = {}
    sqs: Dict[int, np.ndarray] = {}
    counts: Dict[int, np.ndarray] = {}

    for sample in chunk:
        age_bin = age_to_bin(sample["age"], bin_width)
        img = nib.load(sample["label"])
        label = np.asarray(img.dataobj, dtype=np.int16)
        lbl_fg = remap_foreground_only(label)
        lbl_ds = resample_label_nn(lbl_fg, template_shape)
        present = np.unique(lbl_ds)
        present = present[(present >= 0) & (present < NUM_CLASSES)]
        if age_bin not in sums:
            sums[age_bin] = np.zeros((NUM_CLASSES, *template_shape), dtype=np.float32)
            sqs[age_bin] = np.zeros((NUM_CLASSES, *template_shape), dtype=np.float32)
            counts[age_bin] = np.zeros((NUM_CLASSES,), dtype=np.int32)
        for cls in present:
            sdf = compute_sdf(lbl_ds, int(cls), band)
            if sdf is None:
                continue
            sums[age_bin][cls] += sdf
            sqs[age_bin][cls] += sdf * sdf
            counts[age_bin][cls] += 1

    written: List[str] = []
    for age_bin, arr in sums.items():
        shard_path = Path(temp_dir) / f"sdf_shard_ab{age_bin}_{os.getpid()}.npz"
        np.savez_compressed(
            shard_path,
            age_bin=np.int32(age_bin),
            sum=arr,
            sumsq=sqs[age_bin],
            cnt=counts[age_bin],
        )
        written.append(str(shard_path))
    return written


# -----------------------------------------------------------------------------
# main generation logic
# -----------------------------------------------------------------------------

def generate_priors(split_json: str,
                    out_dir: str,
                    workers: int,
                    age_bin_width: float,
                    template_shape: Sequence[int],
                    sdf_band: float,
                    include_sdf: bool,
                    save_sdf_std: bool,
                    adjacency_norm: str,
                    rmask_policy: str,
                    rmask_arg: float,
                    sdf_chunk_size: int) -> None:
    out_path = Path(out_dir)
    ensure_dir(out_path)

    class_map = {
        "index_to_raw_label": {str(i): int(i + 1) for i in range(NUM_CLASSES)},
        "raw_label_to_index": {str(i + 1): int(i) for i in range(NUM_CLASSES)},
    }
    with open(out_path / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    samples = load_split_target(split_json)
    for sample in samples:
        sample["age_bin"] = age_to_bin(sample["age"], age_bin_width)

    vol_sum: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros((NUM_CLASSES,), dtype=np.float64))
    vol_sqs: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros((NUM_CLASSES,), dtype=np.float64))
    vol_cnt: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros((NUM_CLASSES,), dtype=np.int64))

    adj_sum: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64))
    adj_cnt: Dict[int, int] = defaultdict(int)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_basic, sample, sample["age_bin"]) for sample in samples]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Volume+Adjacency"):
            result = future.result()
            age_bin = result["age_bin"]
            frac = result["fractions"].astype(np.float64)
            adj = result["adj"].astype(np.float64)

            vol_sum[age_bin] += frac
            vol_sqs[age_bin] += frac * frac
            vol_cnt[age_bin] += (frac >= 0).astype(np.int64)

            adj_sum[age_bin] += adj
            adj_cnt[age_bin] += 1

    volume_payload: Dict[str, Dict[str, object]] = {}
    for age_bin, counts in vol_cnt.items():
        n = np.maximum(counts, 1)
        mean = vol_sum[age_bin] / n
        var = np.maximum(vol_sqs[age_bin] / n - mean ** 2, 1e-12)
        std = np.sqrt(var)
        volume_payload[str(age_bin)] = {
            "means": mean.tolist(),
            "stds": std.tolist(),
            "n": counts.tolist(),
            "age_bin_width": age_bin_width,
        }

    with open(out_path / "volume_stats.json", "w") as f:
        json.dump(volume_payload, f, indent=2)

    sorted_bins = sorted(adj_cnt.keys())
    adjacency_list: List[np.ndarray] = []
    freq_list: List[np.ndarray] = []
    for age_bin in sorted_bins:
        adj = adj_sum[age_bin]
        if adj.max() > 0:
            freq = adj / (adj.max() + 1e-12)
        else:
            freq = adj
        if adjacency_norm == "row":
            row_sum = freq.sum(axis=1, keepdims=True) + 1e-12
            adj_norm = freq / row_sum
        elif adjacency_norm == "sym":
            d = np.sqrt(freq.sum(axis=1) + 1e-12)
            adj_norm = (freq / d[:, None]) / d[None, :]
            adj_norm[np.isnan(adj_norm)] = 0.0
        else:
            raise ValueError("adjacency_norm must be 'row' or 'sym'")
        np.fill_diagonal(adj_norm, 0.0)
        adjacency_list.append(adj_norm.astype(np.float32))
        freq_list.append(freq.astype(np.float32))

    A_prior = np.stack(adjacency_list, axis=0) if adjacency_list else np.zeros((0, NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    F_freq = np.stack(freq_list, axis=0) if freq_list else np.zeros_like(A_prior)
    np.savez_compressed(
        out_path / "adjacency_prior.npz",
        ages=np.array(sorted_bins, dtype=np.int32),
        A_prior=A_prior,
        freq=F_freq,
        counts=np.array([adj_cnt[b] for b in sorted_bins], dtype=np.int32),
        meta={"norm": adjacency_norm, "bin_width": age_bin_width},
    )

    if F_freq.size > 0:
        aggregated = F_freq.max(axis=0)
        mask = np.zeros_like(aggregated, dtype=np.float32)
        if rmask_policy == "row-topk":
            frac = min(max(float(rmask_arg), 1e-6), 1.0)
            k = max(1, int(round(frac * NUM_CLASSES)))
            for i in range(NUM_CLASSES):
                idx = np.argsort(aggregated[i])[::-1]
                keep = idx[:k]
                mask[i, keep] = 1.0
        elif rmask_policy == "global-quant":
            q = min(max(float(rmask_arg), 0.0), 1.0)
            threshold = np.quantile(aggregated, q)
            mask[aggregated >= threshold] = 1.0
        else:
            raise ValueError("rmask_policy must be 'row-topk' or 'global-quant'")
        np.fill_diagonal(mask, 0.0)
        np.save(out_path / "R_mask.npy", mask.astype(np.float32))
        with open(out_path / "R_meta.json", "w") as f:
            json.dump({"policy": rmask_policy, "arg": rmask_arg}, f, indent=2)

    if include_sdf:
        tmp_dir = Path(tempfile.mkdtemp(prefix="sdf_shards_"))
        try:
            chunks: List[List[Dict[str, object]]] = []
            current: List[Dict[str, object]] = []
            for sample in samples:
                current.append(sample)
                if len(current) >= sdf_chunk_size:
                    chunks.append(current)
                    current = []
            if current:
                chunks.append(current)

            shard_paths: List[str] = []
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(worker_sdf, chunk, str(tmp_dir), template_shape, sdf_band, age_bin_width)
                    for chunk in chunks
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc="SDF shards"):
                    shard_paths.extend(future.result())

            sdf_sum: Dict[int, np.ndarray] = {}
            sdf_sqs: Dict[int, np.ndarray] = {}
            sdf_cnt: Dict[int, np.ndarray] = {}
            for shard in tqdm(shard_paths, desc="SDF reduce"):
                payload = np.load(shard, allow_pickle=True)
                age_bin = int(payload["age_bin"])
                if age_bin not in sdf_sum:
                    sdf_sum[age_bin] = payload["sum"].astype(np.float64)
                    sdf_sqs[age_bin] = payload["sumsq"].astype(np.float64)
                    sdf_cnt[age_bin] = payload["cnt"].astype(np.int64)
                else:
                    sdf_sum[age_bin] += payload["sum"].astype(np.float64)
                    sdf_sqs[age_bin] += payload["sumsq"].astype(np.float64)
                    sdf_cnt[age_bin] += payload["cnt"].astype(np.int64)

            age_bins = sorted(sdf_cnt.keys())
            template_means: List[np.ndarray] = []
            template_stds: List[np.ndarray] = []
            template_counts: List[np.ndarray] = []
            for age_bin in age_bins:
                count = np.maximum(sdf_cnt[age_bin], 1)[:, None, None, None]
                mean = (sdf_sum[age_bin] / count).astype(np.float32)
                template_means.append(mean.astype(np.float16))
                template_counts.append(sdf_cnt[age_bin].astype(np.int32))
                if save_sdf_std:
                    var = np.maximum((sdf_sqs[age_bin] / count) - mean.astype(np.float64) ** 2, 0.0)
                    template_stds.append(np.sqrt(var).astype(np.float16))

            payload = {
                "ages": np.array(age_bins, dtype=np.int32),
                "T_mean": np.stack(template_means, axis=0) if template_means else np.zeros((0, NUM_CLASSES, *template_shape), dtype=np.float16),
                "count": np.stack(template_counts, axis=0) if template_counts else np.zeros((0, NUM_CLASSES), dtype=np.int32),
                "meta": {
                    "shape": list(template_shape),
                    "band": float(sdf_band),
                    "bin_width": age_bin_width,
                    "dtype": "float16",
                },
            }
            if save_sdf_std and template_stds:
                payload["T_std"] = np.stack(template_stds, axis=0)
            np.savez_compressed(out_path / "sdf_templates.npz", **payload)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[OK] Priors written to {out_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Target-only prior generator")
    parser.add_argument("--split_json", required=True, type=str, help="Target split JSON")
    parser.add_argument("--out_root", required=True, type=str, help="Root output directory")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--age-bin-width", type=float, default=1.0)
    parser.add_argument("--template-shape", type=str, default="128x128x128")
    parser.add_argument("--sdf-band", type=float, default=8.0)
    parser.add_argument("--no-sdf", action="store_true")
    parser.add_argument("--save-sdf-std", action="store_true")
    parser.add_argument("--norm", type=str, choices=["row", "sym"], default="row")
    parser.add_argument("--rmask-policy", type=str, choices=["row-topk", "global-quant"], default="row-topk")
    parser.add_argument("--rmask-arg", type=float, default=0.10)
    parser.add_argument("--sdf-chunk-size", type=int, default=3)
    args = parser.parse_args(argv)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.out_root)
    target_dir = out_root / "target"
    ensure_dir(target_dir)
    template_shape = parse_shape(args.template_shape)

    generate_priors(
        split_json=args.split_json,
        out_dir=str(target_dir),
        workers=int(args.workers),
        age_bin_width=float(args.age_bin_width),
        template_shape=template_shape,
        sdf_band=float(args.sdf_band),
        include_sdf=not args.no_sdf,
        save_sdf_std=bool(args.save_sdf_std),
        adjacency_norm=str(args.norm),
        rmask_policy=str(args.rmask_policy),
        rmask_arg=float(args.rmask_arg),
        sdf_chunk_size=int(args.sdf_chunk_size),
    )


if __name__ == "__main__":
    main()
