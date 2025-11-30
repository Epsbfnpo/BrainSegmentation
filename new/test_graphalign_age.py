#!/usr/bin/env python3
"""Inference and evaluation entrypoint for target-only model."""

import argparse
import json
import itertools
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import nibabel as nib
from nibabel import processing
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from torch.cuda.amp import autocast

from data_loader_age_aware import get_target_test_loader
from train_graphalign_age import (build_model, load_class_mapping,
                                  load_model_weights_only, set_seed)
from extra_metrics import compute_cbdice, compute_clce, compute_cldice


DEBUG = False


def _debug(msg: str, payload: Optional[Dict] = None) -> None:
    if DEBUG:
        if payload is None:
            print(f"[DEBUG] {msg}")
        else:
            print(f"[DEBUG] {msg}: {payload}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Target dataset evaluation")
    parser.add_argument("--split_json", required=True, type=str, help="Target test split JSON")
    parser.add_argument("--model_path", required=True, type=str, help="Checkpoint to evaluate")
    parser.add_argument("--output_dir", default="./test_predictions", type=str, help="Where to write NIfTI predictions")
    parser.add_argument("--metrics_path", default="./analysis/test_metrics.json", type=str, help="Where to store evaluation metrics")
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=87, type=int)
    parser.add_argument("--feature_size", default=48, type=int)
    parser.add_argument("--roi_x", default=128, type=int)
    parser.add_argument("--roi_y", default=128, type=int)
    parser.add_argument("--roi_z", default=128, type=int)
    parser.add_argument("--apply_spacing", dest="apply_spacing", action="store_true", default=True)
    parser.add_argument("--no_apply_spacing", dest="apply_spacing", action="store_false")
    parser.add_argument("--apply_orientation", dest="apply_orientation", action="store_true", default=True)
    parser.add_argument("--no_apply_orientation", dest="apply_orientation", action="store_false")
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[0.8, 0.8, 0.8])
    parser.add_argument("--foreground_only", action="store_true", default=True)
    parser.add_argument("--enhanced_class_weights", action="store_true", default=True)
    parser.add_argument("--volume_stats", type=str, default=None)
    parser.add_argument("--pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--sw_overlap", type=float, default=0.25)
    parser.add_argument("--eval_tta", action="store_true", default=False, help="Enable flip-based test-time augmentation")
    parser.add_argument("--tta_flip_axes", type=int, nargs="*", default=[0], help="Spatial axes for TTA flips (0,1,2)")
    parser.add_argument("--eval_scales", type=float, nargs="*", default=[1.0], help="Optional multi-scale evaluation scales")
    parser.add_argument("--multi_scale", action="store_true", default=False, help="Enable multi-scale evaluation")
    parser.add_argument("--use_amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--use_swin_checkpoint", dest="use_swin_checkpoint", action="store_true", default=True)
    parser.add_argument("--no_swin_checkpoint", dest="use_swin_checkpoint", action="store_false")
    parser.add_argument("--debug_mode", action="store_true", default=False)
    parser.add_argument("--class_map_json", type=str, default=None, help="Optional index-to-raw-label map for saving outputs")
    parser.add_argument(
        "--resample_to_native",
        action="store_true",
        default=True,
        help="Resample predictions back to native headers; disable to keep inference-space geometry",
    )
    parser.add_argument(
        "--resample_tolerance",
        type=float,
        default=0.1,
        help="Relative spacing delta tolerated before skipping native-space resample (fractional)",
    )
    parser.add_argument("--adjacency_prior", type=str, default=None, help="Adjacency prior .npz for spectral distance")
    parser.add_argument("--structural_rules", type=str, default=None, help="JSON of required/forbidden edges")
    parser.add_argument("--laterality_pairs_json", type=str, default=None, help="JSON of left/right label pairs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _prepare_output(
    pred_labels: torch.Tensor,
    *,
    foreground_only: bool,
    brain_mask: Optional[torch.Tensor] = None,
    class_map: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    """Convert network indices to raw labels (0..87) while honouring brain mask.

    The model operates on remapped labels (background=-1 ignored, foreground 0..86).
    For export we must recover the original class IDs: background=0, tissue=1..87,
    and optionally map indices through a provided ``class_map``. A brain mask gates
    any voxels outside the skull to 0 to avoid off-brain artefacts.
    """

    volume = pred_labels.squeeze(0).detach().cpu().numpy().astype(np.int16)

    if brain_mask is not None:
        brain_mask_np = brain_mask.squeeze(0).detach().cpu().numpy().astype(bool)
    else:
        brain_mask_np = np.ones_like(volume, dtype=bool)

    raw_volume = np.zeros_like(volume, dtype=np.int16)

    if class_map:
        for idx, raw_label in class_map.items():
            mask = (volume == idx)
            raw_volume[mask] = int(raw_label)
    else:
        if foreground_only:
            raw_volume = volume + 1  # shift 0..86 -> 1..87
        else:
            raw_volume = volume

    raw_volume = np.where(brain_mask_np, raw_volume, 0)
    return raw_volume.astype(np.int16)


def _coerce_to_str_path(value: Union[str, Path, Sequence, None], key: str) -> Optional[str]:
    """Convert metadata values to a usable path string.

    This normalises list/tuple values to their first element, converts Path-like
    objects to strings, and reports unexpected types when debug logging is enabled.
    """

    original_type = type(value).__name__
    if isinstance(value, (list, tuple)):
        if not value:
            _debug(f"meta[{key}] is empty sequence")
            return None
        _debug(f"meta[{key}] provided as sequence", {"len": len(value), "first_type": type(value[0]).__name__})
        value = value[0]

    if value is None:
        return None

    if isinstance(value, (str, Path, bytes)):
        return str(value)

    try:
        coerced = str(value)
        _debug(f"meta[{key}] coerced to string", {"original_type": original_type, "coerced": coerced})
        return coerced
    except Exception:
        _debug(f"meta[{key}] could not be coerced", {"original_type": original_type})
        return None


def _resolve_affine(meta: Dict) -> np.ndarray:
    """Return a valid 4x4 affine matrix from available metadata.

    Some datasets store affines under different keys (e.g., ``original_affine`` or
    ``sform_affine``). We try a handful of common aliases and finally fall back to
    loading the source file header before resorting to an identity matrix. Non-4x4
    shapes are promoted (3x3) or rejected (others) to avoid nibabel errors.
    """

    candidate = None
    for key in (
        "affine",
        "original_affine",
        "resample_affine",
        "sform_affine",
        "qform_affine",
    ):
        value = meta.get(key)
        if value is not None:
            candidate = value
            _debug("Found affine in meta", {"key": key})
            break

    # If no affine is bundled in metadata, try to read it from the original file.
    filename = _coerce_to_str_path(meta.get("filename_or_obj"), "filename_or_obj")
    if candidate is None and filename:
        try:
            header_affine = nib.load(filename).affine
            candidate = header_affine
            _debug("Loaded affine from source file", {"shape": list(header_affine.shape)})
        except Exception as exc:
            _debug("Failed to load affine from source file", {"error": str(exc)})
            candidate = None

    affine = None
    if candidate is not None:
        if isinstance(candidate, torch.Tensor):
            candidate = candidate.detach().cpu().numpy()
        candidate_np = np.asarray(candidate)
        shape = tuple(candidate_np.shape)

        if shape == (4, 4):
            affine = candidate_np
        elif shape == (3, 3):
            affine = np.eye(4, dtype=np.float32)
            affine[:3, :3] = candidate_np
            _debug("Promoted 3x3 affine to 4x4", {"original_shape": list(shape)})
        elif len(shape) == 3 and shape[0] == 1 and shape[1:] == (4, 4):
            affine = candidate_np[0]
            _debug("Squeezed leading dimension from affine", {"original_shape": list(shape)})
        else:
            _debug("Invalid affine shape; falling back to identity", {"shape": list(shape)})
            affine = None

    if affine is None:
        print("⚠️  Missing affine in metadata; using identity for export")
        affine = np.eye(4, dtype=np.float32)

    return np.asarray(affine, dtype=np.float32)


def _extract_spacing(meta: Dict) -> Optional[np.ndarray]:
    """Read voxel spacing from metadata if available."""

    for key in ("pixdim", "spacing", "original_spacing"):
        value = meta.get(key)
        if value is None:
            continue
        try:
            arr = np.asarray(value).astype(float)
            if arr.ndim == 1 and arr.shape[0] >= 3:
                return arr[:3]
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[0, :3]
        except Exception:
            continue
    return None


def _save_prediction(
    pred_volume: np.ndarray,
    image_meta: Dict,
    label_meta: Dict,
    output_dir: Path,
    *,
    resample_to_native: bool,
    spacing_tolerance: float,
    override_affine: Optional[np.ndarray] = None,
    override_spacing: Optional[np.ndarray] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    preferred_meta = label_meta or image_meta
    filename = _coerce_to_str_path(preferred_meta.get("filename_or_obj", "case"), "filename_or_obj") or "case"
    case_stem = Path(filename).stem.replace(".nii", "")
    if override_affine is not None:
        affine = np.asarray(override_affine, dtype=np.float32)
        _debug("Using override affine from tensor", {"shape": list(affine.shape)})
    else:
        affine = _resolve_affine(preferred_meta)

    # Always save in inference space; optionally resample to native only when spacing matches
    native_resample_done = False

    # Optionally resample back to native header if spacing is compatible; otherwise
    # stay in inference space to avoid geometric distortion.
    if resample_to_native:
        filename_for_resample = _coerce_to_str_path(label_meta.get("filename_or_obj"), "label_filename")
        if not filename_for_resample:
            filename_for_resample = _coerce_to_str_path(image_meta.get("filename_or_obj"), "filename_or_obj")

        if filename_for_resample and os.path.exists(filename_for_resample):
            try:
                target_img = nib.load(filename_for_resample)

                infer_spacing = override_spacing
                if infer_spacing is None:
                    infer_spacing = _extract_spacing(preferred_meta)
                native_spacing = np.asarray(target_img.header.get_zooms()[:3], dtype=float)
                if infer_spacing is not None:
                    rel_diff = np.abs(infer_spacing - native_spacing) / np.maximum(native_spacing, 1e-6)
                    max_rel = float(rel_diff.max())
                    _debug(
                        "Spacing comparison",
                        {
                            "inference_spacing": infer_spacing.tolist(),
                            "native_spacing": native_spacing.tolist(),
                            "max_rel_diff": max_rel,
                        },
                    )
                    if max_rel > spacing_tolerance:
                        print(
                            f"⚠️  Skip native resample for {case_stem}: spacing mismatch (max rel diff {max_rel:.3f} > {spacing_tolerance})"
                        )
                    else:
                        src_img = nib.Nifti1Image(pred_volume, affine)
                        resampled = processing.resample_from_to(src_img, target_img, order=0)
                        pred_volume = np.asarray(resampled.dataobj)
                        affine = resampled.affine
                        brain_mask = np.asarray(target_img.dataobj) > 0
                        pred_volume = np.where(brain_mask, pred_volume, 0).astype(np.int16)
                        native_resample_done = True
                        _debug(
                            "Resampled prediction to native space",
                            {"source_shape": list(src_img.shape), "target_shape": list(target_img.shape)},
                        )
                else:
                    print(f"⚠️  Skip native resample for {case_stem}: missing inference spacing metadata")
            except Exception as exc:
                _debug("Failed to resample prediction to native space", {"error": str(exc)})
        elif resample_to_native:
            print(f"⚠️  Skip native resample for {case_stem}: missing filename for resample")

    out_path = output_dir / f"{case_stem}_pred.nii.gz"
    nib.save(nib.Nifti1Image(pred_volume, affine), str(out_path))
    if native_resample_done:
        _debug("Saved native-space prediction", {"path": str(out_path)})
    else:
        _debug("Saved inference-space prediction", {"path": str(out_path)})
    return out_path


def _select_meta(batch: Dict) -> Dict:
    """Robustly fetch the metadata dictionary for the first sample in a batch."""

    def _as_dict(obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "meta"):
            meta_obj = getattr(obj, "meta")
            if isinstance(meta_obj, dict):
                return meta_obj
        return None

    candidates: List[Dict] = []

    # MONAI's MetaTensor dictionaries (list or dict)
    meta_dicts = batch.get("image_meta_dict")
    if isinstance(meta_dicts, list) and meta_dicts:
        maybe = _as_dict(meta_dicts[0]) or meta_dicts[0]
        if isinstance(maybe, dict):
            candidates.append(maybe)
    if isinstance(meta_dicts, dict):
        candidates.append(meta_dicts)

    # Metadata provided explicitly in the JSON (may not include affine)
    metadata = batch.get("metadata")
    if isinstance(metadata, list) and metadata:
        maybe = _as_dict(metadata[0]) or metadata[0]
        if isinstance(maybe, dict):
            candidates.append(maybe)
    if isinstance(metadata, dict):
        candidates.append(metadata)

    # Meta attached to the image tensor
    image = batch.get("image")
    if isinstance(image, list) and image:
        image = image[0]
    meta_from_image = _as_dict(image)
    if isinstance(meta_from_image, dict):
        candidates.append(meta_from_image)

    # Prefer a candidate that already contains affine-like information
    for meta in candidates:
        if any(k in meta for k in ("affine", "original_affine", "sform_affine", "qform_affine")):
            _debug("Selected meta with affine keys", {"keys": list(meta.keys())})
            return meta

    # Otherwise return the first available candidate or an empty dict
    if candidates:
        _debug("Selected first available meta", {"keys": list(candidates[0].keys())})
        return candidates[0]
    _debug("No metadata found in batch")
    return {}


def _select_label_meta(batch: Dict) -> Dict:
    """Fetch metadata associated with the label tensor when available."""

    def _as_dict(obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "meta"):
            meta_obj = getattr(obj, "meta")
            if isinstance(meta_obj, dict):
                return meta_obj
        return None

    candidates: List[Dict] = []

    meta_dicts = batch.get("label_meta_dict")
    if isinstance(meta_dicts, list) and meta_dicts:
        maybe = _as_dict(meta_dicts[0]) or meta_dicts[0]
        if isinstance(maybe, dict):
            candidates.append(maybe)
    if isinstance(meta_dicts, dict):
        candidates.append(meta_dicts)

    label_obj = batch.get("label")
    if isinstance(label_obj, list) and label_obj:
        label_obj = label_obj[0]
    meta_from_label = _as_dict(label_obj)
    if isinstance(meta_from_label, dict):
        candidates.append(meta_from_label)

    if candidates:
        return candidates[0]
    return {}


def _compute_case_id(batch: Dict) -> str:
    label_meta = _select_label_meta(batch)
    meta = label_meta or _select_meta(batch)
    subject = _coerce_to_str_path(meta.get("subject_id"), "subject_id")
    if subject:
        _debug("Using subject_id for case id", {"value": subject})
        return subject

    fname = _coerce_to_str_path(meta.get("filename_or_obj"), "filename_or_obj")
    if fname:
        stem = Path(fname).stem
        _debug("Using filename_or_obj for case id", {"value": fname, "stem": stem})
        return stem

    _debug("Falling back to default case id", {"meta_keys": list(meta.keys())})
    return "case"


def _compute_case_dice(
    preds: torch.Tensor,
    target: torch.Tensor,
    *,
    include_background: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-case Dice scores from one-hot predictions and labels.

    Args:
        preds: one-hot predictions shaped (B, C, H, W, D).
        target: one-hot labels shaped (B, C, H, W, D).
        include_background: whether channel 0 is included in the Dice computation.
        eps: small value to avoid division by zero.

    Returns:
        Dice score tensor shaped (B, C) if include_background else (B, C-1).
    """

    if preds.shape != target.shape:
        raise ValueError(f"preds and target must share shape; got {preds.shape} vs {target.shape}")

    # Drop background channel if excluded
    if not include_background:
        preds = preds[:, 1:]
        target = target[:, 1:]

    reduce_dims = tuple(range(2, preds.ndim))
    intersection = (preds * target).sum(dim=reduce_dims)
    pred_sum = preds.sum(dim=reduce_dims)
    target_sum = target.sum(dim=reduce_dims)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    return dice


class AdvancedMetrics:
    """Bundle topology, geometry, and clinical-style evaluation helpers."""

    def __init__(
        self,
        num_classes: int,
        *,
        include_background: bool,
        adjacency_prior: Optional[str] = None,
        structural_rules: Optional[str] = None,
        laterality_pairs: Optional[str] = None,
        device: torch.device,
    ) -> None:
        self.num_classes = int(num_classes)
        self.include_background = bool(include_background)
        self.device = device

        self.adj_templates: Optional[torch.Tensor] = None
        self.adj_age_values: Optional[torch.Tensor] = None
        self.adj_bin_width: float = 1.0
        self.required_edges: List[Tuple[int, int]] = []
        self.forbidden_edges: List[Tuple[int, int]] = []
        self.lr_pairs: List[Tuple[int, int]] = []

        if adjacency_prior and os.path.exists(adjacency_prior):
            self._load_adjacency_prior(adjacency_prior)
        if structural_rules and os.path.exists(structural_rules):
            self._load_structural_rules(structural_rules)
        if laterality_pairs and os.path.exists(laterality_pairs):
            self._load_lr_pairs(laterality_pairs)

    def _align_classes_3d(self, array: np.ndarray) -> np.ndarray:
        if array.shape[-1] >= self.num_classes:
            array = array[..., : self.num_classes]
        else:
            pad_width = self.num_classes - array.shape[-1]
            array = np.pad(array, (*[(0, 0)] * (array.ndim - 1), (0, pad_width)), mode="constant")
        if array.shape[-2] >= self.num_classes:
            array = array[:, : self.num_classes, :]
        else:
            pad_height = self.num_classes - array.shape[-2]
            array = np.pad(array, (*[(0, 0)] * (array.ndim - 2), (0, pad_height), (0, 0)), mode="constant")
        return array

    def _load_adjacency_prior(self, path: str) -> None:
        payload = np.load(path, allow_pickle=True)
        ages = payload.get("ages")
        matrices = payload.get("A_prior")
        meta = payload.get("meta", {})
        if ages is None or matrices is None:
            return
        self.adj_bin_width = float(meta.get("bin_width", 1.0)) if isinstance(meta, dict) else 1.0
        age_values = ages.astype(np.float32) * self.adj_bin_width
        order = np.argsort(age_values)
        matrices = matrices[order]
        age_values = age_values[order]
        matrices = self._align_classes_3d(matrices.astype(np.float32))
        self.adj_templates = torch.tensor(matrices, dtype=torch.float32, device=self.device)
        self.adj_age_values = torch.tensor(age_values, dtype=torch.float32, device=self.device)

    def _load_structural_rules(self, path: str) -> None:
        with open(path, "r") as f:
            payload = json.load(f)
        required = payload.get("required", []) or []
        forbidden = payload.get("forbidden", []) or []
        self.required_edges = self._filter_rules(required)
        self.forbidden_edges = self._filter_rules(forbidden)

    def _filter_rules(self, rules: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for pair in rules:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                continue
            i, j = int(pair[0]), int(pair[1])
            if not self.include_background:
                i -= 1
                j -= 1
            if 0 <= i < self.num_classes and 0 <= j < self.num_classes and i != j:
                pairs.append((i, j))
        return pairs

    def _load_lr_pairs(self, path: str) -> None:
        with open(path, "r") as f:
            payload = json.load(f)
        pairs: List[Tuple[int, int]] = []
        for pair in payload:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                continue
            left, right = int(pair[0]), int(pair[1])
            if left <= 0 or right <= 0:
                continue
            if not self.include_background:
                left -= 1
                right -= 1
            if 0 <= left < self.num_classes and 0 <= right < self.num_classes:
                pairs.append((left, right))
        self.lr_pairs = pairs

    def _row_normalise(self, mat: torch.Tensor) -> torch.Tensor:
        mat = mat.clone()
        mat = mat - torch.diag_embed(torch.diagonal(mat))
        rowsum = mat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return mat / rowsum

    def _laplacian(self, adj: torch.Tensor) -> torch.Tensor:
        adj_sym = 0.5 * (adj + adj.transpose(-1, -2))
        deg = torch.diag_embed(adj_sym.sum(dim=-1))
        return deg - adj_sym

    def compute_adjacency(self, labels: torch.Tensor, brain_mask: torch.Tensor) -> torch.Tensor:
        adj = torch.zeros((self.num_classes, self.num_classes), device=self.device)

        effective_labels = torch.where(brain_mask, labels, torch.full_like(labels, -1))
        if not self.include_background:
            effective_labels = torch.where(effective_labels <= 0, torch.full_like(effective_labels, -1), effective_labels)

        shifts = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        for shift in shifts:
            rolled = torch.roll(effective_labels, shifts=shift, dims=(0, 1, 2))
            valid = (effective_labels >= 0) & (rolled >= 0) & (effective_labels != rolled)
            if not torch.any(valid):
                continue
            src = effective_labels[valid].flatten().long()
            dst = rolled[valid].flatten().long()
            pairs = torch.stack([src, dst], dim=1)
            pairs = torch.unique(pairs, dim=0)
            adj[pairs[:, 0], pairs[:, 1]] = 1
            adj[pairs[:, 1], pairs[:, 0]] = 1
        adj = adj - torch.diag_embed(torch.diagonal(adj))
        return adj

    def _nearest_adj_prior(self, age: float) -> Optional[torch.Tensor]:
        if self.adj_templates is None or self.adj_age_values is None:
            return None
        age_tensor = torch.tensor(float(age), device=self.device)
        idx = torch.searchsorted(self.adj_age_values, age_tensor).clamp(0, self.adj_age_values.numel() - 1)
        return self.adj_templates[int(idx.item())]

    def compute_spectral_distance(self, adj: torch.Tensor, age: float, top_k: int = 20) -> Optional[float]:
        prior_adj = self._nearest_adj_prior(age)
        if prior_adj is None:
            return None
        adj_norm = self._row_normalise(adj.float())
        prior_norm = self._row_normalise(prior_adj.float())
        lap_pred = self._laplacian(adj_norm)
        lap_prior = self._laplacian(prior_norm)
        try:
            eig_pred = torch.linalg.eigvalsh(lap_pred)
            eig_prior = torch.linalg.eigvalsh(lap_prior)
            k = max(1, min(top_k, min(eig_pred.numel(), eig_prior.numel())))
            return float(torch.mean((eig_pred[:k] - eig_prior[:k]) ** 2).item())
        except Exception:
            return None

    def compute_structural_violations(self, adj: torch.Tensor) -> Dict[str, int]:
        violations = {"forbidden": 0, "required": 0}
        for i, j in self.forbidden_edges:
            if adj[i, j] > 0:
                violations["forbidden"] += 1
        for i, j in self.required_edges:
            if adj[i, j] == 0:
                violations["required"] += 1
        return violations

    def compute_symmetry(self, labels: torch.Tensor, brain_mask: torch.Tensor) -> float:
        if not self.lr_pairs:
            return 0.0
        effective = torch.where(brain_mask, labels, torch.full_like(labels, -1))
        if not self.include_background:
            effective = torch.where(effective <= 0, torch.full_like(effective, -1), effective)
        scores: List[float] = []
        for left, right in self.lr_pairs:
            vol_left = torch.count_nonzero(effective == left).item()
            vol_right = torch.count_nonzero(effective == right).item()
            if vol_left + vol_right > 0:
                scores.append(1.0 - abs(vol_left - vol_right) / float(vol_left + vol_right))
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def compute_rve(self, pred: torch.Tensor, target: torch.Tensor, brain_mask: torch.Tensor) -> float:
        pred_eff = torch.where(brain_mask, pred, torch.full_like(pred, -1))
        target_eff = torch.where(brain_mask, target, torch.full_like(target, -1))
        if not self.include_background:
            pred_eff = torch.where(pred_eff <= 0, torch.full_like(pred_eff, -1), pred_eff)
            target_eff = torch.where(target_eff <= 0, torch.full_like(target_eff, -1), target_eff)
        classes = torch.unique(target_eff)
        classes = classes[classes >= 0]
        errors: List[float] = []
        for cls in classes.tolist():
            vol_pred = torch.count_nonzero(pred_eff == cls).item()
            vol_gt = torch.count_nonzero(target_eff == cls).item()
            if vol_gt > 0:
                errors.append(abs(vol_pred - vol_gt) / float(vol_gt))
        if not errors:
            return 0.0
        return float(sum(errors) / len(errors))


def evaluate(args: argparse.Namespace) -> Dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        args.use_amp = False

    test_loader = get_target_test_loader(args, keep_meta=True)
    model = build_model(args, device)
    load_model_weights_only(model, Path(args.model_path))
    model.eval()

    dice_metric = DiceMetric(include_background=not args.foreground_only, reduction="mean_batch")
    per_class_metric = DiceMetric(
        include_background=not args.foreground_only,
        reduction="none",
        get_not_nans=True,
    )
    hd95_metric = HausdorffDistanceMetric(
        include_background=not args.foreground_only, percentile=95, reduction="mean_batch"
    )
    assd_metric = SurfaceDistanceMetric(
        include_background=not args.foreground_only, symmetric=True, reduction="mean_batch"
    )

    adv_metrics = AdvancedMetrics(
        num_classes=args.out_channels,
        include_background=not args.foreground_only,
        adjacency_prior=args.adjacency_prior,
        structural_rules=args.structural_rules,
        laterality_pairs=args.laterality_pairs_json,
        device=device,
    )

    class_mapping = load_class_mapping(Path(args.class_map_json)) if args.class_map_json else {}
    predictions_dir = Path(args.output_dir)

    per_case: List[Dict] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            labels_eval = labels.clone()
            labels_eval[labels_eval < 0] = 0
            labels_eval = labels_eval.long()
            if labels.ndim == 5 and labels.shape[1] == 1:
                valid_mask = (labels >= 0).squeeze(1)
            else:
                valid_mask = labels >= 0
            brain_mask = labels_eval.clone()
            if brain_mask.ndim == 5 and brain_mask.shape[1] == 1:
                brain_mask = brain_mask.squeeze(1)
            brain_mask = brain_mask > 0

            def _run_model(volume: torch.Tensor) -> torch.Tensor:
                eval_scales = list(args.eval_scales or [1.0])
                if not args.multi_scale and all(abs(s - 1.0) < 1e-6 for s in eval_scales):
                    return sliding_window_inference(
                        volume,
                        roi_size=(args.roi_x, args.roi_y, args.roi_z),
                        sw_batch_size=args.sw_batch_size,
                        predictor=model,
                        overlap=args.sw_overlap,
                    )

                logits_multi: List[torch.Tensor] = []
                for scale in eval_scales:
                    if scale != 1.0:
                        scaled = F.interpolate(
                            volume,
                            scale_factor=scale,
                            mode="trilinear",
                            align_corners=False,
                            recompute_scale_factor=True,
                        )
                    else:
                        scaled = volume
                    logits_scaled = sliding_window_inference(
                        scaled,
                        roi_size=(args.roi_x, args.roi_y, args.roi_z),
                        sw_batch_size=args.sw_batch_size,
                        predictor=model,
                        overlap=args.sw_overlap,
                    )
                    if scale != 1.0:
                        logits_scaled = F.interpolate(
                            logits_scaled,
                            size=volume.shape[2:],
                            mode="trilinear",
                            align_corners=False,
                        )
                    logits_multi.append(logits_scaled)
                return torch.stack(logits_multi, dim=0).mean(dim=0)

            def _flip_tensor(tensor: torch.Tensor, axes):
                if not axes:
                    return tensor
                spatial_dims = list(range(tensor.ndim - 3, tensor.ndim))
                dims = [spatial_dims[ax] for ax in axes if ax < len(spatial_dims)]
                return torch.flip(tensor, dims=dims)

            with autocast(enabled=args.use_amp):
                if args.eval_tta and args.tta_flip_axes:
                    combos = [()]
                    axes = tuple(sorted({int(ax) for ax in (args.tta_flip_axes or []) if ax in (0, 1, 2)}))
                    for r in range(1, len(axes) + 1):
                        combos.extend(itertools.combinations(axes, r))
                    logits_list = []
                    for combo in combos:
                        flipped = _flip_tensor(images, combo)
                        logits_combo = _run_model(flipped)
                        logits_combo = _flip_tensor(logits_combo, combo)
                        logits_list.append(logits_combo)
                    logits = torch.stack(logits_list, dim=0).mean(dim=0)
                else:
                    logits = _run_model(images)

            probs = torch.softmax(logits, dim=1)
            pred_labels = torch.argmax(probs, dim=1)
            pred_labels = torch.where(brain_mask, pred_labels, torch.zeros_like(pred_labels))
            preds = F.one_hot(pred_labels, num_classes=args.out_channels)
            preds = preds.permute(0, 4, 1, 2, 3).to(dtype=probs.dtype)

            labels_eval_wo_channel = labels_eval
            if labels_eval_wo_channel.ndim == 5 and labels_eval_wo_channel.shape[1] == 1:
                labels_eval_wo_channel = labels_eval_wo_channel.squeeze(1)
            labels_eval_wo_channel = torch.where(brain_mask, labels_eval_wo_channel, torch.zeros_like(labels_eval_wo_channel))
            target = F.one_hot(labels_eval_wo_channel, num_classes=args.out_channels)
            target = target.permute(0, 4, 1, 2, 3).to(dtype=probs.dtype)
            preds = preds * brain_mask.unsqueeze(1)
            target = target * brain_mask.unsqueeze(1)

            dice_metric(y_pred=preds, y=target)
            per_class_metric(y_pred=preds, y=target)
            hd95_batch = hd95_metric(y_pred=preds, y=target)
            assd_batch = assd_metric(y_pred=preds, y=target)

            case_dice = _compute_case_dice(
                preds=preds,
                target=target,
                include_background=not args.foreground_only,
            )
            case_dice_value = float(case_dice.mean().item()) if case_dice.numel() > 0 else 0.0
            case_hd95_value = hd95_batch
            if isinstance(case_hd95_value, torch.Tensor):
                case_hd95_value = torch.nan_to_num(case_hd95_value, nan=0.0, posinf=0.0, neginf=0.0)
                case_hd95_value = float(case_hd95_value.mean().item()) if case_hd95_value.numel() > 0 else 0.0
            else:
                case_hd95_value = float(case_hd95_value)

            case_assd_value = assd_batch
            if isinstance(case_assd_value, torch.Tensor):
                case_assd_value = torch.nan_to_num(case_assd_value, nan=0.0, posinf=0.0, neginf=0.0)
                case_assd_value = float(case_assd_value.mean().item()) if case_assd_value.numel() > 0 else 0.0
            else:
                case_assd_value = float(case_assd_value)

            age_value = 40.0
            if "age" in batch:
                age_tensor = batch["age"]
                if isinstance(age_tensor, torch.Tensor):
                    age_value = float(age_tensor.flatten()[0].item())
                else:
                    try:
                        age_value = float(age_tensor)
                    except (TypeError, ValueError):
                        age_value = 40.0

            brain_mask_vol = brain_mask
            if brain_mask_vol.ndim == 4:
                brain_mask_vol = brain_mask_vol[0]
            pred_labels_vol = pred_labels
            if pred_labels_vol.ndim == 4:
                pred_labels_vol = pred_labels_vol[0]
            target_labels_vol = labels_eval_wo_channel
            if target_labels_vol.ndim == 4:
                target_labels_vol = target_labels_vol[0]

            adjacency = adv_metrics.compute_adjacency(pred_labels_vol, brain_mask_vol)
            spec_distance = adv_metrics.compute_spectral_distance(adjacency, age_value)
            violations = adv_metrics.compute_structural_violations(adjacency)
            symmetry_score = adv_metrics.compute_symmetry(pred_labels_vol, brain_mask_vol)
            rve_score = adv_metrics.compute_rve(pred_labels_vol, target_labels_vol, brain_mask_vol)

            pred_np = pred_labels_vol.detach().cpu().numpy()
            target_np = target_labels_vol.detach().cpu().numpy()

            cldice_sum = 0.0
            cbdice_sum = 0.0
            valid_class_count = 0

            for c in range(1, args.out_channels):
                p_c = pred_np == c
                t_c = target_np == c

                if np.sum(t_c) > 0:
                    cldice_sum += compute_cldice(p_c, t_c)
                    cbdice_sum += compute_cbdice(p_c, t_c)
                    valid_class_count += 1

            case_cldice = cldice_sum / valid_class_count if valid_class_count > 0 else 0.0
            case_cbdice = cbdice_sum / valid_class_count if valid_class_count > 0 else 0.0

            labels_for_clce = torch.where(brain_mask.unsqueeze(1), labels_eval, torch.zeros_like(labels_eval))
            case_clce = compute_clce(logits, labels_for_clce)

            case_id = _compute_case_id(batch)
            meta_dict = _select_meta(batch)
            label_meta = _select_label_meta(batch)

            inference_affine: Optional[np.ndarray] = None
            if hasattr(images, "affine"):
                affine_tensor = images.affine
                affine_np = affine_tensor.detach().cpu().numpy() if isinstance(affine_tensor, torch.Tensor) else np.asarray(affine_tensor)
                if affine_np.ndim == 3 and affine_np.shape[0] >= 1:
                    inference_affine = affine_np[0]
                elif affine_np.shape == (4, 4):
                    inference_affine = affine_np

            inference_spacing: Optional[np.ndarray] = None
            image_meta = getattr(images, "meta", None)
            if isinstance(image_meta, dict):
                inference_spacing = _extract_spacing(image_meta)

            pred_volume = _prepare_output(
                pred_labels,
                foreground_only=args.foreground_only,
                brain_mask=brain_mask,
                class_map=class_mapping,
            )
            pred_path = _save_prediction(
                pred_volume,
                meta_dict,
                label_meta,
                predictions_dir,
                resample_to_native=args.resample_to_native,
                spacing_tolerance=args.resample_tolerance,
                override_affine=inference_affine,
                override_spacing=inference_spacing,
            )

            per_case.append({
                "index": batch_idx,
                "case_id": case_id,
                "dice": case_dice_value,
                "hd95": case_hd95_value,
                "assd": case_assd_value,
                "rve": rve_score,
                "symmetry": symmetry_score,
                "spec_distance": spec_distance if spec_distance is not None else None,
                "cldice": case_cldice,
                "cbdice": case_cbdice,
                "clce": case_clce,
                "violation_forbidden": violations.get("forbidden", 0),
                "violation_required": violations.get("required", 0),
                "prediction_path": str(pred_path),
            })

    dice = dice_metric.aggregate()
    dice_metric.reset()
    if isinstance(dice, torch.Tensor):
        dice = float(dice.mean().item()) if dice.numel() > 0 else 0.0
    else:
        dice = float(dice)

    per_class_scores: Optional[List[float]] = None
    per_class_valid: Optional[List[float]] = None
    aggregate_out = per_class_metric.aggregate()
    per_class_metric.reset()
    if aggregate_out is not None:
        if isinstance(aggregate_out, (tuple, list)) and len(aggregate_out) == 2:
            scores, counts = aggregate_out
        else:
            scores, counts = aggregate_out, None

        if isinstance(scores, torch.Tensor):
            scores = scores.mean(dim=0) if scores.ndim >= 2 else scores
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            per_class_scores = scores.detach().cpu().tolist()
        if isinstance(counts, torch.Tensor):
            counts = counts.sum(dim=0) if counts.ndim >= 2 else counts
            counts = torch.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
            per_class_valid = counts.detach().cpu().tolist()

    hd95 = hd95_metric.aggregate()
    hd95_metric.reset()
    if isinstance(hd95, torch.Tensor):
        hd95 = torch.nan_to_num(hd95, nan=0.0, posinf=0.0, neginf=0.0)
        hd95 = float(hd95.mean().item()) if hd95.numel() > 0 else 0.0
    else:
        hd95 = float(hd95)

    assd = assd_metric.aggregate()
    assd_metric.reset()
    if isinstance(assd, torch.Tensor):
        assd = torch.nan_to_num(assd, nan=0.0, posinf=0.0, neginf=0.0)
        assd = float(assd.mean().item()) if assd.numel() > 0 else 0.0
    else:
        assd = float(assd)

    spec_values = [c.get("spec_distance") for c in per_case if c.get("spec_distance") is not None]
    rve_values = [c.get("rve", 0.0) for c in per_case]
    symmetry_values = [c.get("symmetry", 0.0) for c in per_case]
    cldice_values = [c.get("cldice", 0.0) for c in per_case]
    cbdice_values = [c.get("cbdice", 0.0) for c in per_case]
    clce_values = [c.get("clce", 0.0) for c in per_case]
    mean_spec = float(sum(spec_values) / len(spec_values)) if spec_values else 0.0
    mean_rve = float(sum(rve_values) / len(rve_values)) if rve_values else 0.0
    mean_sym = float(sum(symmetry_values) / len(symmetry_values)) if symmetry_values else 0.0
    mean_cldice = float(sum(cldice_values) / len(cldice_values)) if cldice_values else 0.0
    mean_cbdice = float(sum(cbdice_values) / len(cbdice_values)) if cbdice_values else 0.0
    mean_clce = float(sum(clce_values) / len(clce_values)) if clce_values else 0.0
    total_forbidden = int(sum(c.get("violation_forbidden", 0) for c in per_case))
    total_required = int(sum(c.get("violation_required", 0) for c in per_case))

    metrics = {
        "mean_dice": dice,
        "mean_hd95": hd95,
        "mean_assd": assd,
        "mean_rve": mean_rve,
        "mean_symmetry": mean_sym,
        "mean_spec_distance": mean_spec,
        "mean_cldice": mean_cldice,
        "mean_cbdice": mean_cbdice,
        "mean_clce": mean_clce,
        "total_violations_forbidden": total_forbidden,
        "total_violations_required": total_required,
        "per_class_dice": per_class_scores,
        "per_class_valid_counts": per_class_valid,
        "cases": per_case,
    }

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    args = parse_args()
    global DEBUG
    DEBUG = args.debug_mode
    _debug("Debug mode enabled")
    metrics = evaluate(args)
    print("=" * 40)
    print(f"Evaluation Results (N={len(metrics.get('cases', []))}):")
    print(f"  Mean Dice:          {metrics.get('mean_dice', 0.0):.4f}")
    print(f"  Mean HD95:          {metrics.get('mean_hd95', 0.0):.4f}")
    print(f"  Mean ASSD:          {metrics.get('mean_assd', 0.0):.4f}")
    print(f"  Mean clDice:        {metrics.get('mean_cldice', 0.0):.4f}")
    print(f"  Mean cbDice:        {metrics.get('mean_cbdice', 0.0):.4f}")
    print(f"  Mean clCE:          {metrics.get('mean_clce', 0.0):.4f}")
    print(f"  Mean RVE:           {metrics.get('mean_rve', 0.0):.4f}")
    print(f"  Mean Symmetry:      {metrics.get('mean_symmetry', 0.0):.4f}")
    print(f"  Mean Spec Dist:     {metrics.get('mean_spec_distance', 0.0):.4f}")
    print(
        f"  Struct Violations:  F={metrics.get('total_violations_forbidden')}, "
        f"R={metrics.get('total_violations_required')}"
    )
    print("=" * 40)

    print(json.dumps({"mean_dice": metrics.get("mean_dice", 0.0), "cases": len(metrics.get("cases", []))}, indent=2))


if __name__ == "__main__":
    main()
