#!/usr/bin/env python3
"""Inference and evaluation entrypoint for target-only model."""

import argparse
import json
import itertools
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import nibabel as nib
from nibabel import processing
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.cuda.amp import autocast

from data_loader_age_aware import get_target_test_loader
from train_graphalign_age import (build_model, load_class_mapping,
                                  load_model_weights_only, set_seed)


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
        default=False,
        help="Resample predictions back to native headers; disable to keep inference-space geometry",
    )
    parser.add_argument(
        "--resample_tolerance",
        type=float,
        default=0.1,
        help="Relative spacing delta tolerated before skipping native-space resample (fractional)",
    )
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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    preferred_meta = label_meta or image_meta
    filename = _coerce_to_str_path(preferred_meta.get("filename_or_obj", "case"), "filename_or_obj") or "case"
    case_stem = Path(filename).stem.replace(".nii", "")
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

            case_dice = _compute_case_dice(
                preds=preds,
                target=target,
                include_background=not args.foreground_only,
            )
            case_dice_value = float(case_dice.mean().item()) if case_dice.numel() > 0 else 0.0

            case_id = _compute_case_id(batch)
            meta_dict = _select_meta(batch)
            label_meta = _select_label_meta(batch)
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
            )

            per_case.append({
                "index": batch_idx,
                "case_id": case_id,
                "dice": case_dice_value,
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

    metrics = {
        "mean_dice": dice,
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
    print(json.dumps({"mean_dice": metrics.get("mean_dice", 0.0), "cases": len(metrics.get("cases", []))}, indent=2))


if __name__ == "__main__":
    main()
