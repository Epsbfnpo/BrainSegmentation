#!/usr/bin/env python3
"""Inference and evaluation entrypoint for target-only model."""

import argparse
import json
import itertools
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.cuda.amp import autocast

from data_loader_age_aware import get_target_test_loader
from train_graphalign_age import (build_model, load_class_mapping,
                                  load_model_weights_only, set_seed)


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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _prepare_output(pred_labels: torch.Tensor, *, foreground_only: bool) -> np.ndarray:
    volume = pred_labels.squeeze(0).detach().cpu().numpy().astype(np.int16)
    if foreground_only:
        # Model predicts 0..86 where 0 corresponds to raw label 1.
        volume = np.where(volume >= 0, volume + 1, 0)
    return volume


def _resolve_affine(meta: Dict) -> np.ndarray:
    """Return an affine matrix from available metadata.

    Some datasets store affines under different keys (e.g., ``original_affine`` or
    ``sform_affine``). We try a handful of common aliases and finally fall back to
    loading the source file header before resorting to an identity matrix.
    """

    affine = None
    for key in (
        "affine",
        "original_affine",
        "resample_affine",
        "sform_affine",
        "qform_affine",
    ):
        affine = meta.get(key)
        if affine is not None:
            break

    # If no affine is bundled in metadata, try to read it from the original file.
    if affine is None and meta.get("filename_or_obj"):
        try:
            header_affine = nib.load(meta["filename_or_obj"]).affine
            affine = header_affine
        except Exception:
            affine = None

    if affine is None:
        print("⚠️  Missing affine in metadata; using identity for export")
        affine = np.eye(4, dtype=np.float32)

    if isinstance(affine, torch.Tensor):
        affine = affine.cpu().numpy()
    return np.asarray(affine)


def _save_prediction(pred_volume: np.ndarray, meta: Dict, output_dir: Path, class_map: Optional[Dict[int, int]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(meta.get("filename_or_obj", "case"))
    case_stem = filename.stem.replace(".nii", "")
    if class_map:
        remapped = np.zeros_like(pred_volume, dtype=np.int16)
        for idx, raw_label in class_map.items():
            remapped[pred_volume == idx + 1] = int(raw_label)
        pred_volume = remapped
    affine = _resolve_affine(meta)
    out_path = output_dir / f"{case_stem}_pred.nii.gz"
    nib.save(nib.Nifti1Image(pred_volume, affine), str(out_path))
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
            return meta

    # Otherwise return the first available candidate or an empty dict
    if candidates:
        return candidates[0]
    return {}


def _compute_case_id(batch: Dict) -> str:
    meta = _select_meta(batch)
    if "subject_id" in meta:
        return str(meta["subject_id"])
    if "filename_or_obj" in meta:
        return Path(meta.get("filename_or_obj", "case")).stem
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
            preds = F.one_hot(pred_labels, num_classes=args.out_channels)
            preds = preds.permute(0, 4, 1, 2, 3).to(dtype=probs.dtype)

            labels_eval_wo_channel = labels_eval
            if labels_eval_wo_channel.ndim == 5 and labels_eval_wo_channel.shape[1] == 1:
                labels_eval_wo_channel = labels_eval_wo_channel.squeeze(1)
            target = F.one_hot(labels_eval_wo_channel, num_classes=args.out_channels)
            target = target.permute(0, 4, 1, 2, 3).to(dtype=probs.dtype)

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
            pred_volume = _prepare_output(pred_labels, foreground_only=args.foreground_only)
            pred_path = _save_prediction(pred_volume, meta_dict, predictions_dir, class_mapping)

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
    metrics = evaluate(args)
    print(json.dumps({"mean_dice": metrics.get("mean_dice", 0.0), "cases": len(metrics.get("cases", []))}, indent=2))


if __name__ == "__main__":
    main()
