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
from monai.metrics.utils import compute_dice
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
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--sw_overlap", type=float, default=0.25)
    parser.add_argument("--eval_tta", action="store_true", default=False, help="Enable flip-based test-time augmentation")
    parser.add_argument("--tta_flip_axes", type=int, nargs="*", default=[0], help="Spatial axes for TTA flips (0,1,2)")
    parser.add_argument("--eval_scales", type=float, nargs="*", default=[1.0], help="Optional multi-scale evaluation scales")
    parser.add_argument("--multi_scale", action="store_true", default=False, help="Enable multi-scale evaluation")
    parser.add_argument("--use_amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
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
    affine = meta.get("affine") or meta.get("original_affine")
    if affine is None:
        raise ValueError("Missing affine information in metadata; cannot save NIfTI")
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


def _compute_case_id(batch: Dict) -> str:
    metadata_list = batch.get("metadata")
    if metadata_list and isinstance(metadata_list, list):
        meta = metadata_list[0] or {}
        if isinstance(meta, dict) and "subject_id" in meta:
            return str(meta["subject_id"])
    meta_dicts = batch.get("image_meta_dict")
    if meta_dicts:
        meta = meta_dicts[0]
        return Path(meta.get("filename_or_obj", "case")).stem
    return "case"


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

            case_dice = compute_dice(
                y_pred=preds,
                y=target,
                include_background=not args.foreground_only,
                ignore_empty=False,
            )
            case_dice_value = float(case_dice.mean().item()) if case_dice.numel() > 0 else 0.0

            case_id = _compute_case_id(batch)
            meta_dict = batch.get("image_meta_dict", [{}])[0]
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
