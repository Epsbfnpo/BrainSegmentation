#!/usr/bin/env python3
"""
TRM Method Evaluation Script (V7 - Robust Fix)
- 修复: 使用 try-except 和 torch.as_tensor 彻底解决 Tensor 转换报错。
- 修复: 显式处理 NaN 值，防止某些类别无预测时导致结果为 NaN。
- 功能: 完整保留所有 Evaluation Metrics (Dice, HD95, RVE, Skeleton, etc.)。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
from nibabel import processing
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from torch.cuda.amp import autocast

# --- Import TRM Components ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import get_inference_loader
from modules import SwinUNETRWrapper

# --- Import Extra Metrics (clDice etc.) ---
try:
    from extra_metrics import compute_cbdice, compute_clce, compute_cldice
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent / "new"))
    try:
        from extra_metrics import compute_cbdice, compute_clce, compute_cldice
    except ImportError:
        print("Error: Could not import 'extra_metrics.py'. Please ensure 'new/extra_metrics.py' exists.")
        sys.exit(1)

DEBUG = False


def parse_args():
    parser = argparse.ArgumentParser(description="TRM Evaluation V7")

    # Paths
    parser.add_argument("--split_json", required=True, help="Path to test split JSON")
    parser.add_argument("--model_path", required=True, help="Path to best.ckpt")
    parser.add_argument("--output_dir", default="./test_predictions_v7", help="Directory for outputs")
    parser.add_argument("--metrics_path", default="./analysis/test_metrics_v7.json")

    # Model config
    parser.add_argument("--roi_x", type=int, default=96)
    parser.add_argument("--roi_y", type=int, default=96)
    parser.add_argument("--roi_z", type=int, default=96)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=87)
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--no_swin_checkpoint", action="store_true")

    # Inference config
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[0.8, 0.8, 0.8])
    parser.add_argument("--apply_spacing", action="store_true", default=True)
    parser.add_argument("--apply_orientation", action="store_true", default=True)
    parser.add_argument("--foreground_only", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--sw_overlap", type=float, default=0.25)
    parser.add_argument("--use_amp", action="store_true", default=True)

    # Advanced Metrics Args
    parser.add_argument("--adjacency_prior", type=str, default=None)
    parser.add_argument("--structural_rules", type=str, default=None)
    parser.add_argument("--laterality_pairs_json", type=str, default=None)

    # Resampling
    parser.add_argument("--resample_to_native", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_mode", action="store_true")

    return parser.parse_args()


# --- Advanced Metrics Logic ---
class AdvancedMetrics:
    def __init__(self, num_classes: int, *, include_background: bool, adjacency_prior: Optional[str] = None,
                 structural_rules: Optional[str] = None, laterality_pairs: Optional[str] = None,
                 device: torch.device) -> None:
        self.num_classes = int(num_classes)
        self.include_background = bool(include_background)
        self.device = device
        self.adj_templates: Optional[torch.Tensor] = None
        self.adj_age_values: Optional[torch.Tensor] = None
        self.required_edges: List[Tuple[int, int]] = []
        self.forbidden_edges: List[Tuple[int, int]] = []
        self.lr_pairs: List[Tuple[int, int]] = []

        if adjacency_prior and os.path.exists(adjacency_prior): self._load_adjacency_prior(adjacency_prior)
        if structural_rules and os.path.exists(structural_rules): self._load_structural_rules(structural_rules)
        if laterality_pairs and os.path.exists(laterality_pairs): self._load_lr_pairs(laterality_pairs)

    def _align_classes_3d(self, array: np.ndarray) -> np.ndarray:
        if array.shape[-1] >= self.num_classes:
            array = array[..., : self.num_classes]
        else:
            array = np.pad(array, (*[(0, 0)] * (array.ndim - 1), (0, self.num_classes - array.shape[-1])),
                           mode="constant")
        if array.shape[-2] >= self.num_classes:
            array = array[:, : self.num_classes, :]
        else:
            array = np.pad(array, (*[(0, 0)] * (array.ndim - 2), (0, self.num_classes - array.shape[-2]), (0, 0)),
                           mode="constant")
        return array

    def _load_adjacency_prior(self, path: str) -> None:
        payload = np.load(path, allow_pickle=True)
        ages, matrices = payload.get("ages"), payload.get("A_prior")
        meta = payload.get("meta", {})
        if ages is None or matrices is None: return
        bin_width = float(meta.get("bin_width", 1.0)) if isinstance(meta, dict) else 1.0
        age_values = ages.astype(np.float32) * bin_width
        order = np.argsort(age_values)
        matrices = self._align_classes_3d(matrices[order].astype(np.float32))
        self.adj_templates = torch.tensor(matrices, dtype=torch.float32, device=self.device)
        self.adj_age_values = torch.tensor(age_values[order], dtype=torch.float32, device=self.device)

    def _load_structural_rules(self, path: str) -> None:
        with open(path, "r") as f: payload = json.load(f)
        self.required_edges = self._filter_rules(payload.get("required", []) or [])
        self.forbidden_edges = self._filter_rules(payload.get("forbidden", []) or [])

    def _filter_rules(self, rules: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for pair in rules:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2): continue
            i, j = int(pair[0]), int(pair[1])
            if not self.include_background: i -= 1; j -= 1
            if 0 <= i < self.num_classes and 0 <= j < self.num_classes and i != j: pairs.append((i, j))
        return pairs

    def _load_lr_pairs(self, path: str) -> None:
        with open(path, "r") as f:
            payload = json.load(f)
        pairs: List[Tuple[int, int]] = []
        for pair in payload:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2): continue
            l, r = int(pair[0]), int(pair[1])
            if l <= 0 or r <= 0: continue
            if not self.include_background: l -= 1; r -= 1
            if 0 <= l < self.num_classes and 0 <= r < self.num_classes: pairs.append((l, r))
        self.lr_pairs = pairs

    def compute_adjacency(self, labels: torch.Tensor, brain_mask: torch.Tensor) -> torch.Tensor:
        adj = torch.zeros((self.num_classes, self.num_classes), device=self.device)
        eff = torch.where(brain_mask, labels, torch.full_like(labels, -1))
        if not self.include_background: eff = torch.where(eff <= 0, torch.full_like(eff, -1), eff)
        for shift in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            rolled = torch.roll(eff, shifts=shift, dims=(0, 1, 2))
            valid = (eff >= 0) & (rolled >= 0) & (eff != rolled)
            if not torch.any(valid): continue
            pairs = torch.stack([eff[valid].flatten().long(), rolled[valid].flatten().long()], dim=1)
            pairs = torch.unique(pairs, dim=0)
            adj[pairs[:, 0], pairs[:, 1]] = 1
            adj[pairs[:, 1], pairs[:, 0]] = 1
        return adj - torch.diag_embed(torch.diagonal(adj))

    def compute_spectral_distance(self, adj: torch.Tensor, age: float, top_k: int = 20) -> Optional[float]:
        if self.adj_templates is None: return None
        idx = torch.searchsorted(self.adj_age_values, torch.tensor(float(age), device=self.device)).clamp(0,
                                                                                                          len(self.adj_age_values) - 1)
        prior = self.adj_templates[idx]

        def _lap(a):
            deg = torch.diag_embed(a.sum(dim=-1))
            return deg - a

        try:
            e1 = torch.linalg.eigvalsh(_lap(adj.float()))
            e2 = torch.linalg.eigvalsh(_lap(prior.float()))
            k = min(top_k, len(e1), len(e2))
            return float(torch.mean((e1[:k] - e2[:k]) ** 2).item())
        except:
            return None

    def compute_structural_violations(self, adj: torch.Tensor) -> Dict[str, int]:
        return {
            "forbidden": sum(1 for i, j in self.forbidden_edges if adj[i, j] > 0),
            "required": sum(1 for i, j in self.required_edges if adj[i, j] == 0)
        }

    def compute_symmetry(self, labels: torch.Tensor, brain_mask: torch.Tensor) -> float:
        if not self.lr_pairs: return 0.0
        eff = torch.where(brain_mask, labels, torch.full_like(labels, -1))
        if not self.include_background: eff = torch.where(eff <= 0, torch.full_like(eff, -1), eff)
        scores = []
        for l, r in self.lr_pairs:
            vl, vr = torch.count_nonzero(eff == l).item(), torch.count_nonzero(eff == r).item()
            if vl + vr > 0: scores.append(1.0 - abs(vl - vr) / (vl + vr))
        return sum(scores) / len(scores) if scores else 0.0

    def compute_rve(self, pred: torch.Tensor, target: torch.Tensor, brain_mask: torch.Tensor) -> float:
        eff_p = torch.where(brain_mask, pred, torch.full_like(pred, -1))
        eff_t = torch.where(brain_mask, target, torch.full_like(target, -1))
        if not self.include_background:
            eff_p = torch.where(eff_p <= 0, torch.full_like(eff_p, -1), eff_p)
            eff_t = torch.where(eff_t <= 0, torch.full_like(eff_t, -1), eff_t)
        classes = torch.unique(eff_t)
        classes = classes[classes >= 0]
        errors = []
        for c in classes.tolist():
            vp, vt = torch.count_nonzero(eff_p == c).item(), torch.count_nonzero(eff_t == c).item()
            if vt > 0: errors.append(abs(vp - vt) / vt)
        return sum(errors) / len(errors) if errors else 0.0


# --- Helpers ---

def _get_inference_affine(images: torch.Tensor) -> np.ndarray:
    if isinstance(images, MetaTensor):
        return images.affine[0].detach().cpu().numpy()
    elif hasattr(images, "affine"):
        aff = images.affine
        if isinstance(aff, torch.Tensor):
            return aff[0].detach().cpu().numpy()
        return np.array(aff[0])
    return np.eye(4)


def _coerce_to_str_path(value) -> Optional[str]:
    if isinstance(value, (list, tuple)): return str(value[0]) if value else None
    return str(value) if value is not None else None


def _prepare_pred_volume(pred_labels: torch.Tensor, brain_mask: Optional[torch.Tensor],
                         foreground_only: bool) -> np.ndarray:
    vol = pred_labels.squeeze(0).detach().cpu().numpy().astype(np.int16)
    if brain_mask is not None:
        mask = brain_mask.squeeze(0).detach().cpu().numpy().astype(bool)
    else:
        mask = np.ones_like(vol, dtype=bool)
    final_vol = np.zeros_like(vol, dtype=np.int16)
    if foreground_only:
        final_vol[mask] = vol[mask] + 1
    else:
        final_vol[mask] = vol[mask]
    return final_vol


def _save_prediction_aligned(pred_volume: np.ndarray, inference_affine: np.ndarray, meta_dict: Dict, output_dir: Path,
                             case_id: str, resample_to_native: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    src_img = nib.Nifti1Image(pred_volume, inference_affine)
    final_img = src_img
    suffix = "_inf"
    if resample_to_native:
        filename = _coerce_to_str_path(meta_dict.get("filename_or_obj"))
        if filename and os.path.exists(filename):
            try:
                target_img = nib.load(filename)
                resampled = processing.resample_from_to(src_img, target_img, order=0)
                final_img = resampled
                suffix = ""
            except Exception as e:
                print(f"Error resampling {case_id}: {e}")
    out_path = output_dir / f"{case_id}_pred{suffix}.nii.gz"
    nib.save(final_img, str(out_path))


def _save_gap_map(probs: torch.Tensor, labels: torch.Tensor, brain_mask: Optional[torch.Tensor], affine: np.ndarray,
                  output_dir: Path, case_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_idx = labels.clone().long().clamp(min=0, max=probs.shape[1] - 1)
    gt_probs = torch.gather(probs, 1, target_idx)
    gap = (1.0 - gt_probs)
    if brain_mask is not None: gap = gap * brain_mask
    nib.save(nib.Nifti1Image(gap.squeeze().detach().cpu().numpy(), affine), str(output_dir / f"{case_id}_gap.nii.gz"))


def _compute_case_dice_scalar(preds_oh, labels_oh, mask_ex):
    inter = (preds_oh * labels_oh * mask_ex).sum(dim=(2, 3, 4))
    union = (preds_oh * mask_ex).sum(dim=(2, 3, 4)) + (labels_oh * mask_ex).sum(dim=(2, 3, 4))
    d = 2 * inter / (union + 1e-6)
    return float(d.mean().item())


# --- Main Loop ---

def evaluate(args):
    print("Running TRM Evaluation V7 (Robust Fix)")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_inference_loader(args)
    model = SwinUNETRWrapper(args).to(device)
    if os.path.isfile(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=True)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")
    model.eval()

    # --- Metrics Setup ---
    # DiceMetric with mean_batch accumulates [C] size tensor
    dice_metric = DiceMetric(include_background=not args.foreground_only, reduction="mean_batch")
    per_class_metric = DiceMetric(include_background=not args.foreground_only, reduction="none", get_not_nans=True)

    hd95_metric = HausdorffDistanceMetric(include_background=not args.foreground_only, percentile=95, reduction="mean")
    assd_metric = SurfaceDistanceMetric(include_background=not args.foreground_only, symmetric=True, reduction="mean")

    adv = AdvancedMetrics(args.out_channels, include_background=not args.foreground_only,
                          adjacency_prior=args.adjacency_prior, structural_rules=args.structural_rules,
                          laterality_pairs=args.laterality_pairs_json, device=device)

    predictions_dir = Path(args.output_dir)
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            start_time = time.time()
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Meta extraction
            raw_meta = batch.get("image_meta_dict")
            meta_dict = {}
            if isinstance(raw_meta, dict):
                for k, v in raw_meta.items():
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        meta_dict[k] = v[0]
                    elif isinstance(v, torch.Tensor) and v.shape[0] > 0:
                        meta_dict[k] = v[0].item() if v.numel() == 1 else v[0]
                    else:
                        meta_dict[k] = v
            elif isinstance(raw_meta, list):
                meta_dict = raw_meta[0]

            fname = _coerce_to_str_path(meta_dict.get("filename_or_obj"))
            case_id = Path(fname).stem.replace(".nii", "") if fname else f"case_{batch_idx}"
            inference_affine = _get_inference_affine(images)

            # Inference
            brain_mask = (labels > 0)
            if brain_mask.shape[1] == 1: brain_mask = brain_mask.squeeze(1)

            with autocast(enabled=args.use_amp):
                logits = sliding_window_inference(images, (args.roi_x, args.roi_y, args.roi_z), args.sw_batch_size,
                                                  model, args.sw_overlap)

            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            pred = torch.where(brain_mask, pred, torch.zeros_like(pred))

            # Metrics input
            pred_oh = F.one_hot(pred, args.out_channels).permute(0, 4, 1, 2, 3).float()
            labels_sq = labels.squeeze(1).long().clamp(min=0)
            labels_oh = F.one_hot(labels_sq, args.out_channels).permute(0, 4, 1, 2, 3).float()
            mask_ex = brain_mask.unsqueeze(1)

            pred_eval = pred_oh * mask_ex
            target_eval = labels_oh * mask_ex

            # 1. Accumulate Global Metrics
            dice_metric(y_pred=pred_eval, y=target_eval)
            per_class_metric(y_pred=pred_eval, y=target_eval)

            # 2. Per-Case Scalar Dice (Manual)
            if args.foreground_only:
                batch_dice = _compute_case_dice_scalar(pred_eval[:, 1:], target_eval[:, 1:], mask_ex)
            else:
                batch_dice = _compute_case_dice_scalar(pred_eval, target_eval, mask_ex)

            # 3. Heavy Metrics
            hd95_metric.reset()
            assd_metric.reset()
            hd95_metric(y_pred=pred_eval, y=target_eval)
            assd_metric(y_pred=pred_eval, y=target_eval)

            try:
                val_hd = hd95_metric.aggregate()
                batch_hd95 = float(val_hd.item()) if isinstance(val_hd, torch.Tensor) else float(val_hd)
            except:
                batch_hd95 = 0.0
            try:
                val_assd = assd_metric.aggregate()
                batch_assd = float(val_assd.item()) if isinstance(val_assd, torch.Tensor) else float(val_assd)
            except:
                batch_assd = 0.0

            # 4. Advanced Metrics
            adj = adv.compute_adjacency(pred[0], brain_mask[0])
            viol = adv.compute_structural_violations(adj)
            rve = adv.compute_rve(pred[0], labels_sq[0], brain_mask[0])
            sym = adv.compute_symmetry(pred[0], brain_mask[0])

            age_val = 40.0
            if "age" in meta_dict: age_val = float(meta_dict["age"])
            spec = adv.compute_spectral_distance(adj, age_val)

            # 5. Skeleton Metrics
            pred_np = pred[0].detach().cpu().numpy()
            target_np = labels_sq[0].detach().cpu().numpy()
            cldice_sum, cbdice_sum, valid_cnt = 0.0, 0.0, 0
            for c in range(1, args.out_channels):
                p_c = (pred_np == c)
                t_c = (target_np == c)
                if np.sum(t_c) > 0:
                    cldice_sum += compute_cldice(p_c, t_c)
                    cbdice_sum += compute_cbdice(p_c, t_c)
                    valid_cnt += 1
            batch_cldice = cldice_sum / valid_cnt if valid_cnt > 0 else 0.0
            batch_cbdice = cbdice_sum / valid_cnt if valid_cnt > 0 else 0.0

            labels_clce = torch.where(mask_ex, labels.squeeze(1), torch.zeros_like(labels.squeeze(1)))
            batch_clce = compute_clce(logits, labels_clce)

            # Saving
            pred_vol = _prepare_pred_volume(pred, brain_mask, args.foreground_only)
            _save_prediction_aligned(pred_vol, inference_affine, meta_dict, predictions_dir, case_id,
                                     args.resample_to_native)
            _save_gap_map(probs, labels, mask_ex, inference_affine, predictions_dir, case_id)

            elapsed = time.time() - start_time
            print(f"[{batch_idx}] {case_id}: Dice={batch_dice:.4f}, HD95={batch_hd95:.2f}, RVE={rve:.4f}, Viol={viol}")

            results.append({
                "case_id": case_id,
                "dice": batch_dice,
                "hd95": batch_hd95,
                "assd": batch_assd,
                "rve": rve,
                "symmetry": sym,
                "spec_distance": spec,
                "cldice": batch_cldice,
                "cbdice": batch_cbdice,
                "clce": batch_clce,
                "violation_forbidden": viol["forbidden"],
                "violation_required": viol["required"]
            })

    # --- Final Aggregation (Safe) ---
    print("\nCalculating Final Aggregated Metrics...")

    # 1. Mean Dice (Global)
    mean_dice = 0.0
    try:
        dm_agg = dice_metric.aggregate()
        if not isinstance(dm_agg, torch.Tensor):
            dm_agg = torch.as_tensor(dm_agg)

        # dm_agg is Tensor(C) or Tensor(1, C)
        if dm_agg.numel() > 1:
            if hasattr(torch, "nanmean"):
                mean_dice = torch.nanmean(dm_agg).item()
            else:
                # Manual nanmean
                valid_mask = ~torch.isnan(dm_agg)
                if valid_mask.sum() > 0:
                    mean_dice = dm_agg[valid_mask].mean().item()
        else:
            mean_dice = dm_agg.item()

    except Exception as e:
        print(f"WARNING: Global Dice Metric Aggregation failed: {e}")
        print("Fallback: Averaging case-level Mean Dice.")
        mean_dice = np.mean([r['dice'] for r in results])

    # 2. Per-Class Dice
    per_class_scores = []
    per_class_counts = []
    try:
        agg_per_class = per_class_metric.aggregate()
        if isinstance(agg_per_class, (tuple, list)) and len(agg_per_class) == 2:
            scores, counts = agg_per_class
            # scores, counts are tensors
            if scores.ndim > 1:
                final_scores = scores.mean(dim=0)
                final_counts = counts.sum(dim=0)
                per_class_scores = final_scores.detach().cpu().tolist()
                per_class_counts = final_counts.detach().cpu().tolist()
    except Exception as e:
        print(f"WARNING: Per-Class Metric Aggregation failed: {e}")

    # 3. Scalar Summaries
    keys = ["dice", "hd95", "assd", "rve", "symmetry", "cldice", "cbdice", "clce", "spec_distance"]
    summary = {f"mean_{k}": np.mean([r[k] for r in results if r[k] is not None]) for k in keys}

    # Override/Update global mean dice logic
    summary["metric_mean_dice_global"] = mean_dice

    # 4. Total Violations
    total_forbidden = sum(r["violation_forbidden"] for r in results)
    total_required = sum(r["violation_required"] for r in results)
    summary["total_violations_forbidden"] = total_forbidden
    summary["total_violations_required"] = total_required

    # 5. Add Per-Class
    summary["per_class_dice"] = per_class_scores
    summary["per_class_valid_counts"] = per_class_counts

    print("\n" + "=" * 40)
    print("FINAL RESULTS:")
    print(f"  Mean Dice (Global): {mean_dice:.4f}")
    print(f"  Mean Dice (Avg of Cases): {summary['mean_dice']:.4f}")
    print(f"  Mean HD95: {summary['mean_hd95']:.4f}")
    print(f"  Mean RVE:  {summary['mean_rve']:.4f}")
    print(f"  Total Viol: F={total_forbidden}, R={total_required}")
    print("=" * 40)

    out_json = summary.copy()
    out_json["cases"] = results

    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_path, "w") as f:
        json.dump(out_json, f, indent=2)


def main():
    global DEBUG
    args = parse_args()
    DEBUG = args.debug_mode
    evaluate(args)


if __name__ == "__main__":
    main()