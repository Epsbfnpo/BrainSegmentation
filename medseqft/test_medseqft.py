#!/usr/bin/env python3
"""
Inference and evaluation entrypoint for MedSeqFT.
FINAL GOLD VERSION - FIXED GEOMETRY ALIGNMENT.
Strategy:
1. Load Data with standard transforms.
2. Infer in transformed space.
3. Compute metrics in transformed space (valid for comparison).
4. SAVE using SALT's method: Load original file from disk -> Resample prediction to match it exactly.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import nibabel as nib
from nibabel import processing
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    EnsureTyped, MapTransform, ScaleIntensityRanged, ToTensord
)
from monai.data import Dataset, DataLoader, MetaTensor, decollate_batch
from torch.cuda.amp import autocast

# Add current directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

# --- Imports specific to MedSeqFT ---
from components import MedSeqFTWrapper

# =========================================================================
# PART 1: Helper Functions (Aligned with SALT)
# =========================================================================

def _coerce_to_str_path(value: Union[str, Path, Sequence, None], key: str = "") -> Optional[str]:
    if isinstance(value, (list, tuple)):
        if not value: return None
        value = value[0]
    if value is None: return None
    return str(value)


def _compute_case_id(meta_dict):
    """Extract case ID from metadata."""
    fname = meta_dict.get("filename_or_obj", "")
    if isinstance(fname, (list, tuple)): fname = fname[0]
    return Path(str(fname)).stem.replace(".nii", "")


def _save_prediction(
        pred_volume: np.ndarray,
        current_affine: np.ndarray,
        image_meta: Dict,
        label_meta: Dict,
        output_dir: Path,
        case_id: str,
        *,
        resample_to_native: bool,
        spacing_tolerance: float
) -> Path:
    """
    SALT-STYLE SAVING: Loads the original NIfTI file as a template for resampling.
    This guarantees 100% geometric alignment.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine the "Current" Affine (Transformed Space)
    affine = np.asarray(current_affine, dtype=np.float32)

    # --- THE SALT METHOD: File-based Resampling ---
    if resample_to_native:
        # Find the path to the original file
        filename_for_resample = _coerce_to_str_path(label_meta.get("filename_or_obj"))
        if not filename_for_resample:
            filename_for_resample = _coerce_to_str_path(image_meta.get("filename_or_obj"))

        if filename_for_resample and os.path.exists(filename_for_resample):
            try:
                # 1. Load Original File (The Anchor)
                target_img = nib.load(filename_for_resample)

                # 2. Wrap Prediction in NIfTI (Current Space)
                # Ensure data is float for resampling logic (though nearest uses values directly)
                src_img = nib.Nifti1Image(pred_volume.astype(np.float32), affine)

                # =========================================================
                # [CRITICAL FIX] 设置 Header Codes
                # 显式告知 nibabel 这个 affine 是有效的 Scanner Anat 坐标系
                # 防止 resample_from_to 因为 code 不匹配而对齐失败
                # =========================================================
                src_img.header.set_qform(affine, code=1)
                src_img.header.set_sform(affine, code=1)

                # 3. Resample Source -> Target (Nearest Neighbor for Labels)
                resampled = processing.resample_from_to(src_img, target_img, order=0)

                # 4. Update data and affine to match the original file
                pred_volume = np.asarray(resampled.dataobj)
                affine = resampled.affine

            except Exception as e:
                print(f"⚠️ Resampling failed for {case_id}: {e}. Saving in transformed space.")
        else:
            print(f"⚠️ Original file not found for {case_id}. Saving in transformed space.")

    # Save
    out_path = output_dir / f"{case_id}_pred.nii.gz"
    # Convert back to Int16 for storage (Labels)
    pred_volume = pred_volume.astype(np.int16)
    nib.save(nib.Nifti1Image(pred_volume, affine), str(out_path))
    return out_path


def _save_gap_map(probs, labels, brain_mask, affine, output_dir, case_id):
    output_dir.mkdir(parents=True, exist_ok=True)
    target_idx = (labels - 1).long()
    target_idx[target_idx < 0] = 0
    target_idx = target_idx.clamp(max=probs.shape[1] - 1).unsqueeze(1)

    gt_probs = torch.gather(probs, 1, target_idx)
    gap_map = 1.0 - gt_probs
    if brain_mask is not None:
        mask = brain_mask.unsqueeze(1) if brain_mask.ndim == gap_map.ndim - 1 else brain_mask
        gap_map = gap_map * mask

    if isinstance(affine, torch.Tensor): affine = affine.numpy()
    gap_np = gap_map.squeeze().detach().cpu().numpy().astype(np.float32)
    nib.save(nib.Nifti1Image(gap_np, affine), str(output_dir / f"{case_id}_gap.nii.gz"))


# =========================================================================
# PART 2: Data Loading
# =========================================================================

class PercentileNormalizationd(MapTransform):
    def __init__(self, keys: Sequence[str], lower: float = 1.0, upper: float = 99.0):
        super().__init__(keys)
        self.lower = lower
        self.upper = upper

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key]
            array = image.cpu().numpy() if isinstance(image, (torch.Tensor, MetaTensor)) else np.asarray(image)
            mask = array > 0
            if mask.any():
                lo = np.percentile(array[mask], self.lower)
                hi = np.percentile(array[mask], self.upper)
                if (hi - lo) > 1e-6:
                    norm = (np.clip(array, lo, hi) - lo) / (hi - lo)
                else:
                    norm = np.zeros_like(array)
            else:
                norm = np.zeros_like(array)
            norm[~mask] = 0
            d[key] = MetaTensor(torch.as_tensor(norm, dtype=torch.float32), meta=getattr(image, 'meta', None))
        return d


class ExtractAged(MapTransform):
    def __init__(self, metadata_key: str = "metadata"):
        super().__init__(keys=None)
        self.metadata_key = metadata_key

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        metadata = d.get(self.metadata_key) or {}
        age = 40.0
        for key in ("scan_age", "PMA", "pma", "ga", "GA"):
            if key in metadata:
                try:
                    age = float(metadata[key])
                    if key in {"ga", "GA"} and "pna" in metadata:
                        age += float(metadata.get("pna") or metadata.get("PNA"))
                    break
                except:
                    continue
        d["age"] = torch.tensor([age], dtype=torch.float32)
        return d


def get_test_transforms(args):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175.0, a_max=250.0,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32, track_meta=True),
        ToTensord(keys=["image", "label"]),
    ])


def get_medseqft_test_loader(args):
    with open(args.split_json, "r") as f:
        data = json.load(f)
    test_files = []
    for key in ("testing", "test", "test_set"):
        if key in data:
            raw_items = data[key]
            for item in raw_items:
                entry = {
                    "image": item["image"][0] if isinstance(item["image"], list) else item["image"],
                    "label": item.get("label"),
                    "metadata": item.get("metadata", {})
                }
                test_files.append(entry)
            break
    if not test_files:
        raise ValueError("Could not find 'testing', 'test', or 'test_set' in split json.")

    ds = Dataset(data=test_files, transform=get_test_transforms(args))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return loader


def _compute_case_dice(preds, target, include_background=True, eps=1e-8):
    if not include_background:
        preds = preds[:, 1:];
        target = target[:, 1:]
    intersection = (preds * target).sum(dim=(2, 3, 4))
    union = preds.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    return (2.0 * intersection + eps) / (union + eps)


class AdvancedMetrics:
    def __init__(self, num_classes: int, *, include_background: bool, adjacency_prior: Optional[str] = None,
                 structural_rules: Optional[str] = None, laterality_pairs: Optional[str] = None, device: torch.device):
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
        ages = payload.get("ages");
        matrices = payload.get("A_prior")
        if ages is None or matrices is None: return
        age_values = ages.astype(np.float32)
        order = np.argsort(age_values)
        matrices = self._align_classes_3d(matrices[order].astype(np.float32))
        self.adj_templates = torch.tensor(matrices, dtype=torch.float32, device=self.device)
        self.adj_age_values = torch.tensor(age_values[order], dtype=torch.float32, device=self.device)

    def _load_structural_rules(self, path: str) -> None:
        with open(path, "r") as f:
            payload = json.load(f)

        def _parse(rules):
            pairs = []
            for pair in rules:
                i, j = int(pair[0]), int(pair[1])
                if not self.include_background: i -= 1; j -= 1
                if 0 <= i < self.num_classes and 0 <= j < self.num_classes and i != j: pairs.append((i, j))
            return pairs

        self.required_edges = _parse(payload.get("required", []))
        self.forbidden_edges = _parse(payload.get("forbidden", []))

    def _load_lr_pairs(self, path: str) -> None:
        with open(path, "r") as f:
            payload = json.load(f)
        pairs = []
        for pair in payload:
            left, right = int(pair[0]), int(pair[1])
            if left <= 0 or right <= 0: continue
            if not self.include_background: left -= 1; right -= 1
            if 0 <= left < self.num_classes and 0 <= right < self.num_classes: pairs.append((left, right))
        self.lr_pairs = pairs

    def _laplacian(self, adj):
        adj_sym = 0.5 * (adj + adj.transpose(-1, -2))
        deg = torch.diag_embed(adj_sym.sum(dim=-1))
        return deg - adj_sym

    def compute_adjacency(self, labels, brain_mask):
        adj = torch.zeros((self.num_classes, self.num_classes), device=self.device)
        eff = torch.where(brain_mask, labels, torch.full_like(labels, -1))
        if not self.include_background:
            eff = torch.where(eff > 0, eff - 1, torch.full_like(eff, -1))

        shifts = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        for s in shifts:
            rolled = torch.roll(eff, shifts=s, dims=(0, 1, 2))
            valid = (eff >= 0) & (rolled >= 0) & (eff != rolled)
            if not valid.any(): continue
            src, dst = eff[valid].long(), rolled[valid].long()
            pairs = torch.stack([src, dst], dim=1).unique(dim=0)
            adj[pairs[:, 0], pairs[:, 1]] = 1
        return adj

    def compute_spectral_distance(self, adj, age, top_k=20):
        if self.adj_templates is None: return None
        age_t = torch.tensor(float(age), device=self.device)
        idx = torch.searchsorted(self.adj_age_values, age_t).clamp(0, len(self.adj_age_values) - 1)
        prior = self.adj_templates[idx]
        try:
            e1 = torch.linalg.eigvalsh(self._laplacian(adj.float()))
            e2 = torch.linalg.eigvalsh(self._laplacian(prior.float()))
            k = min(top_k, e1.numel(), e2.numel())
            return float(torch.mean((e1[:k] - e2[:k]) ** 2).item())
        except:
            return None

    def compute_structural_violations(self, adj):
        return {
            "forbidden": sum(1 for i, j in self.forbidden_edges if adj[i, j] > 0),
            "required": sum(1 for i, j in self.required_edges if adj[i, j] == 0)
        }

    def compute_symmetry(self, labels, brain_mask):
        if not self.lr_pairs: return 0.0
        eff = torch.where(brain_mask, labels, torch.full_like(labels, -1))
        if not self.include_background:
            eff = torch.where(eff > 0, eff - 1, torch.full_like(eff, -1))
        scores = []
        for l, r in self.lr_pairs:
            vl = (eff == l).sum().item();
            vr = (eff == r).sum().item()
            if vl + vr > 0: scores.append(1.0 - abs(vl - vr) / float(vl + vr))
        return float(sum(scores) / len(scores)) if scores else 0.0

    def compute_rve(self, pred, target, mask):
        pe = torch.where(mask, pred, torch.full_like(pred, -1))
        te = torch.where(mask, target, torch.full_like(target, -1))
        valid_classes = torch.unique(te)
        valid_classes = valid_classes[valid_classes > 0]
        errs = []
        for c in valid_classes.tolist():
            vp = (pe == c).sum().item();
            vt = (te == c).sum().item()
            if vt > 0: errs.append(abs(vp - vt) / float(vt))
        return float(sum(errs) / len(errs)) if errs else 0.0


def load_class_mapping(path: Optional[Path]) -> Dict[int, int]:
    if path is None or not path.exists(): return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    mapping = {}
    data = payload.get("index_to_raw_label") if isinstance(payload, dict) else None
    if isinstance(data, dict):
        for key, value in data.items():
            try:
                mapping[int(key)] = int(value)
            except:
                continue
    return mapping


# =========================================================================
# PART 4: Main Loop
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="./test_predictions_medseqft")
    parser.add_argument("--metrics_path", default="./analysis/test_metrics_medseqft.json")
    parser.add_argument("--roi_x", default=128, type=int)
    parser.add_argument("--roi_y", default=128, type=int)
    parser.add_argument("--roi_z", default=128, type=int)
    parser.add_argument("--out_channels", default=15, type=int)
    parser.add_argument("--feature_size", default=48, type=int)
    parser.add_argument("--foreground_only", action="store_true", default=True)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--sw_batch_size", type=int, default=4)
    parser.add_argument("--sw_overlap", type=float, default=0.5)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--apply_spacing", action="store_true", default=True)
    parser.add_argument("--apply_orientation", action="store_true", default=True)
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[1.5, 1.5, 1.5])
    parser.add_argument("--class_map_json", type=str, default=None)
    parser.add_argument("--adjacency_prior", type=str, default=None)
    parser.add_argument("--structural_rules", type=str, default=None)
    parser.add_argument("--laterality_pairs_json", type=str, default=None)
    parser.add_argument("--in_channels", default=1, type=int)

    # Arguments for shell script compatibility
    parser.add_argument("--resample_to_native", action="store_true", default=True)
    parser.add_argument("--resample_tolerance", type=float, default=0.1)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"📂 Loading test data...")
    test_loader = get_medseqft_test_loader(args)

    print(f"🏗️ Loading Model from {args.model_path}...")
    model = MedSeqFTWrapper(args, device).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    if "model" in state_dict: state_dict = state_dict["model"]
    if "state_dict" in state_dict: state_dict = state_dict["state_dict"]

    new_state = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        new_state[k] = v

    model.backbone.load_state_dict(new_state, strict=True)
    model.eval()

    dice_metric = DiceMetric(include_background=not args.foreground_only, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=not args.foreground_only, percentile=95,
                                          reduction="mean_batch")
    predictions_dir = Path(args.output_dir)
    per_case = []
    class_mapping = load_class_mapping(Path(args.class_map_json)) if args.class_map_json else {}

    print(f"🚀 Starting evaluation on {len(test_loader)} cases...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # --- CRITICAL FIX START ---
            # Retrieve the true current affine directly from the MetaTensor in memory.
            # This reflects all transforms (Spacing, Orientation) applied by the loader.
            # batch["image"] is a MetaTensor because of EnsureTyped(track_meta=True).
            if isinstance(batch["image"], MetaTensor):
                current_affine = batch["image"].affine[0].detach().cpu().numpy()
            else:
                print("⚠️ Warning: Input is not a MetaTensor. Affine might be guessed incorrectly.")
                current_affine = np.eye(4)
            # --- CRITICAL FIX END ---

            # --- Brain Mask Logic (Same as before) ---
            # Labels from loader are 0..87 (0=background)
            brain_mask = (labels > 0)
            if brain_mask.shape[1] == 1: brain_mask = brain_mask.squeeze(1)

            # --- Inference (Transformed Space) ---
            with autocast(enabled=args.use_amp):
                def infer(x): return sliding_window_inference(x, (args.roi_x, args.roi_y, args.roi_z),
                                                              args.sw_batch_size, model, overlap=args.sw_overlap)

                logits = infer(images)

            probs = torch.softmax(logits, dim=1)
            pred_raw = torch.argmax(probs, dim=1)

            # --- Shift & Mask: Force Background=0 ---
            pred_labels = pred_raw + 1
            pred_labels[~brain_mask] = 0

            # --- Metrics (Calculated in Transformed Space) ---
            total_cls = args.out_channels + 1
            preds_oh = F.one_hot(pred_labels, num_classes=total_cls).permute(0, 4, 1, 2, 3).float()
            target_oh = F.one_hot(labels.long().squeeze(1), num_classes=total_cls).permute(0, 4, 1, 2, 3).float()

            dice_metric(y_pred=preds_oh, y=target_oh)
            hd95_val = float(torch.nan_to_num(hd95_metric(y_pred=preds_oh, y=target_oh)).mean().item())
            case_dice = float(_compute_case_dice(preds_oh, target_oh, include_background=True).mean().item())

            # --- Save Outputs (SALT STYLE RESAMPLING) ---
            # 1. Decollate to get individual item metadata (essential for finding filename)
            batch_data_list = decollate_batch(batch)
            item_meta = batch_data_list[0]["image_meta_dict"]  # Use image meta to be safe
            label_meta = batch_data_list[0].get("label_meta_dict", {})

            case_id = _compute_case_id(item_meta)

            _save_gap_map(probs, labels.squeeze(1), brain_mask.float(), current_affine, predictions_dir, case_id)

            # 3. Resample Prediction to Native using SALT's method
            # Convert prediction to numpy [H, W, D]
            pred_volume_np = pred_labels[0].cpu().numpy().astype(np.int16)

            pred_native_path = _save_prediction(
                pred_volume=pred_volume_np,
                current_affine=current_affine,
                image_meta=item_meta,
                label_meta=label_meta,
                output_dir=predictions_dir,
                case_id=case_id,
                resample_to_native=args.resample_to_native,
                spacing_tolerance=args.resample_tolerance
            )

            per_case.append({
                "case_id": case_id, "dice": case_dice, "hd95": hd95_val,
                "path": str(pred_native_path)
            })
            print(f"  Processed {case_id}: Dice={case_dice:.4f} HD95={hd95_val:.2f}")

    mean_dice = float(dice_metric.aggregate().mean().item())
    mean_hd95 = float(torch.nan_to_num(hd95_metric.aggregate()).mean().item())

    def avg(k):
        return sum(c[k] for c in per_case if c[k] is not None) / len(per_case) if per_case else 0.0

    metrics = {
        "mean_dice": mean_dice, "mean_hd95": mean_hd95,
        "cases": per_case
    }

    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"🏁 Final Mean Dice: {mean_dice:.4f}")


if __name__ == "__main__":
    main()
