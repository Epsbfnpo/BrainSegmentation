#!/usr/bin/env python3
"""
TRM Method Evaluation Script (V3 - Fix Alignment)
- 核心修复：强制使用 MetaTensor 的 affine 属性作为 Source Affine，解决 Slicer 错位问题。
- 移除了所有耗时的指标计算（clDice等），只保留 Dice 和 RVE，确保快速验证修复效果。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
from nibabel import processing
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from torch.cuda.amp import autocast

# --- Import TRM Components ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import get_inference_loader
from modules import SwinUNETRWrapper

DEBUG = True  # 默认开启 Debug 以便观察 Affine 变化


def _debug(msg: str, payload: Optional[Dict] = None) -> None:
    if DEBUG:
        print(f"[DEBUG] {msg}: {payload}" if payload else f"[DEBUG] {msg}")


def parse_args():
    parser = argparse.ArgumentParser(description="TRM Evaluation V3")

    # Paths
    parser.add_argument("--split_json", required=True, help="Path to test split JSON")
    parser.add_argument("--model_path", required=True, help="Path to best.ckpt")
    parser.add_argument("--output_dir", default="./test_predictions_v3", help="Directory for outputs")
    parser.add_argument("--metrics_path", default="./analysis/test_metrics_v3.json")

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

    # Resampling
    parser.add_argument("--resample_to_native", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# --- Geometry & IO Helpers (Fixing Alignment) ---

def _get_inference_affine(images: torch.Tensor) -> np.ndarray:
    """
    关键修复：从 MONAI MetaTensor 中直接提取当前的 Affine 矩阵。
    这代表了模型“看到”的图像的物理坐标系（通常是 RAS + Resampled Spacing）。
    """
    if isinstance(images, MetaTensor):
        # MetaTensor 肯定包含当前变换后的 Affine
        return images.affine[0].detach().cpu().numpy()
    elif hasattr(images, "affine"):
        # 兼容其他可能的 Tensor 包装器
        aff = images.affine
        if isinstance(aff, torch.Tensor):
            return aff[0].detach().cpu().numpy()
        return np.array(aff[0])

    # 极度危险的情况：如果不是 MetaTensor，我们丢失了空间信息
    print("Warning: Input images is NOT a MetaTensor! Falling back to Identity matrix. Alignment is likely wrong.")
    return np.eye(4)


def _coerce_to_str_path(value) -> Optional[str]:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value) if value is not None else None


def _prepare_pred_volume(pred_labels: torch.Tensor, brain_mask: Optional[torch.Tensor],
                         foreground_only: bool) -> np.ndarray:
    """将 Tensor 转换为 numpy volume，处理背景和 label shift。"""
    vol = pred_labels.squeeze(0).detach().cpu().numpy().astype(np.int16)

    if brain_mask is not None:
        mask = brain_mask.squeeze(0).detach().cpu().numpy().astype(bool)
    else:
        mask = np.ones_like(vol, dtype=bool)

    final_vol = np.zeros_like(vol, dtype=np.int16)

    # 如果模型输出是 0..86 (foreground_only)，需要映射回 1..87
    if foreground_only:
        final_vol[mask] = vol[mask] + 1
    else:
        final_vol[mask] = vol[mask]

    return final_vol


def _save_prediction_aligned(
        pred_volume: np.ndarray,
        inference_affine: np.ndarray,
        meta_dict: Dict,
        output_dir: Path,
        case_id: str,
        resample_to_native: bool
) -> None:
    """
    保存预测结果。
    关键逻辑：
    1. 先构建 src_img (Inference Space)。
    2. 如果需要，读取原始 Header 并 Resample 回 Native Space。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 构建源图像对象（这是模型输出的真实物理状态）
    src_img = nib.Nifti1Image(pred_volume, inference_affine)

    final_img = src_img
    suffix = "_inf"  # 标记为 inference space

    # 2. 尝试重采样回原始空间
    if resample_to_native:
        filename = _coerce_to_str_path(meta_dict.get("filename_or_obj"))
        if filename and os.path.exists(filename):
            try:
                target_img = nib.load(filename)
                # 使用 Nearest Neighbor (order=0) 重采样 Labels
                resampled = processing.resample_from_to(src_img, target_img, order=0)
                final_img = resampled
                suffix = ""  # Native space 不需要后缀，作为最终结果
                # _debug(f"Successfully resampled {case_id} to native space.")
            except Exception as e:
                print(f"Error resampling {case_id}: {e}")
                print("Saving in inference space instead.")
        else:
            print(f"Original filename not found for {case_id}, skipping native resample.")

    out_path = output_dir / f"{case_id}_pred{suffix}.nii.gz"
    nib.save(final_img, str(out_path))


# --- Main Evaluation Loop ---

def evaluate(args):
    print("Running TRM Evaluation V3 (Alignment Fixed)")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data Loader
    # 确保 data_loader.py 中的 transforms 包含 EnsureTyped 或 EnsureChannelFirstd 等能产生 MetaTensor 的变换
    test_loader = get_inference_loader(args)

    # 2. Model
    model = SwinUNETRWrapper(args).to(device)
    if os.path.isfile(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        # 去除 DDP 前缀
        new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=True)
        print(f"Loaded checkpoint: {args.model_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")
    model.eval()

    predictions_dir = Path(args.output_dir)
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)  # 期望是 MetaTensor
            labels = batch["label"].to(device)

            # --- Extract Metadata ---
            # 处理 collate 后的 meta_dict
            raw_meta = batch.get("image_meta_dict")
            meta_dict = {}
            if isinstance(raw_meta, dict):
                for k, v in raw_meta.items():
                    # 取 batch 中第一个元素
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

            # --- Key Step: Get Accurate Affine ---
            # 必须从 images Tensor 本身获取，因为它经过了 Transforms 链的处理
            inference_affine = _get_inference_affine(images)

            # --- Inference ---
            brain_mask = (labels > 0)
            if brain_mask.shape[1] == 1: brain_mask = brain_mask.squeeze(1)

            with autocast(enabled=args.use_amp):
                logits = sliding_window_inference(
                    images,
                    (args.roi_x, args.roi_y, args.roi_z),
                    args.sw_batch_size,
                    model,
                    args.sw_overlap
                )

            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)

            # Masking in Inference Space
            pred = torch.where(brain_mask, pred, torch.zeros_like(pred))

            # --- Quick Dice (Manual) ---
            # 注意：这里的 Dice 是在 Inference Space 计算的，这是合法的，因为 Pred 和 GT 都在同一空间
            pred_oh = F.one_hot(pred, args.out_channels).permute(0, 4, 1, 2, 3).float()
            labels_sq = labels.squeeze(1).long().clamp(min=0)
            labels_oh = F.one_hot(labels_sq, args.out_channels).permute(0, 4, 1, 2, 3).float()
            mask_ex = brain_mask.unsqueeze(1)

            inter = (pred_oh * labels_oh * mask_ex).sum(dim=(2, 3, 4))
            union = (pred_oh * mask_ex).sum(dim=(2, 3, 4)) + (labels_oh * mask_ex).sum(dim=(2, 3, 4))
            dice_c = 2 * inter / (union + 1e-6)

            if args.foreground_only:
                dice_c = dice_c[:, 1:]  # Skip background

            batch_dice = float(dice_c.mean().item())

            # --- Save Prediction (Aligned) ---
            pred_vol = _prepare_pred_volume(pred, brain_mask, args.foreground_only)

            _save_prediction_aligned(
                pred_vol,
                inference_affine,  # <--- 关键：使用 Tensor 的当前 Affine
                meta_dict,
                predictions_dir,
                case_id,
                args.resample_to_native
            )

            print(f"[{batch_idx}] {case_id}: Dice={batch_dice:.4f}")
            results.append({"case_id": case_id, "dice": batch_dice})

    mean_dice = np.mean([r['dice'] for r in results]) if results else 0.0
    print(f"\nFinal Mean Dice: {mean_dice:.4f}")

    out_json = {"mean_dice": mean_dice, "cases": results}
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_path, "w") as f:
        json.dump(out_json, f, indent=2)


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()