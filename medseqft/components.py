import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple
from monai.data import CacheDataset, DataLoader, Dataset, MetaTensor
from monai.transforms import (
    CenterSpatialCropd, Compose, CopyItemsd, EnsureChannelFirstd, EnsureTyped,
    LoadImaged, MapTransform, Orientationd, RandAdjustContrastd, RandBiasFieldd,
    RandCropByLabelClassesd, RandGaussianNoised, RandGaussianSmoothd,
    RandHistogramShiftd, RandRotated, RandScaleIntensityd, RandShiftIntensityd,
    RandZoomd, Spacingd, SpatialPadd, Randomizable
)
from monai.networks.nets import SwinUNETR


# 移除了 monai.losses 和 one_hot 依赖，完全手写以确保安全

# --- 1. Data Loader Components ---

class ExtractAged(MapTransform):
    def __init__(self, metadata_key: str = "metadata"):
        super().__init__(keys=None)
        self.metadata_key = metadata_key

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        metadata = d.get(self.metadata_key, {}) or {}
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
                norm = np.zeros_like(array)
                if hi > lo:
                    norm = (np.clip(array, lo, hi) - lo) / (hi - lo)
                norm[~mask] = 0
                d[key] = MetaTensor(torch.as_tensor(norm, dtype=torch.float32), meta=getattr(image, 'meta', None))
        return d


class RemapLabelsd(MapTransform):
    def __init__(self, keys: Sequence[str]):
        super().__init__(keys)

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            array = label.cpu().numpy() if isinstance(label, (torch.Tensor, MetaTensor)) else np.asarray(label)
            array = array.astype(np.int32)
            # Init with -1 (Ignore Index)
            remapped = np.full_like(array, -1, dtype=np.int32)
            mask = array > 0
            # Map 1->0, 87->86. (Based on user requirement: 0 is Class 1)
            remapped[mask] = array[mask] - 1
            d[key] = MetaTensor(torch.as_tensor(remapped, dtype=torch.float32), meta=getattr(label, 'meta', None))
        return d


def _get_transforms(args, mode="train"):
    spatial_size = (args.roi_x, args.roi_y, args.roi_z)
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ExtractAged(),
    ]
    if args.apply_spacing:
        transforms.append(Spacingd(keys=["image", "label"], pixdim=args.target_spacing, mode=("bilinear", "nearest")))
    if args.apply_orientation:
        transforms.append(Orientationd(keys=["image", "label"], axcodes="RAS"))
    transforms.append(PercentileNormalizationd(keys=["image"]))

    # 既然 Pretrain 也是 Remapping 的，我们必须启用它
    if args.foreground_only:
        transforms.append(RemapLabelsd(keys=["label"]))

    transforms.append(SpatialPadd(keys=["image", "label"], spatial_size=spatial_size, method="end"))

    if mode == "train":
        transforms.append(
            RandCropByLabelClassesd(
                keys=["image", "label"], label_key="label", spatial_size=spatial_size,
                num_classes=args.out_channels, num_samples=1, image_key="image", image_threshold=0
            )
        )
        transforms.extend([
            RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.35,
                        mode=["bilinear", "nearest"]),
            RandZoomd(keys=["image", "label"], prob=0.25, min_zoom=0.85, max_zoom=1.15, mode=["trilinear", "nearest"]),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
            RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.5, 1.0)),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.3), prob=0.3),
        ])
    else:
        transforms.append(CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size))

    transforms.append(EnsureTyped(keys=["image", "label", "age"], track_meta=False))
    return Compose(transforms)


def get_dataloaders(args):
    with open(args.split_json, "r") as f:
        data = json.load(f)

    def process(items):
        return [{"image": i["image"][0] if isinstance(i.get("image"), list) else i.get("image"),
                 "label": i.get("label"), "metadata": i.get("metadata")} for i in items]

    train_files = process(data.get("training", []))
    val_files = process(data.get("validation", []))

    train_ds = CacheDataset(data=train_files, transform=_get_transforms(args, "train"),
                            cache_rate=args.cache_rate, num_workers=args.num_workers) if args.cache_rate > 0 else \
        Dataset(data=train_files, transform=_get_transforms(args, "train"))

    val_ds = Dataset(data=val_files, transform=_get_transforms(args, "val"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader


# --- 2. Model Wrapper ---

class MedSeqFTWrapper(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.backbone = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=True,
        ).to(device)

    def forward(self, x):
        return self.backbone(x)

    def load_pretrained(self, path):
        print(f"📦 Loading weights from {path}")
        state = torch.load(path, map_location=next(self.parameters()).device)
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        self.backbone.load_state_dict(state, strict=False)


# --- 3. Loss Functions ---

class MedSeqFTLoss(nn.Module):
    """
    【自定义 Masked Loss】
    专门为 "Softmax 且无背景通道" 的场景设计。
    通过空间 Mask，强制忽略所有 Label=-1 的区域，防止 Loss 爆炸。
    """

    def __init__(self, num_classes: int, lambda_kd: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = -1
        self.lambda_kd = lambda_kd
        # 不再使用 Monai Loss，手写实现以确保 absolute control

    def forward(
            self, pred: torch.Tensor, target: torch.Tensor, teacher_pred: Optional[torch.Tensor],
            pred_kd: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # pred: [B, C, D, H, W] (Logits)
        # target: [B, 1, D, H, W] (Indices)

        # 1. 制作 Mask (剔除背景 -1)
        # 形状广播到 [B, C, D, H, W]
        valid_mask = (target != self.ignore_index)
        valid_mask_float = valid_mask.to(dtype=pred.dtype)

        # 2. 准备 Target One-Hot (Custom Scatter, No monai dependency)
        # 先把 -1 变成 0 以免 scatter 报错，反正后面会被 mask 掉
        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        # Scatter 到 One-Hot [B, C, D, H, W]
        target_onehot = torch.zeros_like(pred)
        # 必须 clamp 防止越界 (如果 out_channels 设置小了)
        target_indices = torch.clamp(target_safe.long(), max=self.num_classes - 1)
        target_onehot.scatter_(1, target_indices, 1.0)

        # 3. 计算 Logits 的 Softmax
        probs = F.softmax(pred, dim=1)

        # 4. 手动计算 Masked Dice Loss
        # 仅在 valid_mask 为 True 的区域计算
        # Intersection = Sum(Prob * GT * Mask)
        # Denominator = Sum(Prob * Mask) + Sum(GT * Mask)
        # 这样背景区域 (Mask=0) 的贡献会被完全抹除

        intersection = torch.sum(probs * target_onehot * valid_mask_float, dim=(2, 3, 4))
        denominator = torch.sum(probs * valid_mask_float, dim=(2, 3, 4)) + \
                      torch.sum(target_onehot * valid_mask_float, dim=(2, 3, 4))

        dice_score = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
        # Dice Loss = 1 - Mean(Dice Score per channel per batch)
        l_seg = 1.0 - torch.mean(dice_score)

        # 5. Masked KD Loss
        if teacher_pred is not None:
            # 同样只在有效区域计算 KD
            pred_for_kd = pred_kd if pred_kd is not None else pred

            # 使用 Masked MSE
            diff = (pred_for_kd - teacher_pred) * valid_mask_float
            # 归一化时只除以有效像素数，而不是总像素数
            valid_pixels = torch.sum(valid_mask_float) * pred.shape[1]  # pixels * channels
            l_kd = torch.sum(diff ** 2) / (valid_pixels + 1e-5)
        else:
            l_kd = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        return l_seg + self.lambda_kd * l_kd, l_seg, l_kd