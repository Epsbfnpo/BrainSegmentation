import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple
from monai.data import CacheDataset, DataLoader, Dataset, MetaTensor, list_data_collate
from monai.transforms import (
    CenterSpatialCropd, Compose, CopyItemsd, EnsureChannelFirstd, EnsureTyped,
    LoadImaged, MapTransform, Orientationd, RandAdjustContrastd, RandBiasFieldd,
    RandCropByLabelClassesd, RandGaussianNoised, RandGaussianSmoothd,
    RandHistogramShiftd, RandRotated, RandScaleIntensityd, RandShiftIntensityd,
    RandZoomd, Spacingd, SpatialPadd, Randomizable, RandCropByPosNegLabeld
)
from monai.networks.nets import SwinUNETR
from utils_medseqft import robust_one_hot


# --- 0. Safe Collate Function (关键修复) ---
def safe_collate(batch):
    """
    自定义 Collate 函数：
    1. 过滤掉 Dataset 返回的 None 样本。
    2. 处理 RandCropByLabelClassesd 返回的列表结构（Flatten）。
    3. 过滤掉列表中的 None 元素。
    4. [新增] 移除不一致的 metadata 字典，防止 collate 崩溃。
    """
    cleaned_batch = []
    for item in batch:
        if item is None:
            continue
        if isinstance(item, list):
            # 展开列表并过滤 None
            valid_sub_items = [x for x in item if x is not None]
            cleaned_batch.extend(valid_sub_items)
        else:
            cleaned_batch.append(item)

    if len(cleaned_batch) == 0:
        print("⚠️ Warning: Batch is empty after filtering None/Invalid samples.")
        return {}

    # --- [关键修复] ---
    # 训练循环不需要 metadata，且它的 Key 不一致会导致 list_data_collate 崩溃。
    # 我们在这里统一移除它。
    for item in cleaned_batch:
        if isinstance(item, dict) and "metadata" in item:
            del item["metadata"]
    # ------------------

    return list_data_collate(cleaned_batch)


# --- 1. Data Loader Components ---

class ExtractAged(MapTransform):
    def __init__(self, metadata_key: str = "metadata"):
        super().__init__(keys=None)
        self.metadata_key = metadata_key

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        # 增强鲁棒性：处理 metadata 为 None 的情况
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
                if (hi - lo) > 1e-6 and not np.isnan(hi) and not np.isnan(lo):
                    norm = (np.clip(array, lo, hi) - lo) / (hi - lo)
                else:
                    norm = np.zeros_like(array)
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

    if args.foreground_only:
        transforms.append(RemapLabelsd(keys=["label"]))

    transforms.append(SpatialPadd(keys=["image", "label"], spatial_size=spatial_size, method="end"))

    if mode == "train":
        transforms.append(
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,  # 前景采样权重
                neg=1,  # 背景采样权重
                num_samples=1,
                image_key="image",
                image_threshold=0
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


def get_dataloaders(args, shuffle_train: bool = True):
    with open(args.split_json, "r") as f:
        data = json.load(f)

    def process(items):
        # 增强鲁棒性：确保 i 不为 None 且 key 存在
        valid_items = []
        if items:
            for i in items:
                if not i: continue
                img = i.get("image")
                valid_items.append({
                    "image": img[0] if isinstance(img, list) else img,
                    "label": i.get("label"),
                    # 强制转换为字典，如果是 None 则变为空字典 {}
                    "metadata": i.get("metadata") or {}
                })
        return valid_items

    train_files = process(data.get("training", []))
    val_files = process(data.get("validation", []))

    if getattr(args, "buffer_json", None) and os.path.exists(args.buffer_json):
        with open(args.buffer_json, "r") as f:
            buffer_data = json.load(f)
        train_files.extend(process(buffer_data))

    train_ds = CacheDataset(data=train_files, transform=_get_transforms(args, "train"),
                            cache_rate=args.cache_rate, num_workers=args.num_workers) if args.cache_rate > 0 else \
        Dataset(data=train_files, transform=_get_transforms(args, "train"))

    val_ds = Dataset(data=val_files, transform=_get_transforms(args, "val"))

    # 关键修改：使用 safe_collate 替代默认 collation
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle_train,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=safe_collate)  # <--- 使用自定义 Collate

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader


# --- 2. Model Wrapper (保持不变) ---
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


# --- 3. Loss Functions (保持不变) ---
class MedSeqFTLoss(nn.Module):
    def __init__(self, num_classes: int, lambda_kd: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = -1
        self.lambda_kd = lambda_kd

    def forward(
            self, pred: torch.Tensor, target: torch.Tensor, teacher_pred: Optional[torch.Tensor],
            pred_kd: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        target_onehot, valid_mask = robust_one_hot(target, self.num_classes, self.ignore_index)
        valid_mask_float = valid_mask.to(dtype=pred.dtype)

        probs = F.softmax(pred, dim=1)

        intersection = torch.sum(probs * target_onehot * valid_mask_float, dim=(2, 3, 4))
        denominator = torch.sum(probs * valid_mask_float, dim=(2, 3, 4)) + \
                      torch.sum(target_onehot * valid_mask_float, dim=(2, 3, 4))

        dice_score = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
        l_seg = 1.0 - torch.mean(dice_score)

        if teacher_pred is not None:
            pred_for_kd = pred_kd if pred_kd is not None else pred
            diff = (pred_for_kd - teacher_pred) * valid_mask_float
            valid_pixels = torch.sum(valid_mask_float) * pred.shape[1]
            l_kd = torch.sum(diff ** 2) / (valid_pixels + 1e-5)
        else:
            l_kd = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        return l_seg + self.lambda_kd * l_kd, l_seg, l_kd