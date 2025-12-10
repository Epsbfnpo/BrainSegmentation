import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
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
from monai.losses import DiceCELoss, DiceFocalLoss

# --- 1. Data Loader Components (Copied from new/data_loader_age_aware.py) ---

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
            remapped = np.full_like(array, -1, dtype=np.int32)
            mask = array > 0
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
            RandCropByLabelClassesd(
                keys=["image", "label"], label_key="label", spatial_size=spatial_size,
                num_classes=args.out_channels, num_samples=1, image_key="image", image_threshold=0
            )
        )
        # Augmentations matching your baseline
        transforms.extend([
            RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, prob=0.35, mode=["bilinear", "nearest"]),
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader

# --- 2. Model Wrapper (Copied/Simplified from new/age_aware_modules.py) ---

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
        if "state_dict" in state: state = state["state_dict"]
        elif "model_state_dict" in state: state = state["model_state_dict"]
        # Strip module. prefix if DDP
        state = {k.replace("module.", ""): v for k, v in state.items()}
        # Strip backbone. prefix if loaded from previous wrapper
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        self.backbone.load_state_dict(state, strict=False)

# --- 3. Loss Functions ---

class MedSeqFTLoss(nn.Module):
    """
    Combined Loss for MedSeqFT Phase 1 (KD-based FFT).
    L = L_seg + lambda * L_kd
    """
    def __init__(self, num_classes, lambda_kd=1.0):
        super().__init__()
        self.seg_loss = DiceFocalLoss(include_background=True, softmax=True, to_onehot_y=True, batch=True, gamma=2.0)
        self.kd_loss = nn.MSELoss()
        self.lambda_kd = lambda_kd

    def forward(self, pred, target, teacher_pred):
        # Segmentation Loss (Supervised)
        l_seg = self.seg_loss(pred, target)
        
        # Knowledge Distillation Loss (Unsupervised constraint from Source)
        # Using Logits Distillation (MSE on output logits)
        l_kd = self.kd_loss(pred, teacher_pred)
        
        return l_seg + self.lambda_kd * l_kd, l_seg, l_kd
