import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandFlipd, RandGaussianNoised,
    RandGibbsNoised, ToTensord, RandSpatialCropd, SpatialPadd, CenterSpatialCropd,
    RandScaleIntensityd, RandShiftIntensityd, Transform, MapTransform, RandRotated,
    RandZoomd, RandGaussianSmoothd, RandAdjustContrastd, RandHistogramShiftd,
    RandBiasFieldd, RandCropByLabelClassesd, Spacingd, Orientationd, Randomizable, EnsureTyped
)
from monai.data import Dataset, DataLoader, MetaTensor, CacheDataset
import json
import traceback
import os
import gc
import nibabel as nib
import warnings
from typing import Optional

from age_aware_modules import prepare_class_ratios

warnings.filterwarnings('ignore')


class ExtractAged(MapTransform):
    """Extract age information from metadata and add to data dict"""

    def __init__(self, keys=["age"], metadata_key="metadata"):
        super().__init__(keys)
        self.metadata_key = metadata_key

    def __call__(self, data):
        d = dict(data)

        # Extract age from metadata
        if self.metadata_key in d:
            metadata = d[self.metadata_key]

            # Try different age fields
            age = None
            if isinstance(metadata, dict):
                # Try scan_age first (dHCP format)
                if 'scan_age' in metadata:
                    age = float(metadata['scan_age'])
                # Try PMA (postmenstrual age) for PPREMO/PREBO
                elif 'PMA' in metadata:
                    age = float(metadata['PMA'])
                elif 'pma' in metadata:
                    age = float(metadata['pma'])
                # Try gestational age + postnatal age
                elif 'ga' in metadata and 'pna' in metadata:
                    age = float(metadata['ga']) + float(metadata['pna'])
                elif 'GA' in metadata and 'PNA' in metadata:
                    age = float(metadata['GA']) + float(metadata['PNA'])

            # Store age in data dict
            if age is not None:
                d['age'] = torch.tensor([age], dtype=torch.float32)
            else:
                # Default to average age if not found
                d['age'] = torch.tensor([40.0], dtype=torch.float32)  # Average PMA

        else:
            # Default age if no metadata
            d['age'] = torch.tensor([40.0], dtype=torch.float32)

        return d


class PercentileNormalizationd(MapTransform):
    def __init__(self, keys, lower=1, upper=99, b_min=0.0, b_max=1.0, use_label_mask=True):
        super().__init__(keys)
        self.lower = lower
        self.upper = upper
        self.b_min = b_min
        self.b_max = b_max
        self.use_label_mask = use_label_mask

    def __call__(self, data):
        d = dict(data)
        label_mask = None
        if self.use_label_mask and 'label' in d:
            label = d['label']
            if isinstance(label, MetaTensor):
                label_np = label.array
            else:
                label_np = np.asarray(label)
            label_mask = label_np > 0
        for key in self.key_iterator(d):
            image = d[key]
            if isinstance(image, MetaTensor):
                image_np = image.array
                image_meta = image.meta
            else:
                image_np = np.asarray(image)
                image_meta = None
            if label_mask is not None:
                non_zero_mask = label_mask
            else:
                non_zero_mask = image_np > 0
            if non_zero_mask.any():
                non_zero_voxels = image_np[non_zero_mask]
                lower_percentile = np.percentile(non_zero_voxels, self.lower)
                upper_percentile = np.percentile(non_zero_voxels, self.upper)
                image_clipped = np.clip(image_np, lower_percentile, upper_percentile)
                if upper_percentile > lower_percentile:
                    image_norm = (image_clipped - lower_percentile) / (upper_percentile - lower_percentile)
                    image_norm = image_norm * (self.b_max - self.b_min) + self.b_min
                else:
                    image_norm = np.ones_like(image_np) * self.b_min
                image_norm[~non_zero_mask] = 0
                if image_meta is not None:
                    d[key] = MetaTensor(image_norm, meta=image_meta)
                else:
                    d[key] = image_norm
        return d


class RemapLabelsd(MapTransform):
    def __init__(self, keys, do_remap=True):
        super().__init__(keys)
        self.do_remap = do_remap
        self._first_call = True

    def __call__(self, data):
        if not self.do_remap:
            return data
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            if isinstance(label, MetaTensor):
                label_np = label.array
                label_meta = label.meta
            else:
                label_np = np.asarray(label)
                label_meta = None
            label_np = label_np.astype(np.int32)
            if self._first_call and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                unique_before = np.unique(label_np)
                print(f"\n[RemapLabelsd] First sample analysis:")
                print(f"  Original label shape: {label_np.shape}")
                print(f"  Original unique values: {len(unique_before)} values")
                print(f"  Original range: [{unique_before.min()}, {unique_before.max()}]")
                if len(unique_before) < 100:
                    print(f"  Label distribution:")
                    for val in unique_before[:10]:
                        count = (label_np == val).sum()
                        print(f"    Label {val}: {count} voxels")
                    if len(unique_before) > 10:
                        print(f"    ... and {len(unique_before) - 10} more labels")
            unique_before = np.unique(label_np)
            if unique_before.max() > 87 or unique_before.min() < 0:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(f"⚠️  WARNING: Labels out of expected range!")
                    print(f"   Found: min={unique_before.min()}, max={unique_before.max()}")
                    print(f"   Expected: 0-87")
                label_np = np.clip(label_np, 0, 87)
            label_remapped = np.full_like(label_np, -1, dtype=np.int32)
            for orig_label in range(1, 88):
                mask = (label_np == orig_label)
                if mask.any():
                    label_remapped[mask] = orig_label - 1
            if self._first_call and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                unique_after = np.unique(label_remapped)
                print(f"  Remapped unique values: {len(unique_after)} values")
                print(f"  Remapped range: [{unique_after.min()}, {unique_after.max()}]")
                if -1 in unique_after:
                    print(f"  Background voxels (mapped to -1): {(label_remapped == -1).sum()}")
                if unique_after.max() > 86:
                    print(f"  ⌠ERROR: Remapped labels exceed 86!")
                self._first_call = False
            valid_mask = (label_remapped == -1) | ((label_remapped >= 0) & (label_remapped <= 86))
            if not valid_mask.all():
                invalid_values = np.unique(label_remapped[~valid_mask])
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(f"⌠ERROR: Invalid remapped values found: {invalid_values}")
                label_remapped[~valid_mask] = 0
            label_remapped = label_remapped.astype(np.float32)
            if label_meta is not None:
                d[key] = MetaTensor(label_remapped, meta=label_meta)
            else:
                d[key] = label_remapped
        return d


class RandFlipLateralityd(Randomizable, MapTransform):
    """
    Flip on given spatial axis; if axis is the left-right axis (x in RAS) then
    swap paired label IDs to preserve laterality semantics.
    Fixed to handle both numpy arrays and torch tensors properly.
    """

    def __init__(self, keys, spatial_axis=0, prob=0.5, swap_label_pairs=None):
        super().__init__(keys)
        self.spatial_axis = spatial_axis  # 0:x, 1:y, 2:z (after channel dim)
        self.prob = float(prob)
        self.swap_label_pairs = swap_label_pairs or []

    def randomize(self, data=None):
        self._do = self.R.random() < self.prob

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if not self._do:
            return d

        for key in self.key_iterator(d):
            arr = d[key]
            meta = None
            if isinstance(arr, MetaTensor):
                meta = arr.meta
                arr = arr.array

            # For channel-first data: shape (C, X, Y, Z)
            # spatial_axis 0 means X dimension (index 1 after channel)
            axis = 1 + self.spatial_axis

            # Handle both numpy arrays and torch tensors
            is_tensor = isinstance(arr, torch.Tensor)
            if is_tensor:
                # Use torch flip for tensors
                device = arr.device
                dtype = arr.dtype
                arr = torch.flip(arr, dims=[axis]).contiguous()
            else:
                # Use numpy flip for numpy arrays
                arr = np.flip(arr, axis=axis).copy()

            # Only swap labels for the label key and x-axis flip
            if key == "label" and self.spatial_axis == 0 and len(self.swap_label_pairs) > 0:
                if is_tensor:
                    # Handle torch tensor case
                    arr_copy = arr.clone()

                    # Handle both (1, X, Y, Z) and (X, Y, Z) shapes
                    if arr_copy.ndim == 4 and arr_copy.shape[0] == 1:
                        lab = arr_copy[0]
                        for a, b in self.swap_label_pairs:
                            # Create masks for swapping
                            mask_a = (lab == a)
                            mask_b = (lab == b)
                            # Use temporary variable to avoid overwriting
                            temp = torch.zeros_like(lab)
                            temp[mask_a] = b
                            temp[mask_b] = a
                            # Apply only where masks are true
                            lab = torch.where(mask_a | mask_b, temp, lab)
                        arr_copy[0] = lab
                        arr = arr_copy
                    else:
                        for a, b in self.swap_label_pairs:
                            mask_a = (arr_copy == a)
                            mask_b = (arr_copy == b)
                            temp = torch.zeros_like(arr_copy)
                            temp[mask_a] = b
                            temp[mask_b] = a
                            arr_copy = torch.where(mask_a | mask_b, temp, arr_copy)
                        arr = arr_copy
                else:
                    # Handle numpy array case
                    arr_copy = arr.copy()

                    # Handle both (1, X, Y, Z) and (X, Y, Z) shapes
                    if arr_copy.ndim == 4 and arr_copy.shape[0] == 1:
                        lab = arr_copy[0]
                        for a, b in self.swap_label_pairs:
                            mask_a = (lab == a)
                            mask_b = (lab == b)
                            lab[mask_a] = b
                            lab[mask_b] = a
                        arr = arr_copy
                    else:
                        for a, b in self.swap_label_pairs:
                            mask_a = (arr_copy == a)
                            mask_b = (arr_copy == b)
                            arr_copy[mask_a] = b
                            arr_copy[mask_b] = a
                        arr = arr_copy

            d[key] = MetaTensor(arr, meta=meta) if meta is not None else arr
        return d


def _load_lr_pairs(args):
    """Load laterality pairs from JSON and convert to 0-based if needed"""
    pairs = []
    if getattr(args, 'laterality_pairs_json', None) and os.path.exists(args.laterality_pairs_json):
        with open(args.laterality_pairs_json, 'r') as f:
            raw = json.load(f)  # Expected format: [[17,18],[36,37],...]
        for a, b in raw:
            # If foreground_only: map 1..87 -> 0..86
            if args.foreground_only:
                a_idx = a - 1
                b_idx = b - 1
            else:
                a_idx = a
                b_idx = b
            pairs.append((a_idx, b_idx))

        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main:
            print(f"\n📋 Loaded {len(pairs)} laterality pairs")
            if len(pairs) > 0:
                print(f"  First few pairs (0-based): {pairs[:3]}...")
    return pairs


def get_weighted_ratios_for_small_classes(class_prior_path: str, num_small_classes: int = 20,
                                          boost_factor: float = 2.0,
                                          foreground_only: bool = True,
                                          expected_num_classes: Optional[int] = None):
    if class_prior_path is None or not os.path.exists(class_prior_path):
        return None
    with open(class_prior_path, 'r') as f:
        prior_data = json.load(f)
    if expected_num_classes is None:
        expected_num_classes = prior_data.get('num_classes')
        if expected_num_classes is None:
            expected_num_classes = 87 if foreground_only else 88
    class_ratios = prepare_class_ratios(
        prior_data,
        expected_num_classes=expected_num_classes,
        foreground_only=foreground_only,
        is_main=(not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0),
        context="Weighted sampling prior"
    )
    epsilon = 1e-7
    weights = 1.0 / (class_ratios + epsilon)
    weights = weights / weights.mean()
    sorted_indices = np.argsort(class_ratios)
    small_class_indices = sorted_indices[:num_small_classes]
    ratios = np.ones(len(weights))
    for idx in small_class_indices:
        ratios[idx] = boost_factor
    ratios = ratios / ratios.sum() * len(ratios)

    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if is_main:
        print(f"\n📊 Weighted sampling ratios computed:")
        print(f"  Number of classes: {len(ratios)}")
        print(f"  Boosted {num_small_classes} smallest classes by {boost_factor}x")
        print(f"  Ratio range: [{ratios.min():.3f}, {ratios.max():.3f}]")
        print(f"  Boosted classes (indices after removing background):")
        for i, idx in enumerate(small_class_indices[:5]):
            print(f"    Class {idx}: ratio={class_ratios[idx]:.6f}, weight={ratios[idx]:.3f}")
        if len(small_class_indices) > 5:
            print(f"    ... and {len(small_class_indices) - 5} more")
    return ratios.tolist()


def get_cache_transforms(args, is_registered=False):
    """Get transforms for cached dataset (train) with age extraction"""
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ExtractAged(keys=["age"]),  # NEW: Extract age information
    ]

    # Add spacing transform if enabled
    if getattr(args, 'apply_spacing', True):
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=getattr(args, 'target_spacing', [0.8, 0.8, 0.8]),
                mode=("bilinear", "nearest"),
                align_corners=True,
            )
        )

    # Add orientation transform if enabled
    if getattr(args, 'apply_orientation', True):
        transforms.append(
            Orientationd(
                keys=["image", "label"],
                axcodes="RAS",
            )
        )

    # Add normalization
    transforms.append(
        PercentileNormalizationd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            use_label_mask=is_registered
        )
    )

    # Add label remapping if foreground only
    if args.foreground_only:
        transforms.append(RemapLabelsd(keys=["label"], do_remap=True))

    # Add cropping based on registration status
    if is_registered and args.use_label_crop:
        ratios = get_weighted_ratios_for_small_classes(
            args.target_prior_json,
            num_small_classes=getattr(args, 'num_small_classes_boost', 20),
            boost_factor=getattr(args, 'small_class_boost_factor', 2.0),
            foreground_only=args.foreground_only,
            expected_num_classes=args.out_channels
        )
        if ratios is None:
            ratios = [1] * args.out_channels
        transforms.extend([
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                ratios=ratios,
                num_classes=args.out_channels,
                num_samples=1,
                image_key="image",
                image_threshold=0,
                allow_smaller=True
            ),
            EnsureTyped(keys=["image", "label", "age"], track_meta=False)
        ])
    else:
        transforms.extend([
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                mode="constant",
                constant_values=0,
            ),
            CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
            ),
            EnsureTyped(keys=["image", "label", "age"], track_meta=False)
        ])

    return Compose(transforms)


def get_train_augmentations(args, is_registered=False):
    """Get augmentation transforms for training"""
    # Load laterality pairs for left-right flipping
    lr_pairs = _load_lr_pairs(args)

    if is_registered:
        # Minimal augmentations for registered data
        transforms = [
            RandFlipLateralityd(keys=["image", "label"], spatial_axis=0, prob=0.5, swap_label_pairs=lr_pairs),
            RandFlipLateralityd(keys=["image", "label"], spatial_axis=1, prob=0.5, swap_label_pairs=[]),
            RandFlipLateralityd(keys=["image", "label"], spatial_axis=2, prob=0.5, swap_label_pairs=[]),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
            RandShiftIntensityd(keys="image", offsets=0.05, prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.02),
        ]
    else:
        # Full augmentations for non-registered data
        transforms = [
            RandFlipLateralityd(keys=["image", "label"], spatial_axis=0, prob=0.5, swap_label_pairs=lr_pairs),
            RandFlipLateralityd(keys=["image", "label"], spatial_axis=1, prob=0.5, swap_label_pairs=[]),
            RandFlipLateralityd(keys=["image", "label"], spatial_axis=2, prob=0.5, swap_label_pairs=[]),
            RandRotated(
                keys=["image", "label"],
                range_x=0.5,
                range_y=0.5,
                range_z=0.5,
                prob=0.3,
                mode=["bilinear", "nearest"],
            ),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.7,
                max_zoom=1.3,
                prob=0.3,
                mode=["trilinear", "nearest"],
            ),
            RandScaleIntensityd(keys="image", factors=0.3, prob=0.3),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
            RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.7, 1.5)),
            RandHistogramShiftd(keys="image", prob=0.2, num_control_points=(5, 15)),
            RandBiasFieldd(keys="image", prob=0.2, degree=3, coeff_range=(-0.5, 0.5)),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
                prob=0.2
            ),
            RandGibbsNoised(keys=["image"], prob=0.2, alpha=(0.0, 1.0)),
        ]

    return Compose(transforms)


def get_transforms(args, mode="train", dataset_type=None, is_registered=False):
    """Get transforms for validation with age extraction"""
    if mode == "train":
        return get_cache_transforms(args, is_registered=is_registered)
    else:
        transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ExtractAged(keys=["age"]),  # NEW: Extract age information
        ]

        # Add spacing transform if enabled
        if getattr(args, 'apply_spacing', True):
            transforms.append(
                Spacingd(
                    keys=["image", "label"],
                    pixdim=getattr(args, 'target_spacing', [0.8, 0.8, 0.8]),
                    mode=("bilinear", "nearest"),
                    align_corners=True,
                )
            )

        # Add orientation transform if enabled
        if getattr(args, 'apply_orientation', True):
            transforms.append(
                Orientationd(
                    keys=["image", "label"],
                    axcodes="RAS",
                )
            )

        # Add normalization
        transforms.append(
            PercentileNormalizationd(
                keys=["image"],
                lower=1,
                upper=99,
                b_min=0.0,
                b_max=1.0,
                use_label_mask=is_registered
            )
        )

        # Add label remapping if foreground only
        if args.foreground_only:
            transforms.append(RemapLabelsd(keys=["label"], do_remap=True))

        # Add padding and cropping
        transforms.extend([
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                mode="constant",
                constant_values=0,
            ),
            CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
            ),
            EnsureTyped(keys=["image", "label", "age"], track_meta=False)
        ])

    return Compose(transforms)


def process_data_files(data_files):
    """Process data files and preserve metadata"""
    processed_files = []
    for i, item in enumerate(data_files):
        processed_item = {}
        if isinstance(item['image'], list):
            processed_item['image'] = item['image'][0]
        else:
            processed_item['image'] = item['image']
        processed_item['label'] = item['label']
        # Preserve metadata for age extraction
        if 'metadata' in item:
            processed_item['metadata'] = item['metadata']
        processed_files.append(processed_item)
    return processed_files


def shard_data_for_rank(data_files, world_size, rank):
    sharded_files = []
    for i, item in enumerate(data_files):
        if i % world_size == rank:
            sharded_files.append(item)
    return sharded_files


class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, augmentation_transform):
        self.base_dataset = base_dataset
        self.augmentation_transform = augmentation_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        data = self.base_dataset[index]
        if self.augmentation_transform is not None:
            data = self.augmentation_transform(data)
        return data


def get_source_target_dataloaders(args, is_distributed=False, world_size=1, rank=0):
    try:
        is_main = (not is_distributed) or rank == 0
        if torch.cuda.is_available() and is_main:
            print(f"\n🖥️  Initial GPU memory: {torch.cuda.memory_allocated(rank) / 1024 ** 3:.2f} GB allocated")

        # Use explicit registration flags from args
        is_source_registered = bool(getattr(args, 'source_is_registered', False))
        is_target_registered = bool(getattr(args, 'target_is_registered', False))

        if is_main:
            print(f"\n📊 Data Registration Status (from CLI args):")
            print(f"  Source data registered: {is_source_registered}")
            print(f"  Target data registered: {is_target_registered}")
            print(f"  Apply spacing: {getattr(args, 'apply_spacing', True)}")
            print(f"  Target spacing: {getattr(args, 'target_spacing', [0.8, 0.8, 0.8])}")
            print(f"  Apply orientation: {getattr(args, 'apply_orientation', True)}")
            if getattr(args, 'laterality_pairs_json', None):
                print(f"  Laterality pairs: {args.laterality_pairs_json}")
            print(f"  ✨ AGE-AWARE TRAINING ENABLED")

        if is_main:
            print(f"\n📂 Loading source split from: {args.source_split_json}")
        with open(args.source_split_json, 'r') as f:
            source_split = json.load(f)
        source_train_files = source_split['training']
        source_val_files = source_split['validation']

        if is_main:
            print(f"📂 Loading target split from: {args.split_json}")
        with open(args.split_json, 'r') as f:
            target_split = json.load(f)
        target_train_files = target_split['training']
        target_val_files = target_split['validation']

        if is_main:
            print(f"\n📊 Dataset Statistics:")
            print(f"  Source Domain (dHCP):")
            print(f"    Training: {len(source_train_files)} samples")
            print(f"    Validation: {len(source_val_files)} samples")
            print(f"  Target Domain (PPREMO/PREBO):")
            print(f"    Training: {len(target_train_files)} samples")
            print(f"    Validation: {len(target_val_files)} samples")

            # Sample age statistics
            ages_source = []
            for item in source_train_files[:20]:
                if 'metadata' in item and 'scan_age' in item['metadata']:
                    ages_source.append(float(item['metadata']['scan_age']))
            if ages_source:
                print(f"  Source age range (sample): {min(ages_source):.1f} - {max(ages_source):.1f} weeks")

        if is_main:
            print("\n📄 Processing data files...")
        source_train_files = process_data_files(source_train_files)
        source_val_files = process_data_files(source_val_files)
        target_train_files = process_data_files(target_train_files)
        target_val_files = process_data_files(target_val_files)

        if is_main:
            print("\n🎨 Creating transforms...")
            print("  📦 Cache transforms: Load + Age + Spacing + Orient + Normalize + Remap + Crop")
            if is_source_registered or is_target_registered:
                print("  🎲 Runtime augmentations: Laterality-aware flips + mild intensity")
            else:
                print("  🎲 Runtime augmentations: Full (Laterality-aware flips + Rotations + Intensity)")

        source_cache_transforms = get_cache_transforms(args, is_registered=is_source_registered)
        target_cache_transforms = get_cache_transforms(args, is_registered=is_target_registered)
        source_train_augmentations = get_train_augmentations(args, is_registered=is_source_registered)
        target_train_augmentations = get_train_augmentations(args, is_registered=is_target_registered)
        source_val_transforms = get_transforms(args, mode="val", is_registered=is_source_registered)
        target_val_transforms = get_transforms(args, mode="val", is_registered=is_target_registered)

        if is_main:
            print("\n📄 Creating datasets...")

        use_cache = args.cache_rate > 0
        if use_cache:
            DatasetClass = CacheDataset
            should_preshard = is_distributed and args.cache_rate == 1.0
            if should_preshard:
                if is_main:
                    print(f"\n🔪 Pre-sharding training data for distributed caching")
                    print(f"  Cache rate: {args.cache_rate}")
                    print(f"  World size: {world_size}")
                    print(f"  Each rank will cache 1/{world_size} of the data")
                    print(f"  This reduces memory usage by {world_size}x")
                source_train_files_sharded = shard_data_for_rank(source_train_files, world_size, rank)
                target_train_files_sharded = shard_data_for_rank(target_train_files, world_size, rank)
                print(
                    f"  🎯 Rank {rank}: caching {len(source_train_files_sharded)}/{len(source_train_files)} source samples")
                print(
                    f"  🎯 Rank {rank}: caching {len(target_train_files_sharded)}/{len(target_train_files)} target samples")
                source_train_files_for_cache = source_train_files_sharded
                target_train_files_for_cache = target_train_files_sharded
                use_distributed_sampler_for_training = False
            else:
                if is_main:
                    print(f"  Using CacheDataset with cache_rate={args.cache_rate}")
                    if args.cache_rate == 1.0:
                        print(f"  ⚠️  WARNING: Each GPU will cache ALL data (no pre-sharding)")
                        print(f"  ⚠️  This may cause OOM. Consider using distributed mode.")
                source_train_files_for_cache = source_train_files
                target_train_files_for_cache = target_train_files
                use_distributed_sampler_for_training = is_distributed
            cache_num_workers = min(2, args.cache_num_workers)
            if is_main:
                print(f"  Cache num_workers: {cache_num_workers} (reduced for memory)")
            dataset_kwargs = {'cache_rate': args.cache_rate, 'num_workers': cache_num_workers, 'progress': is_main}
        else:
            if is_main:
                print("  Using standard Dataset (no caching)")
            DatasetClass = Dataset
            dataset_kwargs = {}
            source_train_files_for_cache = source_train_files
            target_train_files_for_cache = target_train_files
            use_distributed_sampler_for_training = is_distributed

        if is_main:
            print("\n📦 Creating source training dataset (cache)...")
        source_train_cache_ds = DatasetClass(
            data=source_train_files_for_cache,
            transform=source_cache_transforms,
            **dataset_kwargs
        )
        gc.collect()
        torch.cuda.empty_cache()

        if is_main:
            print("📦 Creating target training dataset (cache)...")
        target_train_cache_ds = DatasetClass(
            data=target_train_files_for_cache,
            transform=target_cache_transforms,
            **dataset_kwargs
        )
        gc.collect()
        torch.cuda.empty_cache()

        if is_main:
            print("🎲 Wrapping with augmentation layer...")
        source_train_ds = AugmentedDataset(source_train_cache_ds, source_train_augmentations)
        target_train_ds = AugmentedDataset(target_train_cache_ds, target_train_augmentations)

        if is_main:
            print("📊 Creating validation datasets (no cache)...")
        source_val_ds = Dataset(
            data=source_val_files,
            transform=source_val_transforms,
        )
        target_val_ds = Dataset(
            data=target_val_files,
            transform=target_val_transforms,
        )

        if is_main:
            print("\n📄 Creating data loaders...")

        if use_distributed_sampler_for_training:
            source_train_sampler = DistributedSampler(source_train_ds, num_replicas=world_size, rank=rank, shuffle=True,
                                                      drop_last=True)
            target_train_sampler = DistributedSampler(target_train_ds, num_replicas=world_size, rank=rank, shuffle=True,
                                                      drop_last=True)
            train_shuffle = False
            if is_main:
                print(f"  Using DistributedSampler for training")
        else:
            source_train_sampler = None
            target_train_sampler = None
            train_shuffle = True
            if is_main:
                print(f"  Using pre-sharded data without DistributedSampler")

        if is_distributed:
            source_val_sampler = DistributedSampler(source_val_ds, num_replicas=world_size, rank=rank, shuffle=False,
                                                    drop_last=False)
            target_val_sampler = DistributedSampler(target_val_ds, num_replicas=world_size, rank=rank, shuffle=False,
                                                    drop_last=False)
        else:
            source_val_sampler = None
            target_val_sampler = None

        dataloader_num_workers = min(2, args.num_workers)
        source_train_loader = DataLoader(
            source_train_ds,
            batch_size=args.batch_size,
            sampler=source_train_sampler,
            shuffle=train_shuffle,
            num_workers=dataloader_num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
            prefetch_factor=1,
        )
        source_val_loader = DataLoader(
            source_val_ds,
            batch_size=1,
            sampler=source_val_sampler,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        target_train_loader = DataLoader(
            target_train_ds,
            batch_size=args.batch_size,
            sampler=target_train_sampler,
            shuffle=train_shuffle,
            num_workers=dataloader_num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
            prefetch_factor=1,
        )
        target_val_loader = DataLoader(
            target_val_ds,
            batch_size=1,
            sampler=target_val_sampler,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        if is_main:
            print("\n✅ Data loaders created successfully!")
            print(f"  Training batch size per GPU: {args.batch_size}")
            if is_distributed:
                print(f"  Total effective batch size: {args.batch_size * world_size}")
                if args.cache_rate == 1.0 and use_cache:
                    if should_preshard:
                        print(f"  ✨ Memory optimization: Each GPU caches ~{100 / world_size:.1f}% of training data")
                        print(f"  ✨ Total coverage: 100% (distributed across {world_size} GPUs)")
                    else:
                        print(f"  ⚠️  No memory optimization: Each GPU caches 100% of data")
            print(f"  DataLoader num_workers: {dataloader_num_workers} (reduced for memory)")
            print(f"  Cache num_workers: {cache_num_workers if use_cache else 'N/A'}")
            print(f"  ✨ Weighted sampling enabled for small classes")
            print(f"  ✨ Laterality-aware augmentations enabled")
            print(f"  ✨ AGE-AWARE TRAINING ENABLED")

        gc.collect()
        torch.cuda.empty_cache()
        return source_train_loader, source_val_loader, target_train_loader, target_val_loader

    except Exception as e:
        if is_main:
            print(f"\n⌠Error creating data loaders: {str(e)}")
            traceback.print_exc()
        raise