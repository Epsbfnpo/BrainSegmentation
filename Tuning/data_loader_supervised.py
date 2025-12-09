"""
Data loader for supervised fine-tuning on dHCP dataset
ENHANCED: Better class-aware sampling and distributed caching
"""
import torch
import numpy as np
from monai.data import Dataset, DataLoader, CacheDataset, PersistentDataset
from monai.transforms import Compose
import json
import os
from typing import Dict, List, Tuple, Optional
import gc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import nibabel as nib
import torch.distributed as dist

from transforms_supervised import get_supervised_transforms


def load_data_split(split_json_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load data split from JSON file"""
    print(f"📂 Loading data split from: {split_json_path}")

    if not os.path.exists(split_json_path):
        raise FileNotFoundError(f"Data split file not found: {split_json_path}")

    with open(split_json_path, 'r') as f:
        split_data = json.load(f)

    train_files = split_data.get('training', [])
    val_files = split_data.get('validation', [])

    # Process files to ensure correct format
    processed_train = []
    processed_val = []

    for item in train_files:
        if isinstance(item['image'], list):
            image_path = item['image'][0]
        else:
            image_path = item['image']

        processed_item = {
            'image': image_path,
            'label': item['label']
        }

        # Verify files exist
        if not os.path.exists(image_path):
            print(f"⚠️ Warning: Image file not found: {image_path}")
            continue
        if not os.path.exists(item['label']):
            print(f"⚠️ Warning: Label file not found: {item['label']}")
            continue

        processed_train.append(processed_item)

    for item in val_files:
        if isinstance(item['image'], list):
            image_path = item['image'][0]
        else:
            image_path = item['image']

        processed_item = {
            'image': image_path,
            'label': item['label']
        }

        # Verify files exist
        if not os.path.exists(image_path):
            print(f"⚠️ Warning: Image file not found: {image_path}")
            continue
        if not os.path.exists(item['label']):
            print(f"⚠️ Warning: Label file not found: {item['label']}")
            continue

        processed_val.append(processed_item)

    print(f"✓ Loaded {len(processed_train)} training samples")
    print(f"✓ Loaded {len(processed_val)} validation samples")

    return processed_train, processed_val


def compute_sample_weights(train_files: List[Dict], class_prior_json: str = None,
                          num_classes: int = 87, weight_power: float = 0.75,
                          max_weight: float = 20.0) -> torch.Tensor:
    """Compute sample weights based on class presence for balanced sampling

    ENHANCED: Higher weights for rare classes with better scaling
    """
    print("\n🎯 Computing sample weights for class-aware sampling...")

    # Load class priors if available
    class_weights = None
    if class_prior_json and os.path.exists(class_prior_json):
        with open(class_prior_json, 'r') as f:
            prior_data = json.load(f)
        class_ratios = prior_data['class_ratios']

        # Compute inverse frequency weights for regions 1-87
        class_weights = []
        for i in range(1, 88):  # Skip background
            if i < len(class_ratios):
                # Use power scaling for more aggressive weighting
                weight = np.power(1.0 / (class_ratios[i] + 1e-6), weight_power)
            else:
                weight = 1.0
            class_weights.append(weight)

        class_weights = np.array(class_weights)
        class_weights = class_weights / class_weights.mean()  # Normalize
        class_weights = np.clip(class_weights, 0.1, max_weight)  # Higher upper limit

        print(f"  Class weight range: [{class_weights.min():.2f}, {class_weights.max():.2f}]")

    # Compute sample weights
    sample_weights = []

    for idx, item in enumerate(train_files):
        label_path = item['label']

        if class_weights is not None:
            try:
                label_nii = nib.load(label_path)
                label_data = label_nii.get_fdata()

                # Find unique classes in this sample (excluding background)
                unique_classes = np.unique(label_data)
                unique_classes = unique_classes[unique_classes > 0].astype(int)

                if len(unique_classes) > 0:
                    # Map brain regions (1-87) to weight indices (0-86)
                    weight_indices = unique_classes - 1
                    valid_indices = weight_indices[(weight_indices >= 0) & (weight_indices < len(class_weights))]

                    if len(valid_indices) > 0:
                        # Use max weight to prioritize samples with rare classes
                        sample_weight = class_weights[valid_indices].max()
                        # Boost if multiple rare classes present
                        num_rare = np.sum(class_weights[valid_indices] > 5.0)
                        if num_rare > 1:
                            sample_weight *= (1 + 0.1 * num_rare)
                    else:
                        sample_weight = 1.0
                else:
                    sample_weight = 0.1  # Low weight for empty samples

            except Exception as e:
                print(f"  Warning: Could not load {label_path}: {e}")
                sample_weight = 1.0
        else:
            sample_weight = 1.0

        sample_weights.append(sample_weight)

        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(train_files)} samples...")

    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

    print(f"✓ Sample weight statistics:")
    print(f"  Min: {sample_weights.min():.3f}")
    print(f"  Max: {sample_weights.max():.3f}")
    print(f"  Mean: {sample_weights.mean():.3f}")
    print(f"  Std: {sample_weights.std():.3f}")

    # Count high-weight samples
    high_weight_count = (sample_weights > 2.0).sum()
    print(f"  Samples with weight > 2.0: {high_weight_count} ({high_weight_count/len(sample_weights)*100:.1f}%)")

    return sample_weights


def load_class_weights(class_prior_json: str, num_classes: int = 87,
                      weight_power: float = 0.75, max_weight: float = 20.0) -> torch.Tensor:
    """Load class weights from prior distribution

    ENHANCED: More aggressive weighting for rare classes
    """
    if class_prior_json and os.path.exists(class_prior_json):
        print(f"📊 Loading class weights from: {class_prior_json}")
        with open(class_prior_json, 'r') as f:
            prior_data = json.load(f)

        class_ratios = prior_data['class_ratios']

        # For foreground-only mode with 87 classes
        if num_classes == 87:
            weights = []
            for i in range(1, 88):  # Brain regions 1-87
                if i < len(class_ratios):
                    # Power inverse frequency
                    weight = np.power(1.0 / (class_ratios[i] + 1e-6), weight_power)
                else:
                    weight = 1.0
                weights.append(weight)

            weights = torch.tensor(weights, dtype=torch.float32)

            # Normalize weights
            weights = weights / weights.mean()

            # Clip to reasonable range
            weights = torch.clamp(weights, 0.1, max_weight)

            print(f"✓ Class weights computed for 87 brain regions")
            print(f"  Min weight: {weights.min():.3f}")
            print(f"  Max weight: {weights.max():.3f}")
            print(f"  Mean weight: {weights.mean():.3f}")

            # Report extreme weights
            high_weight_classes = (weights > 10.0).sum()
            if high_weight_classes > 0:
                print(f"  Classes with weight > 10: {high_weight_classes}")

        return weights
    else:
        print("⚠️ No class weights file provided, using uniform weights")
        return torch.ones(num_classes)


def create_supervised_datasets(
    train_files: List[Dict],
    val_files: List[Dict],
    args
) -> Tuple[Dataset, Dataset]:
    """Create supervised datasets with appropriate transforms

    ENHANCED: Better caching strategy for distributed training
    """

    train_transforms = get_supervised_transforms(
        args=args,
        mode='train'
    )

    val_transforms = get_supervised_transforms(
        args=args,
        mode='val'
    )

    print("\n📄 Creating datasets...")

    # Determine caching strategy
    use_persistent = hasattr(args, 'use_persistent_dataset') and args.use_persistent_dataset
    use_cache = args.cache_rate > 0 and not use_persistent

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 24:
            use_cache = False
            use_persistent = False
            print(f"  Disabling cache due to limited GPU memory ({gpu_memory:.1f} GB)")

    if use_persistent:
        print("  Using PersistentDataset for better performance")

        # For distributed training, use rank-specific cache directories
        if torch.distributed.is_initialized():
            rank = dist.get_rank()
            cache_dir_train = os.path.join(args.cache_dir, f"rank{rank}", "train")
            cache_dir_val = os.path.join(args.cache_dir, f"rank{rank}", "val")
        else:
            cache_dir_train = os.path.join(args.cache_dir, "train")
            cache_dir_val = os.path.join(args.cache_dir, "val")

        # Clean cache if requested
        if hasattr(args, 'clean_cache') and args.clean_cache:
            import shutil
            if os.path.exists(cache_dir_train):
                shutil.rmtree(cache_dir_train)
            if os.path.exists(cache_dir_val):
                shutil.rmtree(cache_dir_val)
            print("  Cleaned cache directories")

        os.makedirs(cache_dir_train, exist_ok=True)
        os.makedirs(cache_dir_val, exist_ok=True)

        train_dataset = PersistentDataset(
            data=train_files,
            transform=train_transforms,
            cache_dir=cache_dir_train,
        )

        val_dataset = PersistentDataset(
            data=val_files,
            transform=val_transforms,
            cache_dir=cache_dir_val,
        )

    elif use_cache:
        print("  Using CacheDataset for better performance")
        print(f"  Cache rate: {args.cache_rate}")

        train_dataset = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=args.cache_rate,
            num_workers=args.num_workers,
        )

        val_dataset = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=min(args.cache_rate * 2, 1.0),
            num_workers=args.num_workers,
        )
    else:
        print("  Using standard Dataset (no caching)")

        train_dataset = Dataset(
            data=train_files,
            transform=train_transforms,
        )

        val_dataset = Dataset(
            data=val_files,
            transform=val_transforms,
        )

    return train_dataset, val_dataset


def get_supervised_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for supervised training

    ENHANCED: Better sampling strategy
    """

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n🖥️ GPU memory available: {gpu_memory:.2f} GB")

        # Adjust batch size based on GPU memory
        if gpu_memory < 16:
            suggested_batch_size = min(args.batch_size, 2)
            if suggested_batch_size < args.batch_size:
                print(f"  ⚠️ Reducing batch size from {args.batch_size} to {suggested_batch_size} due to limited GPU memory")
                args.batch_size = suggested_batch_size

    # Load data split
    train_files, val_files = load_data_split(args.data_split_json)

    # Optionally limit dataset size for debugging
    if hasattr(args, 'debug_samples') and args.debug_samples > 0:
        train_files = train_files[:args.debug_samples]
        val_files = val_files[:args.debug_samples]
        print(f"\n⚠️ DEBUG MODE: Using only {args.debug_samples} samples")

    # Create datasets
    train_dataset, val_dataset = create_supervised_datasets(
        train_files=train_files,
        val_files=val_files,
        args=args
    )

    print("\n📄 Creating data loaders...")

    # Create sampler for class-aware sampling if enabled
    train_sampler = None
    if hasattr(args, 'class_aware_sampling') and args.class_aware_sampling:
        print("\n🎯 Using class-aware sampling for training...")
        sample_weights = compute_sample_weights(
            train_files,
            args.class_prior_json if hasattr(args, 'class_prior_json') else None,
            args.out_channels,
            weight_power=args.weight_power if hasattr(args, 'weight_power') else 0.75,
            max_weight=args.max_weight if hasattr(args, 'max_weight') else 20.0
        )

        # Create weighted sampler
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # Sampler handles randomization
    else:
        shuffle = True

    # [CRITICAL FIX] Set pin_memory=False to avoid storage resizing errors with MetaTensors
    pin_memory = False

    # Training data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Validation data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch_size=1 for validation to handle different image sizes
        shuffle=False,
        num_workers=max(2, args.num_workers // 2),
        pin_memory=pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Test data loading
    print("\n🧪 Testing data loading...")
    try:
        test_batch = next(iter(train_loader))
        test_image = test_batch['image']
        test_label = test_batch['label']

        print(f"  Sample image shape: {test_image.shape}")
        print(f"  Sample label shape: {test_label.shape}")
        print(f"  Sample data type - Image: {test_image.dtype}, Label: {test_label.dtype}")
        print(f"  Sample value range: [{test_image.min():.3f}, {test_image.max():.3f}]")

        # Check label values
        unique_labels = torch.unique(test_label)
        print(f"  Label unique values: {unique_labels.tolist()[:10]}{'...' if len(unique_labels) > 10 else ''}")
        print(f"  Number of unique labels: {len(unique_labels)}")

        # Check for ignore index (-1)
        if -1 in unique_labels:
            print(f"  ✓ Background pixels (label=-1) detected")

        if torch.isnan(test_image).any():
            print("  ⚠️ WARNING: NaN detected in sample batch!")
        else:
            print("  ✓ No NaN in sample batch")

    except Exception as e:
        print(f"  ❌ Error during data loading test: {str(e)}")
        raise

    print(f"\n✓ Data loaders created successfully!")
    print(f"  Training batches: {len(train_loader)} x batch_size={args.batch_size}")
    print(f"  Validation batches: {len(val_loader)} x batch_size=1")
    if train_sampler is not None:
        print(f"  Using class-aware sampling: Yes")

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()

    return train_loader, val_loader


def get_optimal_batch_size(model: nn.Module,
                          device: torch.device,
                          roi_size: List[int],
                          start_batch_size: int = 2,
                          max_batch_size: int = 32,
                          use_amp: bool = False) -> int:
    """Dynamically find the maximum batch size that fits in GPU memory"""

    print("🔎 Finding optimal batch size...")

    batch_size = start_batch_size
    model.eval()

    # Binary search for optimal batch size
    low = 1
    high = max_batch_size
    optimal_batch_size = 1

    while low <= high:
        batch_size = (low + high) // 2

        try:
            # Clear cache before test
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Create dummy input
            dummy_input = torch.randn(
                batch_size, 1, roi_size[0], roi_size[1], roi_size[2],
                device=device, dtype=torch.float32
            )

            # Try forward pass
            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(dummy_input)
                else:
                    output = model(dummy_input)

            # Try backward pass (more memory intensive)
            if batch_size > 1:
                target = torch.randint(0, model.out_channels if hasattr(model, 'out_channels') else 87,
                                      (batch_size, roi_size[0], roi_size[1], roi_size[2]),
                                      device=device)
                loss = F.cross_entropy(output, target, ignore_index=-1)
                loss.backward()

            # If successful, this batch size works
            optimal_batch_size = batch_size
            low = batch_size + 1

            print(f"  ✓ Batch size {batch_size} fits in memory")

            # Clean up
            del dummy_input, output
            if batch_size > 1:
                del target, loss
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA out of memory" in str(e):
                high = batch_size - 1
                print(f"  ✗ Batch size {batch_size} exceeds memory")
                torch.cuda.empty_cache()
            else:
                print(f"  ⚠️ Unexpected error with batch size {batch_size}: {e}")
                raise e

    # Final cleanup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Safety margin: use 90% of the optimal batch size
    safe_batch_size = max(1, int(optimal_batch_size * 0.9))

    print(f"\n✓ Optimal batch size: {optimal_batch_size}")
    print(f"✓ Safe batch size (90%): {safe_batch_size}")

    return safe_batch_size