"""
Data loader for SSL pretraining on dHCP dataset
Handles T2w single modality data with appropriate preprocessing
DISTRIBUTED VERSION with multi-GPU support
"""
import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from monai.data import Dataset, DataLoader, CacheDataset, PersistentDataset
from monai.transforms import Compose
import json
import os
from typing import Dict, List, Tuple
import gc
import tempfile

from transforms_ssl import get_ssl_transforms


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
        processed_val.append(processed_item)

    print(f"✓ Loaded {len(processed_train)} training samples")
    print(f"✓ Loaded {len(processed_val)} validation samples")

    return processed_train, processed_val


def shard_data_for_rank(data_files, world_size, rank):
    """Shard data for specific rank when cache_rate=1.0"""
    sharded_files = []
    for i, item in enumerate(data_files):
        if i % world_size == rank:
            sharded_files.append(item)
    return sharded_files


def create_ssl_datasets(
        train_files: List[Dict],
        val_files: List[Dict],
        args,
        is_distributed: bool = False,
        world_size: int = 1,
        rank: int = 0
) -> Tuple[Dataset, Dataset]:
    """Create SSL datasets with appropriate transforms and optimized caching"""

    train_transforms = get_ssl_transforms(
        args=args,
        mode='train',
        target_spacing=args.target_spacing
    )

    val_transforms = get_ssl_transforms(
        args=args,
        mode='val',
        target_spacing=args.target_spacing
    )

    is_main = (not is_distributed) or rank == 0

    if is_main:
        print("\n📄 Creating datasets...")

    # Determine caching strategy based on args
    use_cache = args.cache_rate > 0
    use_persistent = args.use_persistent_cache if hasattr(args, 'use_persistent_cache') else False

    if use_persistent:
        # Use PersistentDataset for disk-based caching
        if is_main:
            print("  Using PersistentDataset for disk-based caching")

        # Create cache directory
        cache_dir = getattr(args, 'cache_dir', None)
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp(prefix='ssl_cache_')

        if is_distributed:
            cache_dir = os.path.join(cache_dir, f"rank{rank}")
        os.makedirs(os.path.join(cache_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, 'val'), exist_ok=True)

        if is_main:
            print(f"  Cache directory: {cache_dir}")

        train_dataset = PersistentDataset(
            data=train_files,
            transform=train_transforms,
            cache_dir=os.path.join(cache_dir, 'train'),
        )

        val_dataset = PersistentDataset(
            data=val_files,
            transform=val_transforms,
            cache_dir=os.path.join(cache_dir, 'val'),
        )

    elif use_cache:
        # Use CacheDataset for memory-based caching
        if is_main:
            print("  Using CacheDataset for memory-based caching")
            print(f"  Cache rate: {args.cache_rate}")

        # Get available system memory
        import psutil
        available_memory = psutil.virtual_memory().available / 1024**3  # GB
        if is_main:
            print(f"  Available system memory: {available_memory:.2f} GB")

        # Adjust cache rate based on available memory if needed
        if available_memory < 32 and args.cache_rate > 0.5:
            if is_main:
                print(f"  ⚠️  Limited memory detected, adjusting cache rate from {args.cache_rate} to 0.3")
            args.cache_rate = 0.3

        # For distributed training with full caching, shard data before caching
        should_preshard = is_distributed and args.cache_rate == 1.0

        if should_preshard:
            if is_main:
                print(f"\n🔪 Pre-sharding training data for distributed caching")
                print(f"  World size: {world_size}")
                print(f"  Each rank will cache 1/{world_size} of the data")

            train_files_sharded = shard_data_for_rank(train_files, world_size, rank)
            print(f"  🎯 Rank {rank}: caching {len(train_files_sharded)}/{len(train_files)} training samples")

            train_files_for_cache = train_files_sharded
        else:
            train_files_for_cache = train_files

        # Reduce workers for caching to save memory
        cache_num_workers = min(2, args.num_workers)

        train_dataset = CacheDataset(
            data=train_files_for_cache,
            transform=train_transforms,
            cache_rate=args.cache_rate,
            num_workers=cache_num_workers,
            progress=is_main,
        )

        val_dataset = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=min(args.cache_rate, 0.5),  # Lower cache rate for validation
            num_workers=max(2, cache_num_workers // 2),
            progress=is_main,
        )
    else:
        # Standard dataset without caching
        if is_main:
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


def get_ssl_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    """Create optimized data loaders for SSL pretraining (single GPU version)"""

    # Check available resources
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"\n🖥️  GPU memory available: {gpu_memory:.2f} GB")

    # Get CPU count for optimal num_workers
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"🖥️  CPU cores available: {cpu_count}")

    # Auto-adjust num_workers if not optimal
    if args.num_workers < min(8, cpu_count - 2):
        old_workers = args.num_workers
        args.num_workers = min(12, cpu_count - 2)  # Leave some cores for system
        print(f"  📈 Increased num_workers from {old_workers} to {args.num_workers} for better throughput")

    train_files, val_files = load_data_split(args.data_split_json)

    if len(train_files) > 1000:
        print(f"\n⚠️  Large dataset detected ({len(train_files)} samples). Consider using persistent cache.")

    train_dataset, val_dataset = create_ssl_datasets(
        train_files=train_files,
        val_files=val_files,
        args=args
    )

    print("\n📄 Creating optimized data loaders...")

    # Calculate optimal prefetch_factor
    prefetch_factor = getattr(args, 'prefetch_factor', 4)
    print(f"  Prefetch factor: {prefetch_factor}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=prefetch_factor if args.num_workers > 0 else None,
    )

    # Validation loader with optimized settings
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,  # Fixed for contrastive learning
        shuffle=False,
        num_workers=max(2, args.num_workers // 2),  # Use fewer workers for validation
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,  # Lower prefetch for validation
        drop_last=True,
    )

    print("\n🧪 Testing data loading...")
    test_batch = next(iter(train_loader))
    test_image = test_batch['image']
    print(f"  Sample batch shape: {test_image.shape}")
    print(f"  Sample data type: {test_image.dtype}")
    print(f"  Sample value range: [{test_image.min():.3f}, {test_image.max():.3f}]")

    if torch.isnan(test_image).any():
        print("  ⚠️  WARNING: NaN detected in sample batch!")
    else:
        print("  ✓ No NaN in sample batch")

    print(f"\n✓ Data loaders created successfully!")
    print(f"  Training batches: {len(train_loader)} x batch_size={args.batch_size}")
    print(f"  Validation batches: {len(val_loader)} x batch_size=2")
    print(f"  Optimization features enabled:")
    print(f"    - Persistent workers: ✓")
    print(f"    - Pin memory: ✓")
    print(f"    - Prefetching: ✓ (factor={prefetch_factor})")
    print(f"    - Multi-worker loading: ✓ ({args.num_workers} workers)")

    gc.collect()
    torch.cuda.empty_cache()

    return train_loader, val_loader


def get_ssl_dataloaders_distributed(
        args,
        is_distributed: bool = False,
        world_size: int = 1,
        rank: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Create distributed data loaders for SSL pretraining"""

    is_main = (not is_distributed) or rank == 0

    # Check available resources
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(rank).total_memory / 1024 ** 3
        if is_main:
            print(f"\n🖥️  GPU {rank} memory available: {gpu_memory:.2f} GB")

    # Get CPU count for optimal num_workers
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    if is_main:
        print(f"🖥️  CPU cores available: {cpu_count}")

    # Auto-adjust num_workers if not optimal
    if args.num_workers < min(8, cpu_count - 2):
        old_workers = args.num_workers
        args.num_workers = min(12, cpu_count - 2)  # Leave some cores for system
        if is_main:
            print(f"  📈 Increased num_workers from {old_workers} to {args.num_workers} for better throughput")

    train_files, val_files = load_data_split(args.data_split_json)

    if is_main and len(train_files) > 1000:
        print(f"\n⚠️  Large dataset detected ({len(train_files)} samples). Consider using persistent cache.")

    train_dataset, val_dataset = create_ssl_datasets(
        train_files=train_files,
        val_files=val_files,
        args=args,
        is_distributed=is_distributed,
        world_size=world_size,
        rank=rank
    )

    if is_main:
        print("\n📄 Creating distributed data loaders...")

    # Calculate optimal prefetch_factor
    prefetch_factor = getattr(args, 'prefetch_factor', 4)
    if is_main:
        print(f"  Prefetch factor: {prefetch_factor}")

    # Create distributed samplers
    if is_distributed:
        # Check if we already pre-sharded for caching
        use_distributed_sampler = not (args.cache_rate == 1.0)

        if use_distributed_sampler:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True
            )
            train_shuffle = False
            if is_main:
                print(f"  Using DistributedSampler for training")
        else:
            train_sampler = None
            train_shuffle = True
            if is_main:
                print(f"  Using pre-sharded data without DistributedSampler")

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    # Reduce workers for distributed training to avoid OOM
    dataloader_num_workers = min(args.num_workers, 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if dataloader_num_workers > 0 else False,
        prefetch_factor=prefetch_factor if dataloader_num_workers > 0 else None,
    )

    # Validation loader with optimized settings
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,  # Fixed for contrastive learning
        sampler=val_sampler,
        shuffle=False,
        num_workers=max(2, dataloader_num_workers // 2),  # Use fewer workers for validation
        pin_memory=True,
        persistent_workers=True if dataloader_num_workers > 0 else False,
        prefetch_factor=2 if dataloader_num_workers > 0 else None,  # Lower prefetch for validation
        drop_last=True,
    )

    if is_main:
        print("\n🧪 Testing data loading...")
        test_batch = next(iter(train_loader))
        test_image = test_batch['image']
        print(f"  Sample batch shape: {test_image.shape}")
        print(f"  Sample data type: {test_image.dtype}")
        print(f"  Sample value range: [{test_image.min():.3f}, {test_image.max():.3f}]")

        if torch.isnan(test_image).any():
            print("  ⚠️  WARNING: NaN detected in sample batch!")
        else:
            print("  ✓ No NaN in sample batch")

        print(f"\n✓ Distributed data loaders created successfully!")
        print(f"  World size: {world_size} GPUs")
        print(f"  Training batches per GPU: {len(train_loader)} x batch_size={args.batch_size}")
        print(f"  Total effective batch size: {args.batch_size * world_size}")
        print(f"  Validation batches: {len(val_loader)} x batch_size=2")
        print(f"  Optimization features enabled:")
        print(f"    - Distributed sampling: ✓")
        print(f"    - Persistent workers: ✓")
        print(f"    - Pin memory: ✓")
        print(f"    - Prefetching: ✓ (factor={prefetch_factor})")
        print(f"    - Multi-worker loading: ✓ ({dataloader_num_workers} workers per GPU)")

        if args.cache_rate == 1.0:
            if use_distributed_sampler:
                print(f"  ⚠️  No memory optimization: Each GPU caches 100% of data")
            else:
                print(f"  ✨ Memory optimization: Each GPU caches ~{100/world_size:.1f}% of training data")

    gc.collect()
    torch.cuda.empty_cache()

    return train_loader, val_loader


class InfiniteDataLoader:
    """Infinite data loader for SSL training"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)
        return batch

    def __len__(self):
        return len(self.data_loader)