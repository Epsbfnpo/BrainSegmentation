#!/usr/bin/env python3
"""
Compute class prior distribution for PPREMOPREBO dataset
Creates a JSON file with class ratios for SFDA training
"""

import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse
import os


def compute_class_prior_foreground(split_json, output_file='PPREMOPREBO_class_prior_foreground.json'):
    """
    Compute class prior for foreground-only mode (87 brain regions)
    """
    print("Computing foreground-only class prior distribution...")

    with open(split_json, 'r') as f:
        data_split = json.load(f)

    train_files = data_split['training']

    # Count voxels for each brain region (1-87)
    class_counts = np.zeros(87, dtype=np.int64)
    total_foreground_voxels = 0

    print(f"Processing {len(train_files)} training files...")

    for file_info in tqdm(train_files):
        label_path = file_info['label']

        # Load label
        label_nii = nib.load(label_path)
        label_data = label_nii.get_fdata().astype(np.int32)

        # Count each brain region (1-87)
        for region in range(1, 88):
            count = (label_data == region).sum()
            class_counts[region - 1] += count  # Map to 0-86 for array indexing
            total_foreground_voxels += count

    # Calculate ratios
    class_ratios = (class_counts / total_foreground_voxels).tolist()

    # Print statistics
    print(f"\nForeground-only class distribution:")
    print(f"Total foreground voxels: {total_foreground_voxels:,}")
    print(f"\nTop 10 most common brain regions:")

    # Sort by frequency
    sorted_indices = np.argsort(class_counts)[::-1]
    for i in range(10):
        idx = sorted_indices[i]
        brain_region = idx + 1  # Convert back to 1-87
        ratio = class_ratios[idx]
        count = class_counts[idx]
        print(f"  Brain Region {brain_region} (class {idx}): {ratio:.6f} ({ratio * 100:.3f}%, {count:,} voxels)")

    # Save to JSON
    prior_data = {
        'dataset': 'PPREMOPREBO',
        'mode': 'foreground_only',
        'num_classes': 87,
        'total_voxels': int(total_foreground_voxels),
        'class_ratios': class_ratios,
        'class_counts': class_counts.tolist(),
        'description': 'Class distribution for brain regions 1-87 (mapped to classes 0-86)'
    }

    with open(output_file, 'w') as f:
        json.dump(prior_data, f, indent=2)

    print(f"\n✓ Saved class prior to: {output_file}")

    # Additional analysis
    print("\nAdditional statistics:")
    print(f"  Most common region: Brain Region {sorted_indices[0] + 1} ({class_ratios[sorted_indices[0]] * 100:.2f}%)")
    print(
        f"  Least common region: Brain Region {sorted_indices[-1] + 1} ({class_ratios[sorted_indices[-1]] * 100:.4f}%)")
    print(f"  Entropy: {-np.sum(np.array(class_ratios) * np.log(np.array(class_ratios) + 1e-10)):.3f}")
    print(f"  Max/Min ratio: {class_ratios[sorted_indices[0]] / (class_ratios[sorted_indices[-1]] + 1e-10):.1f}")

    return class_ratios


def compute_class_prior_standard(split_json, output_file='PPREMOPREBO_class_prior_standard.json'):
    """
    Compute class prior for standard mode (88 classes including background)
    """
    print("Computing standard class prior distribution (including background)...")

    with open(split_json, 'r') as f:
        data_split = json.load(f)

    train_files = data_split['training']

    # Count voxels for each class (0-87)
    class_counts = np.zeros(88, dtype=np.int64)
    total_voxels = 0

    print(f"Processing {len(train_files)} training files...")

    for file_info in tqdm(train_files):
        label_path = file_info['label']

        # Load label
        label_nii = nib.load(label_path)
        label_data = label_nii.get_fdata().astype(np.int32)

        # Count each class (0-87)
        for class_id in range(88):
            count = (label_data == class_id).sum()
            class_counts[class_id] += count

        total_voxels += label_data.size

    # Calculate ratios
    class_ratios = (class_counts / total_voxels).tolist()

    # Print statistics
    print(f"\nStandard class distribution:")
    print(f"Total voxels: {total_voxels:,}")
    print(f"Background ratio: {class_ratios[0]:.4f} ({class_ratios[0] * 100:.2f}%)")
    print(f"Foreground ratio: {1 - class_ratios[0]:.4f} ({(1 - class_ratios[0]) * 100:.2f}%)")

    print(f"\nTop 10 classes:")
    sorted_indices = np.argsort(class_counts)[::-1]
    for i in range(10):
        idx = sorted_indices[i]
        ratio = class_ratios[idx]
        count = class_counts[idx]
        if idx == 0:
            print(f"  Background: {ratio:.6f} ({ratio * 100:.3f}%, {count:,} voxels)")
        else:
            print(f"  Brain Region {idx}: {ratio:.6f} ({ratio * 100:.3f}%, {count:,} voxels)")

    # Save to JSON
    prior_data = {
        'dataset': 'PPREMOPREBO',
        'mode': 'standard',
        'num_classes': 88,
        'total_voxels': int(total_voxels),
        'background_ratio': class_ratios[0],
        'class_ratios': class_ratios,
        'class_counts': class_counts.tolist(),
        'description': 'Class distribution for all 88 classes (0=background, 1-87=brain regions)'
    }

    with open(output_file, 'w') as f:
        json.dump(prior_data, f, indent=2)

    print(f"\n✓ Saved class prior to: {output_file}")

    return class_ratios


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute class prior distribution')
    parser.add_argument('--split_json', type=str, required=True,
                        help='Path to data split JSON file')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['foreground', 'standard', 'both'],
                        help='Which prior to compute')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for JSON files')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in ['foreground', 'both']:
        output_file = os.path.join(args.output_dir, 'PPREMOPREBO_class_prior_foreground.json')
        compute_class_prior_foreground(args.split_json, output_file)

    if args.mode in ['standard', 'both']:
        output_file = os.path.join(args.output_dir, 'PPREMOPREBO_class_prior_standard.json')
        compute_class_prior_standard(args.split_json, output_file)

    print("\n✅ Done!")