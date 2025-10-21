"""
Build anatomical graph priors from segmentation labels
Computes soft adjacency matrices, laterality pairs, and restricted masks
Enhanced version with R_mask generation for dual-branch alignment
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import binary_dilation, generate_binary_structure
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


def load_split(json_path):
    """Load data split from JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Combine training and validation for better statistics
    files = data.get('training', []) + data.get('validation', [])
    return files


def read_label(path):
    """Read label volume from NIfTI file"""
    try:
        nii = nib.load(path)
        arr = nii.get_fdata().astype(np.int32)
        return arr
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def compute_soft_adjacency(label, num_classes=87, dilate_iter=1, foreground_only=True):
    """
    Compute soft adjacency matrix from a single label volume

    Args:
        label: 3D label array
        num_classes: Number of classes (87 for foreground-only)
        dilate_iter: Dilation iterations for adjacency detection
        foreground_only: Whether labels are 0-86 (True) or 1-87 (False)

    Returns:
        A: (num_classes, num_classes) adjacency counts
    """
    struct = generate_binary_structure(3, 1)  # 6-connectivity
    A = np.zeros((num_classes, num_classes), dtype=np.float64)

    # Determine label range
    if foreground_only:
        label_range = range(num_classes)  # 0-86
    else:
        label_range = range(1, num_classes + 1)  # 1-87

    for c_idx, c_label in enumerate(label_range):
        mask_c = (label == c_label)
        if not mask_c.any():
            continue

        # Dilate to find neighbors
        dilated = binary_dilation(mask_c, structure=struct, iterations=dilate_iter)
        boundary = dilated & (~mask_c)

        # Check what labels are in the boundary
        if boundary.any():
            boundary_labels = label[boundary]
            for d_idx, d_label in enumerate(label_range):
                if d_label == c_label:
                    continue
                contact_count = (boundary_labels == d_label).sum()
                if contact_count > 0:
                    A[c_idx, d_idx] += contact_count

    return A


def compute_symmetry_scores(label, lr_pairs, num_classes=87, foreground_only=True):
    """
    Compute symmetry scores for left-right paired structures

    Args:
        label: 3D label array
        lr_pairs: List of (left, right) label pairs

    Returns:
        symmetry_scores: Dict mapping pairs to symmetry scores
    """
    symmetry_scores = {}

    for left, right in lr_pairs:
        # Adjust for 0-based indexing if needed
        if foreground_only:
            left_mask = (label == (left - 1))
            right_mask = (label == (right - 1))
        else:
            left_mask = (label == left)
            right_mask = (label == right)

        # Compute volumes
        left_vol = left_mask.sum()
        right_vol = right_mask.sum()

        if left_vol > 0 or right_vol > 0:
            # Symmetry score based on volume ratio
            min_vol = min(left_vol, right_vol)
            max_vol = max(left_vol, right_vol)
            if max_vol > 0:
                symmetry = min_vol / max_vol
            else:
                symmetry = 0.0
        else:
            symmetry = 1.0  # Both absent = symmetric

        symmetry_scores[(left, right)] = symmetry

    return symmetry_scores


def generate_restricted_mask(num_classes, required_edges, forbidden_edges, lr_pairs, out_dir):
    """
    Generate restricted mask R for valid adjacencies

    Args:
        num_classes: Number of classes (87 for foreground-only)
        required_edges: List of required edge tuples
        forbidden_edges: List of forbidden edge tuples
        lr_pairs: List of laterality pairs
        out_dir: Output directory

    Returns:
        R_mask: (num_classes, num_classes) binary mask
    """
    # Initialize with all adjacencies allowed
    R_mask = np.ones((num_classes, num_classes), dtype=np.float32)

    # Remove diagonal (no self-adjacency)
    np.fill_diagonal(R_mask, 0)

    # Set forbidden edges to 0
    for i, j in forbidden_edges:
        if i < num_classes and j < num_classes:
            R_mask[i, j] = 0
            R_mask[j, i] = 0  # Ensure symmetry

    # Ensure required edges are 1 (overrides forbidden if conflict)
    for i, j in required_edges:
        if i < num_classes and j < num_classes:
            R_mask[i, j] = 1
            R_mask[j, i] = 1  # Ensure symmetry

    # Ensure laterality pairs can connect
    for left, right in lr_pairs:
        # Adjust for 0-based indexing if needed
        left_idx = left - 1 if left > 0 else left
        right_idx = right - 1 if right > 0 else right

        if left_idx < num_classes and right_idx < num_classes:
            R_mask[left_idx, right_idx] = 1
            R_mask[right_idx, left_idx] = 1

    # Save the mask
    mask_path = os.path.join(out_dir, "R_mask.npy")
    np.save(mask_path, R_mask)
    print(f"Saved restricted mask to: {mask_path}")

    # Generate summary statistics
    total_possible = num_classes * (num_classes - 1)  # Excluding diagonal
    allowed_edges = np.sum(R_mask)
    restricted_edges = total_possible - allowed_edges

    print(f"Restricted mask statistics:")
    print(f"  Total possible edges: {total_possible}")
    print(f"  Allowed edges: {int(allowed_edges)} ({100 * allowed_edges / total_possible:.1f}%)")
    print(f"  Restricted edges: {int(restricted_edges)} ({100 * restricted_edges / total_possible:.1f}%)")

    return R_mask


def main(args):
    """Main function to build graph priors with restricted mask"""

    print(f"Loading data split from: {args.split_json}")
    files = load_split(args.split_json)
    print(f"Found {len(files)} files")

    # Initialize accumulators
    A_sum = np.zeros((args.num_classes, args.num_classes), dtype=np.float64)
    A_count = 0

    symmetry_all = defaultdict(list)

    # Load laterality pairs if provided
    lr_pairs = []
    if args.lr_pairs_json and os.path.exists(args.lr_pairs_json):
        with open(args.lr_pairs_json, 'r') as f:
            lr_pairs = json.load(f)
        print(f"Loaded {len(lr_pairs)} laterality pairs")

    # Process each file for adjacency
    print("Computing adjacency statistics...")
    for item in tqdm(files):
        label_path = item['label']
        label = read_label(label_path)
        if label is None:
            continue

        # Compute adjacency
        A = compute_soft_adjacency(
            label,
            num_classes=args.num_classes,
            dilate_iter=args.dilate_iter,
            foreground_only=args.foreground_only
        )
        A_sum += A
        A_count += 1

        # Compute symmetry if pairs provided
        if lr_pairs:
            sym_scores = compute_symmetry_scores(
                label, lr_pairs,
                num_classes=args.num_classes,
                foreground_only=args.foreground_only
            )
            for pair, score in sym_scores.items():
                symmetry_all[pair].append(score)

    if A_count == 0:
        print("Error: No valid labels processed")
        return

    # Normalize adjacency to get probabilities
    print(f"Processed {A_count} valid labels")
    A_prior = A_sum / A_count

    # Row-normalize to get transition probabilities
    row_sums = A_prior.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    A_prior_norm = A_prior / row_sums

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Save adjacency matrix (normalized)
    adj_path = os.path.join(args.out_dir, "prior_adj.npy")
    np.save(adj_path, A_prior_norm)
    print(f"Saved adjacency matrix to: {adj_path}")

    # Save raw counts too for analysis
    adj_counts_path = os.path.join(args.out_dir, "prior_adj_counts.npy")
    np.save(adj_counts_path, A_sum)

    # Generate required/forbidden edges based on thresholds
    required_edges = []
    forbidden_edges = []

    for i in range(args.num_classes):
        for j in range(args.num_classes):
            if i == j:
                continue

            prob = A_prior_norm[i, j]

            # Required edges: high probability of adjacency
            if prob >= args.th_required:
                required_edges.append([int(i), int(j)])

            # Forbidden edges: very low probability
            if prob <= args.th_forbidden:
                forbidden_edges.append([int(i), int(j)])

    print(f"Found {len(required_edges)} required edges (threshold={args.th_required})")
    print(f"Found {len(forbidden_edges)} forbidden edges (threshold={args.th_forbidden})")

    # Save required edges
    if required_edges:
        req_path = os.path.join(args.out_dir, "prior_required.json")
        with open(req_path, 'w') as f:
            json.dump({"required": required_edges}, f, indent=2)
        print(f"Saved required edges to: {req_path}")

    # Save forbidden edges
    if forbidden_edges:
        fbd_path = os.path.join(args.out_dir, "prior_forbidden.json")
        with open(fbd_path, 'w') as f:
            json.dump({"forbidden": forbidden_edges}, f, indent=2)
        print(f"Saved forbidden edges to: {fbd_path}")

    # Generate restricted mask R
    print("\nGenerating restricted mask R...")
    R_mask = generate_restricted_mask(
        args.num_classes,
        required_edges,
        forbidden_edges,
        lr_pairs,
        args.out_dir
    )

    # Save symmetry statistics
    if symmetry_all:
        symmetry_stats = {}
        for pair, scores in symmetry_all.items():
            symmetry_stats[f"{pair[0]}_{pair[1]}"] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            }

        sym_path = os.path.join(args.out_dir, "symmetry_stats.json")
        with open(sym_path, 'w') as f:
            json.dump(symmetry_stats, f, indent=2)
        print(f"Saved symmetry statistics to: {sym_path}")

    # Generate enhanced analysis report
    report_path = os.path.join(args.out_dir, "prior_analysis.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ANATOMICAL PRIOR ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Data source: {args.split_json}\n")
        f.write(f"Number of samples: {A_count}\n")
        f.write(f"Number of classes: {args.num_classes}\n")
        f.write(f"Foreground only: {args.foreground_only}\n")
        f.write(f"Dilation iterations: {args.dilate_iter}\n")
        f.write(f"Required threshold: {args.th_required}\n")
        f.write(f"Forbidden threshold: {args.th_forbidden}\n\n")

        f.write("ADJACENCY STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Non-zero adjacencies: {(A_prior_norm > 0).sum()}\n")
        f.write(f"Max adjacency probability: {A_prior_norm.max():.4f}\n")
        f.write(f"Mean adjacency probability: {A_prior_norm[A_prior_norm > 0].mean():.4f}\n\n")

        # Restricted mask statistics
        f.write("RESTRICTED MASK STATISTICS\n")
        f.write("-" * 40 + "\n")
        total_possible = args.num_classes * (args.num_classes - 1)
        allowed_edges = np.sum(R_mask)
        restricted_ratio = 1 - (allowed_edges / total_possible)
        f.write(f"Total possible edges: {total_possible}\n")
        f.write(f"Allowed edges: {int(allowed_edges)} ({100 * (1 - restricted_ratio):.1f}%)\n")
        f.write(f"Restricted edges: {int(total_possible - allowed_edges)} ({100 * restricted_ratio:.1f}%)\n")
        f.write(f"Required edges enforced: {len(required_edges)}\n")
        f.write(f"Forbidden edges blocked: {len(forbidden_edges)}\n")
        f.write(f"Laterality pairs preserved: {len(lr_pairs)}\n\n")

        # Find most strongly connected classes
        f.write("TOP 10 STRONGEST ADJACENCIES\n")
        f.write("-" * 40 + "\n")

        # Flatten and sort adjacencies
        adj_pairs = []
        for i in range(args.num_classes):
            for j in range(args.num_classes):
                if i != j and A_prior_norm[i, j] > 0:
                    adj_pairs.append((A_prior_norm[i, j], i, j))

        adj_pairs.sort(reverse=True)
        for k, (prob, i, j) in enumerate(adj_pairs[:10]):
            # Adjust for original label IDs if needed
            if args.foreground_only:
                orig_i, orig_j = i + 1, j + 1
            else:
                orig_i, orig_j = i, j

            # Check if this edge is in restricted mask
            mask_status = "âœ“" if R_mask[i, j] > 0 else "âœ—"
            f.write(f"{k + 1}. Class {orig_i} <-> Class {orig_j}: {prob:.4f} [{mask_status}]\n")

        # Symmetry analysis
        if symmetry_stats:
            f.write("\nSYMMETRY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            for pair_str, stats in list(symmetry_stats.items())[:10]:
                f.write(f"Pair {pair_str}: mean={stats['mean']:.3f}, "
                        f"std={stats['std']:.3f}, "
                        f"range=[{stats['min']:.3f}, {stats['max']:.3f}]\n")

        # Analyze connectivity patterns
        f.write("\nCONNECTIVITY PATTERNS\n")
        f.write("-" * 40 + "\n")

        # Classes with most connections
        connection_counts = (A_prior_norm > 0.01).sum(axis=1)
        most_connected = np.argsort(connection_counts)[::-1][:5]
        f.write("Most connected classes:\n")
        for idx in most_connected:
            orig_idx = idx + 1 if args.foreground_only else idx
            f.write(f"  Class {orig_idx}: {connection_counts[idx]} connections\n")

        # Classes with fewest connections
        least_connected = np.argsort(connection_counts)[:5]
        f.write("\nLeast connected classes:\n")
        for idx in least_connected:
            orig_idx = idx + 1 if args.foreground_only else idx
            f.write(f"  Class {orig_idx}: {connection_counts[idx]} connections\n")

        # Analyze impact of restricted mask
        f.write("\nRESTRICTED MASK IMPACT\n")
        f.write("-" * 40 + "\n")

        # Count how many edges were removed per class
        edges_removed_per_class = []
        for i in range(args.num_classes):
            removed = np.sum(R_mask[i, :] == 0) - 1  # Subtract 1 for diagonal
            edges_removed_per_class.append(removed)

        max_removed_idx = np.argmax(edges_removed_per_class)
        min_removed_idx = np.argmin(edges_removed_per_class)

        orig_max = max_removed_idx + 1 if args.foreground_only else max_removed_idx
        orig_min = min_removed_idx + 1 if args.foreground_only else min_removed_idx

        f.write(f"Class with most edges removed: {orig_max} ({edges_removed_per_class[max_removed_idx]} edges)\n")
        f.write(f"Class with fewest edges removed: {orig_min} ({edges_removed_per_class[min_removed_idx]} edges)\n")
        f.write(f"Average edges removed per class: {np.mean(edges_removed_per_class):.1f}\n")

    print(f"Saved analysis report to: {report_path}")
    print("\nâœ… Prior generation complete with restricted mask!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build anatomical graph priors with restricted mask")

    parser.add_argument("--split_json", required=True, type=str,
                        help="Path to data split JSON file")
    parser.add_argument("--out_dir", required=True, type=str,
                        help="Output directory for priors")
    parser.add_argument("--num_classes", type=int, default=87,
                        help="Number of segmentation classes")
    parser.add_argument("--foreground_only", action="store_true", default=True,
                        help="Whether using foreground-only labels (0-86)")
    parser.add_argument("--dilate_iter", type=int, default=1,
                        help="Dilation iterations for adjacency detection")
    parser.add_argument("--th_required", type=float, default=0.02,
                        help="Threshold for required edges (default: 0.02, stricter)")
    parser.add_argument("--th_forbidden", type=float, default=5e-4,
                        help="Threshold for forbidden edges (default: 5e-4, stricter)")
    parser.add_argument("--lr_pairs_json", type=str, default=None,
                        help="Path to laterality pairs JSON")

    args = parser.parse_args()
    main(args)