"""
Segmentation metrics for brain parcellation evaluation
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.transforms import AsDiscrete
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class SegmentationMetrics:
    """Comprehensive metrics for multi-class segmentation evaluation

    FIXED: Properly handles foreground-only mode with ignore_index
    """

    def __init__(self,
                 num_classes: int,
                 include_background: bool = True,  # FIXED: Changed to True for foreground-only mode consistency
                 class_names: Optional[List[str]] = None,
                 device: torch.device = torch.device('cpu'),
                 ignore_index: int = -1):

        self.num_classes = num_classes
        self.include_background = include_background
        self.device = device
        self.ignore_index = ignore_index

        # Class names for reporting
        if class_names is None:
            # For foreground-only mode with 87 classes
            if num_classes == 87:
                self.class_names = [f"Region_{i+1}" for i in range(num_classes)]
            else:
                self.class_names = [f"Class_{i}" for i in range(num_classes)]
        else:
            self.class_names = class_names

        # Initialize metrics
        # FIXED: Set include_background=True for foreground-only mode
        # The ignore_index mechanism handles background exclusion
        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction="none",
            get_not_nans=False
        )

        self.hausdorff_metric = HausdorffDistanceMetric(
            include_background=include_background,
            reduction="none",
            percentile=95  # 95th percentile Hausdorff distance
        )

        # Storage for batch metrics
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.dice_scores = []
        self.hausdorff_distances = []
        self.volume_differences = []
        self.true_positives = torch.zeros(self.num_classes)
        self.false_positives = torch.zeros(self.num_classes)
        self.false_negatives = torch.zeros(self.num_classes)
        self.sample_count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with a batch of predictions

        FIXED: Properly handles ignore_index for background pixels

        Args:
            pred: (B, C, H, W, D) or (B, H, W, D) predictions
            target: (B, H, W, D) ground truth labels (may contain ignore_index)
        """
        # Create mask for valid pixels (not ignore_index)
        if self.ignore_index >= 0:
            valid_mask = (target != self.ignore_index).float()
        else:
            valid_mask = torch.ones_like(target, dtype=torch.float32)

        # Convert to one-hot manually
        if pred.dim() == 4:  # Already argmaxed (B, H, W, D)
            pred_argmax = pred
        else:  # Logits (B, C, H, W, D)
            pred_argmax = torch.argmax(pred, dim=1)

        # Create one-hot encoding for valid pixels only
        pred_onehot = torch.zeros(pred_argmax.shape[0], self.num_classes,
                                  pred_argmax.shape[1], pred_argmax.shape[2],
                                  pred_argmax.shape[3], device=self.device)
        target_onehot = torch.zeros_like(pred_onehot)

        for c in range(self.num_classes):
            pred_onehot[:, c] = (pred_argmax == c).float() * valid_mask
            target_onehot[:, c] = (target == c).float() * valid_mask

        # Calculate Dice scores
        dice_batch = self._calculate_dice(pred_onehot, target_onehot)
        self.dice_scores.append(dice_batch.cpu())

        # Calculate confusion matrix elements
        self._update_confusion_matrix(pred_onehot, target_onehot)

        # Calculate volume differences
        self._update_volume_metrics(pred_onehot, target_onehot, valid_mask)

        self.sample_count += pred_onehot.shape[0]

    def _calculate_dice(self, pred_onehot: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        """Calculate Dice score for each class"""
        # pred_onehot and target_onehot are already masked by valid pixels
        dims = (2, 3, 4)  # Spatial dimensions

        intersection = torch.sum(pred_onehot * target_onehot, dim=dims)
        pred_sum = torch.sum(pred_onehot, dim=dims)
        target_sum = torch.sum(target_onehot, dim=dims)

        # Calculate dice, but handle cases where class doesn't exist
        dice = torch.where(
            (pred_sum + target_sum) > 0,
            (2.0 * intersection + 1e-5) / (pred_sum + target_sum + 1e-5),
            torch.ones_like(intersection)  # If class doesn't exist, consider dice as 1 (will be filtered later)
        )

        return dice  # (B, C)

    def _update_confusion_matrix(self, pred_onehot: torch.Tensor, target_onehot: torch.Tensor):
        """Update confusion matrix statistics"""
        # Calculate TP, FP, FN for each class
        for c in range(self.num_classes):
            pred_c = pred_onehot[:, c]
            target_c = target_onehot[:, c]

            tp = (pred_c * target_c).sum()
            fp = (pred_c * (1 - target_c)).sum()
            fn = ((1 - pred_c) * target_c).sum()

            self.true_positives[c] += tp.cpu()
            self.false_positives[c] += fp.cpu()
            self.false_negatives[c] += fn.cpu()

    def _update_volume_metrics(self, pred_onehot: torch.Tensor, target_onehot: torch.Tensor,
                               valid_mask: torch.Tensor):
        """Calculate volume-based metrics"""
        batch_size = pred_onehot.shape[0]
        volume_diff = torch.zeros(batch_size, self.num_classes)

        for b in range(batch_size):
            for c in range(self.num_classes):
                pred_vol = pred_onehot[b, c].sum().cpu()
                target_vol = target_onehot[b, c].sum().cpu()

                if target_vol > 0:
                    volume_diff[b, c] = (pred_vol - target_vol) / target_vol * 100
                else:
                    volume_diff[b, c] = 0 if pred_vol == 0 else 100

        self.volume_differences.append(volume_diff)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        if self.sample_count == 0:
            return {}

        results = {}

        # Aggregate Dice scores
        all_dice = torch.cat(self.dice_scores, dim=0)  # (N, C)

        # Filter out classes that don't exist (dice=1 by our convention)
        valid_dice_per_class = []
        for c in range(self.num_classes):
            class_dice = all_dice[:, c]
            valid_dice = class_dice[class_dice < 1.0]  # Filter out non-existent classes
            if len(valid_dice) > 0:
                valid_dice_per_class.append(valid_dice.mean().item())
                results[f'dice_class_{c}'] = valid_dice.mean().item()

        # Overall mean Dice (only over classes that exist)
        if valid_dice_per_class:
            results['dice_mean'] = np.mean(valid_dice_per_class)
        else:
            results['dice_mean'] = 0.0

        # Calculate other metrics from confusion matrix
        for c in range(self.num_classes):
            tp = self.true_positives[c]
            fp = self.false_positives[c]
            fn = self.false_negatives[c]

            # Skip metrics for classes that don't exist
            if tp + fn == 0:  # Class doesn't exist in ground truth
                continue

            # Precision
            precision = tp / (tp + fp + 1e-7)
            results[f'precision_class_{c}'] = precision.item()

            # Recall (Sensitivity)
            recall = tp / (tp + fn + 1e-7)
            results[f'recall_class_{c}'] = recall.item()

            # F1 Score
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
            results[f'f1_class_{c}'] = f1.item()

        # Volume metrics
        if self.volume_differences:
            all_vol_diff = torch.cat(self.volume_differences, dim=0)
            mean_vol_diff = all_vol_diff.mean(dim=0)

            for i in range(self.num_classes):
                results[f'volume_diff_class_{i}'] = mean_vol_diff[i].item()

        return results

    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics as a DataFrame"""
        metrics = self.compute()

        # Organize metrics by class
        summary_data = []

        for i in range(self.num_classes):
            class_data = {
                'Class': self.class_names[i],
                'Dice': metrics.get(f'dice_class_{i}', np.nan),
                'Precision': metrics.get(f'precision_class_{i}', np.nan),
                'Recall': metrics.get(f'recall_class_{i}', np.nan),
                'F1': metrics.get(f'f1_class_{i}', np.nan),
                'Volume_Diff_%': metrics.get(f'volume_diff_class_{i}', np.nan)
            }
            summary_data.append(class_data)

        df = pd.DataFrame(summary_data)
        # Filter out classes with NaN Dice (don't exist in dataset)
        df = df.dropna(subset=['Dice'])
        return df

    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot metrics visualization"""
        df = self.get_summary()

        if df.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Dice scores by class
        ax = axes[0, 0]
        df_sorted = df.sort_values('Dice')
        ax.barh(df_sorted['Class'], df_sorted['Dice'])
        ax.set_xlabel('Dice Score')
        ax.set_title('Dice Score by Class')
        ax.set_xlim(0, 1)

        # 2. Precision vs Recall
        ax = axes[0, 1]
        ax.scatter(df['Recall'], df['Precision'], alpha=0.6, s=50)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision vs Recall')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

        # 3. F1 scores
        ax = axes[1, 0]
        df_sorted = df.sort_values('F1')
        ax.barh(df_sorted['Class'], df_sorted['F1'])
        ax.set_xlabel('F1 Score')
        ax.set_title('F1 Score by Class')
        ax.set_xlim(0, 1)

        # 4. Volume differences
        ax = axes[1, 1]
        ax.hist(df['Volume_Diff_%'].dropna(), bins=30, edgecolor='black')
        ax.set_xlabel('Volume Difference (%)')
        ax.set_ylabel('Number of Classes')
        ax.set_title('Distribution of Volume Differences')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def save_detailed_report(self, save_path: str):
        """Save detailed metrics report"""
        df = self.get_summary()
        metrics = self.compute()

        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SEGMENTATION METRICS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Overall Mean Dice: {metrics.get('dice_mean', 0):.4f}\n\n")

            f.write("Per-Class Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")

            # Worst performing classes
            f.write("Worst Performing Classes (by Dice):\n")
            f.write("-" * 40 + "\n")
            worst_classes = df.nsmallest(10, 'Dice')
            f.write(worst_classes[['Class', 'Dice']].to_string(index=False))
            f.write("\n\n")

            # Best performing classes
            f.write("Best Performing Classes (by Dice):\n")
            f.write("-" * 40 + "\n")
            best_classes = df.nlargest(10, 'Dice')
            f.write(best_classes[['Class', 'Dice']].to_string(index=False))


def calculate_metrics_batch(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        include_background: bool = True,  # FIXED: Changed to True for consistency
        ignore_index: int = -1
) -> Dict[str, torch.Tensor]:
    """Calculate metrics for a single batch

    Args:
        pred: (B, C, H, W, D) predictions (logits or probabilities)
        target: (B, H, W, D) ground truth labels
        num_classes: Number of classes
        include_background: Whether to include background in metrics
        ignore_index: Label index to ignore (e.g., background)

    Returns:
        Dictionary of metrics
    """
    # Create mask for valid pixels
    if ignore_index >= 0:
        valid_mask = (target != ignore_index).float()
    else:
        valid_mask = torch.ones_like(target, dtype=torch.float32)

    # Convert predictions to class labels
    if pred.dim() == 5:  # (B, C, H, W, D)
        pred_labels = torch.argmax(pred, dim=1)
    else:
        pred_labels = pred

    # One-hot encode with masking
    pred_onehot = torch.zeros(pred_labels.shape[0], num_classes,
                              pred_labels.shape[1], pred_labels.shape[2],
                              pred_labels.shape[3], device=pred.device)
    target_onehot = torch.zeros_like(pred_onehot)

    for c in range(num_classes):
        pred_onehot[:, c] = (pred_labels == c).float() * valid_mask
        target_onehot[:, c] = (target == c).float() * valid_mask

    # Calculate Dice
    dims = (2, 3, 4)  # Spatial dimensions
    intersection = torch.sum(pred_onehot * target_onehot, dim=dims)
    pred_sum = torch.sum(pred_onehot, dim=dims)
    target_sum = torch.sum(target_onehot, dim=dims)

    # Calculate dice only for classes that exist
    dice = torch.where(
        (pred_sum + target_sum) > 0,
        (2.0 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7),
        torch.zeros_like(intersection)  # Zero dice for non-existent classes
    )

    # Calculate mean only over existing classes
    existing_classes = (target_sum > 0).any(dim=0)  # Which classes exist in batch
    if existing_classes.any():
        dice_mean = dice[:, existing_classes].mean()
    else:
        dice_mean = torch.tensor(0.0)

    return {
        'dice': dice_mean,
        'dice_per_class': dice.mean(dim=0)
    }