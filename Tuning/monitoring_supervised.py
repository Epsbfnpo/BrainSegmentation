"""
Monitoring utilities for supervised fine-tuning
Tracks loss, dice scores, and per-region performance
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from collections import defaultdict
import seaborn as sns


class SupervisedMonitor:
    """Monitor supervised training progress and metrics"""

    def __init__(self, results_dir: str, num_classes: int = 87):
        self.results_dir = results_dir
        self.monitor_dir = os.path.join(results_dir, 'monitoring')
        os.makedirs(self.monitor_dir, exist_ok=True)
        self.num_classes = num_classes

        # Initialize history
        self.history = defaultdict(list)
        self.best_metrics = {
            'best_val_dice': 0.0,
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }

        # File paths
        self.json_path = os.path.join(self.monitor_dir, 'training_history.json')
        self.csv_path = os.path.join(self.monitor_dir, 'training_history.csv')
        self.report_path = os.path.join(self.monitor_dir, 'training_report.txt')
        self.per_class_csv_path = os.path.join(self.monitor_dir, 'per_class_dice.csv')

    def update(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict] = None):
        """Update monitoring with new metrics"""

        # Basic info
        current_len = len(self.history['epoch'])
        self.history['epoch'].append(epoch)
        self.history['timestamp'].append(datetime.now().isoformat())

        # Training metrics
        for key, value in train_metrics.items():
            if key == 'dice_per_class':
                # Handle per-class dice separately
                continue
            self.history[f'train_{key}'].append(value)

        # Ensure all train_* columns have same length
        current_len_after = len(self.history['epoch'])
        for key in list(self.history.keys()):
            if key.startswith('train_') and len(self.history[key]) < current_len_after:
                self.history[key].append(None)

        # Validation metrics
        if val_metrics:
            for key, value in val_metrics.items():
                if key == 'dice_per_class':
                    # Handle per-class dice
                    for class_name, dice_score in value.items():
                        self.history[f'val_dice_{class_name}'].append(dice_score)
                else:
                    self.history[f'val_{key}'].append(value)

            # Update best metrics
            if val_metrics['dice'] > self.best_metrics['best_val_dice']:
                self.best_metrics['best_val_dice'] = val_metrics['dice']
                self.best_metrics['best_epoch'] = epoch
            if val_metrics.get('total_loss', val_metrics.get('loss', float('inf'))) < self.best_metrics['best_val_loss']:
                self.best_metrics['best_val_loss'] = val_metrics.get('total_loss', val_metrics.get('loss'))
        else:
            # Fill validation metrics with None
            for key in self.history:
                if key.startswith('val_') and len(self.history[key]) < current_len_after:
                    self.history[key].append(None)

        # Ensure all columns have same length
        for key, lst in self.history.items():
            if len(lst) < current_len_after:
                lst.extend([None] * (current_len_after - len(lst)))

        # Save history
        self.save_history()

    def save_history(self):
        """Save training history to files"""

        # Save to JSON
        with open(self.json_path, 'w') as f:
            json.dump({
                'history': dict(self.history),
                'best_metrics': self.best_metrics
            }, f, indent=2)

        # Save main metrics to CSV
        if self.history['epoch']:
            # Extract main metrics
            main_metrics = {k: v for k, v in self.history.items()
                          if not k.startswith('val_dice_region_')}
            df_main = pd.DataFrame(main_metrics)
            df_main.to_csv(self.csv_path, index=False)

            # Save per-class dice scores to separate CSV
            self.save_per_class_dice()

    def save_per_class_dice(self):
        """Save per-class dice scores to separate CSV"""
        per_class_data = {}
        per_class_data['epoch'] = self.history['epoch']

        # Extract all per-class dice scores
        for key in self.history:
            if key.startswith('val_dice_region_'):
                per_class_data[key.replace('val_dice_', '')] = self.history[key]

        if len(per_class_data) > 1:  # More than just epoch
            df_per_class = pd.DataFrame(per_class_data)
            df_per_class.to_csv(self.per_class_csv_path, index=False)

    def get_history(self) -> Dict:
        """Get the training history"""
        return dict(self.history)

    def _format_value(self, value, format_str=".4f"):
        """Safely format a value that might be None"""
        if value is None:
            return "N/A"
        try:
            return f"{value:{format_str}}"
        except (TypeError, ValueError):
            return str(value)

    def _get_latest_non_none(self, key, default=None):
        """Get the latest non-None value from history"""
        if key not in self.history or not self.history[key]:
            return default

        # Search backwards for the first non-None value
        for value in reversed(self.history[key]):
            if value is not None:
                return value
        return default

    def generate_report(self):
        """Generate comprehensive text report of training progress"""

        with open(self.report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SUPERVISED FINE-TUNING REPORT (FOREGROUND-ONLY)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            if not self.history['epoch']:
                f.write("No training data available yet.\n")
                return

            # Training summary
            f.write("TRAINING SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total epochs: {len(self.history['epoch'])}\n")
            f.write(f"Current epoch: {self.history['epoch'][-1]}\n")
            f.write(f"Number of brain regions: {self.num_classes}\n")
            f.write(f"Best validation Dice: {self._format_value(self.best_metrics['best_val_dice'])} ")
            f.write(f"(epoch {self.best_metrics['best_epoch']})\n")
            f.write(f"Best validation loss: {self._format_value(self.best_metrics['best_val_loss'])}\n")

            if 'val_dice' in self.history:
                latest_val_dice = self._get_latest_non_none('val_dice')
                if latest_val_dice is not None:
                    f.write(f"Current validation Dice: {self._format_value(latest_val_dice)}\n")

            # Latest metrics
            f.write("\nLATEST METRICS\n")
            f.write("-" * 40 + "\n")

            # Training metrics
            f.write("Training:\n")
            train_loss = self._get_latest_non_none('train_loss')
            if train_loss is not None:
                f.write(f"  Loss: {self._format_value(train_loss)}\n")

            train_dice = self._get_latest_non_none('train_dice')
            if train_dice is not None:
                f.write(f"  Dice: {self._format_value(train_dice)}\n")

            train_lr = self._get_latest_non_none('train_lr')
            if train_lr is not None:
                f.write(f"  Learning rate: {self._format_value(train_lr, '.6f')}\n")

            train_grad = self._get_latest_non_none('train_grad_norm')
            if train_grad is not None:
                f.write(f"  Gradient norm: {self._format_value(train_grad)}\n")

            # Validation metrics
            val_loss = self._get_latest_non_none('val_loss')
            val_dice = self._get_latest_non_none('val_dice')

            if val_loss is not None or val_dice is not None:
                f.write("\nValidation:\n")
                if val_loss is not None:
                    f.write(f"  Loss: {self._format_value(val_loss)}\n")
                if val_dice is not None:
                    f.write(f"  Dice: {self._format_value(val_dice)}\n")

            # Per-region analysis
            f.write("\nPER-REGION ANALYSIS (Latest Epoch)\n")
            f.write("-" * 40 + "\n")

            # Get latest per-region dice scores
            latest_per_region = {}
            for key in self.history:
                if key.startswith('val_dice_region_') and self.history[key]:
                    latest_value = self._get_latest_non_none(key)
                    if latest_value is not None:
                        region_name = key.replace('val_dice_', '')
                        latest_per_region[region_name] = latest_value

            if latest_per_region:
                # Sort by performance
                sorted_regions = sorted(latest_per_region.items(), key=lambda x: x[1])

                # Worst performing regions
                f.write("\nWorst 10 performing regions:\n")
                for region, dice in sorted_regions[:10]:
                    f.write(f"  {region}: {self._format_value(dice)}\n")

                # Best performing regions
                f.write("\nBest 10 performing regions:\n")
                for region, dice in sorted_regions[-10:]:
                    f.write(f"  {region}: {self._format_value(dice)}\n")

                # Statistics
                dice_values = list(latest_per_region.values())
                f.write(f"\nRegion statistics:\n")
                f.write(f"  Mean dice: {self._format_value(np.mean(dice_values))}\n")
                f.write(f"  Std dice: {self._format_value(np.std(dice_values))}\n")
                f.write(f"  Min dice: {self._format_value(np.min(dice_values))}\n")
                f.write(f"  Max dice: {self._format_value(np.max(dice_values))}\n")
                f.write(f"  Number of regions with dice > 0.8: {sum(1 for d in dice_values if d > 0.8)}\n")
                f.write(f"  Number of regions with dice < 0.1: {sum(1 for d in dice_values if d < 0.1)}\n")

                # Check for zero dice regions
                zero_regions = [region for region, dice in latest_per_region.items() if dice == 0.0]
                if zero_regions:
                    f.write(f"\n⚠️  Regions with zero dice: {zero_regions}\n")

            # Training dynamics
            f.write("\nTRAINING DYNAMICS\n")
            f.write("-" * 40 + "\n")

            if len(self.history['epoch']) > 1:
                # Calculate trends
                if 'train_loss' in self.history:
                    train_losses = [x for x in self.history['train_loss'] if x is not None]
                    if len(train_losses) >= 10:
                        recent_trend = np.polyfit(range(10), train_losses[-10:], 1)[0]
                        f.write(f"Recent training loss trend: {'↓' if recent_trend < 0 else '↑'} ")
                        f.write(f"({self._format_value(recent_trend, '.6f')} per epoch)\n")

                if 'val_dice' in self.history:
                    val_dice = [x for x in self.history['val_dice'] if x is not None]
                    if len(val_dice) >= 5:
                        recent_trend = np.polyfit(range(5), val_dice[-5:], 1)[0]
                        f.write(f"Recent validation dice trend: {'↑' if recent_trend > 0 else '↓'} ")
                        f.write(f"({self._format_value(recent_trend, '.6f')} per epoch)\n")

        print(f"📊 Report saved to: {self.report_path}")

    def plot_metrics(self):
        """Generate comprehensive plots for training metrics"""

        if not self.history['epoch']:
            return

        epochs = self.history['epoch']

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))

        # 1. Loss curves
        ax1 = plt.subplot(3, 3, 1)
        if 'train_loss' in self.history:
            train_losses = self.history['train_loss']
            # Plot only non-None values
            valid_epochs = [e for e, v in zip(epochs, train_losses) if v is not None]
            valid_losses = [v for v in train_losses if v is not None]
            if valid_losses:
                ax1.plot(valid_epochs, valid_losses, 'b-', label='Train', alpha=0.7, linewidth=2)

        if 'val_loss' in self.history:
            val_epochs = [e for e, v in zip(epochs, self.history['val_loss']) if v is not None]
            val_losses = [v for v in self.history['val_loss'] if v is not None]
            if val_losses:
                ax1.plot(val_epochs, val_losses, 'r-', label='Val', marker='o', markersize=8)

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Dice scores
        ax2 = plt.subplot(3, 3, 2)
        if 'train_dice' in self.history:
            train_dice = self.history['train_dice']
            valid_epochs = [e for e, v in zip(epochs, train_dice) if v is not None]
            valid_dice = [v for v in train_dice if v is not None]
            if valid_dice:
                ax2.plot(valid_epochs, valid_dice, 'b-', label='Train', alpha=0.7, linewidth=2)

        if 'val_dice' in self.history:
            val_epochs = [e for e, v in zip(epochs, self.history['val_dice']) if v is not None]
            val_dice = [v for v in self.history['val_dice'] if v is not None]
            if val_dice:
                ax2.plot(val_epochs, val_dice, 'r-', label='Val', marker='o', markersize=8)
                # Mark best epoch
                if self.best_metrics['best_epoch'] > 0 and self.best_metrics['best_epoch'] in val_epochs:
                    best_idx = val_epochs.index(self.best_metrics['best_epoch'])
                    ax2.plot(self.best_metrics['best_epoch'], val_dice[best_idx],
                           'g*', markersize=15, label=f'Best: {val_dice[best_idx]:.4f}')

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.set_title('Training and Validation Dice Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # 3. Learning rate
        ax3 = plt.subplot(3, 3, 3)
        if 'train_lr' in self.history:
            lr_values = self.history['train_lr']
            valid_epochs = [e for e, v in zip(epochs, lr_values) if v is not None]
            valid_lr = [v for v in lr_values if v is not None]
            if valid_lr:
                ax3.plot(valid_epochs, valid_lr, 'g-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        # 4. Gradient norm
        ax4 = plt.subplot(3, 3, 4)
        if 'train_grad_norm' in self.history:
            grad_norms = self.history['train_grad_norm']
            valid_epochs = [e for e, v in zip(epochs, grad_norms) if v is not None]
            valid_grads = [v for v in grad_norms if v is not None]
            if valid_grads:
                ax4.plot(valid_epochs, valid_grads, 'purple', alpha=0.7)
                # Add moving average
                if len(valid_grads) >= 5:
                    window = min(10, len(valid_grads) // 5)
                    grad_array = np.array(valid_grads)
                    ma = np.convolve(grad_array, np.ones(window) / window, mode='valid')
                    ma_epochs = valid_epochs[window - 1:]
                    ax4.plot(ma_epochs, ma, 'darkviolet', linewidth=2, label=f'{window}-epoch MA')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Gradient Norm')
        ax4.set_title('Gradient Norm Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Per-region dice distribution (latest epoch)
        ax5 = plt.subplot(3, 3, 5)
        latest_per_region = {}
        for key in self.history:
            if key.startswith('val_dice_region_') and self.history[key]:
                latest_value = self._get_latest_non_none(key)
                if latest_value is not None:
                    region_num = int(key.split('_')[-1])
                    latest_per_region[region_num] = latest_value

        if latest_per_region:
            regions = sorted(latest_per_region.keys())
            dice_values = [latest_per_region[r] for r in regions]
            colors = ['red' if v == 0.0 else 'orange' if v < 0.5 else 'skyblue' for v in dice_values]
            bars = ax5.bar(regions, dice_values, color=colors, edgecolor='navy', alpha=0.7)

            # Add value labels for zero dice regions
            for i, (region, dice) in enumerate(zip(regions, dice_values)):
                if dice == 0.0:
                    ax5.text(i, 0.02, f'{region}', ha='center', va='bottom', fontsize=8, rotation=45)

            # Calculate mean excluding zeros
            non_zero_dice = [v for v in dice_values if v > 0]
            if non_zero_dice:
                ax5.axhline(y=np.mean(non_zero_dice), color='g', linestyle='--',
                           label=f'Mean (non-zero): {np.mean(non_zero_dice):.3f}')

            # Overall mean
            ax5.axhline(y=np.mean(dice_values), color='r', linestyle='--',
                       label=f'Mean (all): {np.mean(dice_values):.3f}')

            ax5.set_xlabel('Brain Region')
            ax5.set_ylabel('Dice Score')
            ax5.set_title(f'Per-Region Dice Scores (Latest) - {len([v for v in dice_values if v == 0])} regions with zero dice')
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')
            ax5.set_ylim(0, 1)

        # 6. Dice score evolution for selected regions
        ax6 = plt.subplot(3, 3, 6)
        # Plot worst and best performing regions over time
        if latest_per_region:
            sorted_regions = sorted(latest_per_region.items(), key=lambda x: x[1])

            # Plot worst 3 regions
            for region, _ in sorted_regions[:3]:
                key = f'val_dice_region_{region}'
                if key in self.history:
                    val_epochs = [e for e, v in zip(epochs, self.history[key]) if v is not None]
                    val_scores = [v for v in self.history[key] if v is not None]
                    if val_scores:
                        ax6.plot(val_epochs, val_scores, '--', label=f'Region {region}', alpha=0.7)

            # Plot best 3 regions
            for region, _ in sorted_regions[-3:]:
                key = f'val_dice_region_{region}'
                if key in self.history:
                    val_epochs = [e for e, v in zip(epochs, self.history[key]) if v is not None]
                    val_scores = [v for v in self.history[key] if v is not None]
                    if val_scores:
                        ax6.plot(val_epochs, val_scores, '-', label=f'Region {region}', linewidth=2)

        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Dice Score')
        ax6.set_title('Dice Evolution: Worst & Best Regions')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1)

        # 7. Loss vs Dice correlation
        ax7 = plt.subplot(3, 3, 7)
        if 'val_loss' in self.history and 'val_dice' in self.history:
            # Get paired non-None values
            paired_data = [(l, d) for l, d in zip(self.history['val_loss'], self.history['val_dice'])
                          if l is not None and d is not None]
            if paired_data:
                val_losses, val_dice = zip(*paired_data)
                ax7.scatter(val_losses, val_dice, alpha=0.6, s=50)
                # Add trend line
                if len(val_losses) > 1:
                    z = np.polyfit(val_losses, val_dice, 1)
                    p = np.poly1d(z)
                    ax7.plot(sorted(val_losses), p(sorted(val_losses)), "r--", alpha=0.8)
                ax7.set_xlabel('Validation Loss')
                ax7.set_ylabel('Validation Dice')
                ax7.set_title('Loss vs Dice Correlation')
                ax7.grid(True, alpha=0.3)

        # 8. Heatmap of region dice scores over time
        ax8 = plt.subplot(3, 3, 8)
        # Create matrix of dice scores
        dice_matrix = []
        region_numbers = []
        for i in range(1, self.num_classes + 1):
            key = f'val_dice_region_{i}'
            if key in self.history and any(v is not None for v in self.history[key]):
                dice_matrix.append(self.history[key])
                region_numbers.append(i)

        if dice_matrix:
            # Replace None with NaN for visualization
            dice_matrix_np = []
            for row in dice_matrix:
                row_np = [float(v) if v is not None else np.nan for v in row]
                dice_matrix_np.append(row_np)
            dice_matrix_np = np.array(dice_matrix_np)

            # Only show every nth epoch for readability
            step = max(1, len(epochs) // 20)

            im = ax8.imshow(dice_matrix_np[:, ::step], aspect='auto', cmap='RdYlGn',
                          vmin=0, vmax=1, interpolation='nearest')
            ax8.set_xlabel('Epoch')
            ax8.set_ylabel('Brain Region')
            ax8.set_title('Region Dice Scores Over Time')
            ax8.set_xticks(range(0, len(epochs[::step]), max(1, len(epochs[::step])//10)))
            ax8.set_xticklabels(epochs[::step][::max(1, len(epochs[::step])//10)])

            # Only show some y-labels for readability
            y_step = max(1, len(region_numbers) // 20)
            ax8.set_yticks(range(0, len(region_numbers), y_step))
            ax8.set_yticklabels(region_numbers[::y_step])

            # Add colorbar
            plt.colorbar(im, ax=ax8, label='Dice Score')

        # 9. Training efficiency
        ax9 = plt.subplot(3, 3, 9)
        if 'train_dice' in self.history and 'val_dice' in self.history:
            # Get paired non-None values
            paired_epochs = []
            overfitting = []

            for e, (t, v) in enumerate(zip(self.history['train_dice'], self.history['val_dice'])):
                if t is not None and v is not None:
                    paired_epochs.append(epochs[e])
                    overfitting.append(t - v)

            if overfitting:
                ax9.plot(paired_epochs, overfitting, 'o-', color='orange', linewidth=2)
                ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax9.fill_between(paired_epochs, 0, overfitting, where=np.array(overfitting) > 0,
                               alpha=0.3, color='red', label='Overfitting')
                ax9.fill_between(paired_epochs, 0, overfitting, where=np.array(overfitting) <= 0,
                               alpha=0.3, color='green', label='Underfitting')
                ax9.set_xlabel('Epoch')
                ax9.set_ylabel('Train Dice - Val Dice')
                ax9.set_title('Overfitting Analysis')
                ax9.legend()
                ax9.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.monitor_dir, 'training_metrics.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📈 Plots saved to: {plot_path}")

        # Generate additional detailed plots
        self._plot_region_analysis()

    def _plot_region_analysis(self):
        """Create detailed region analysis plots"""

        # Get latest per-region dice scores
        latest_per_region = {}
        for key in self.history:
            if key.startswith('val_dice_region_') and self.history[key]:
                latest_value = self._get_latest_non_none(key)
                if latest_value is not None:
                    region_num = int(key.split('_')[-1])
                    latest_per_region[region_num] = latest_value

        if not latest_per_region:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Region performance distribution
        ax = axes[0, 0]
        dice_values = list(latest_per_region.values())
        valid_dice = [v for v in dice_values if v > 0]  # Exclude zeros for better histogram

        if valid_dice:
            ax.hist(valid_dice, bins=20, color='skyblue', edgecolor='navy', alpha=0.7)
            ax.axvline(np.mean(valid_dice), color='red', linestyle='--',
                      label=f'Mean (non-zero): {np.mean(valid_dice):.3f}')
            ax.axvline(np.median(valid_dice), color='green', linestyle='--',
                      label=f'Median (non-zero): {np.median(valid_dice):.3f}')

        # Add text about zero dice regions
        zero_count = len([v for v in dice_values if v == 0])
        if zero_count > 0:
            ax.text(0.05, 0.95, f'{zero_count} regions with zero dice',
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

        ax.set_xlabel('Dice Score')
        ax.set_ylabel('Number of Regions')
        ax.set_title(f'Distribution of Region Dice Scores ({self.num_classes} total regions)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Improvement over time
        ax = axes[0, 1]
        if len(self.history['epoch']) > 1:
            improvements = {}
            for region in latest_per_region:
                key = f'val_dice_region_{region}'
                if key in self.history:
                    # Get first and last non-None values
                    first_value = None
                    last_value = None

                    for v in self.history[key]:
                        if v is not None and first_value is None:
                            first_value = v
                            break

                    last_value = self._get_latest_non_none(key)

                    if first_value is not None and last_value is not None:
                        improvement = last_value - first_value
                        improvements[region] = improvement

            if improvements:
                sorted_imp = sorted(improvements.items(), key=lambda x: x[1])
                regions = [x[0] for x in sorted_imp]
                imps = [x[1] for x in sorted_imp]
                colors = ['red' if x < 0 else 'green' for x in imps]

                ax.bar(range(len(regions)), imps, color=colors, alpha=0.7)
                ax.set_xlabel('Region (sorted by improvement)')
                ax.set_ylabel('Dice Score Change')
                ax.set_title('Region Performance Change (First → Last Epoch)')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax.grid(True, alpha=0.3, axis='y')

        # 3. Consistency analysis
        ax = axes[1, 0]
        stds = {}
        for region in latest_per_region:
            key = f'val_dice_region_{region}'
            if key in self.history:
                scores = [v for v in self.history[key] if v is not None]
                if len(scores) >= 3:
                    stds[region] = np.std(scores)

        if stds:
            sorted_stds = sorted(stds.items(), key=lambda x: x[1], reverse=True)
            top_inconsistent = sorted_stds[:10]
            regions = [f'R{x[0]}' for x in top_inconsistent]
            std_values = [x[1] for x in top_inconsistent]

            ax.barh(regions, std_values, color='coral')
            ax.set_xlabel('Standard Deviation of Dice Score')
            ax.set_ylabel('Region')
            ax.set_title('Top 10 Most Inconsistent Regions')
            ax.grid(True, alpha=0.3, axis='x')

        # 4. Final performance ranking
        ax = axes[1, 1]
        sorted_regions = sorted(latest_per_region.items(), key=lambda x: x[1])

        # Show bottom 10 and top 10
        bottom_10 = sorted_regions[:10]
        top_10 = sorted_regions[-10:]

        regions = [f'R{x[0]}' for x in bottom_10] + ['...'] + [f'R{x[0]}' for x in top_10]
        scores = [x[1] for x in bottom_10] + [0.5] + [x[1] for x in top_10]
        colors = ['red' if x < 0.1 else 'orange' for x in scores[:10]] + ['white'] + ['green' for _ in scores[-10:]]

        y_pos = np.arange(len(regions))
        ax.barh(y_pos, scores, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(regions)
        ax.set_xlabel('Dice Score')
        ax.set_title('Bottom 10 & Top 10 Regions')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 1)

        plt.tight_layout()
        plot_path = os.path.join(self.monitor_dir, 'region_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Region analysis plots saved to: {plot_path}")