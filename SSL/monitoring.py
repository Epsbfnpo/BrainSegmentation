"""
Monitoring utilities for SSL pretraining
DISTRIBUTED VERSION with support for multi-GPU training
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from collections import defaultdict
import torch.distributed as dist


def is_dist():
    """Check if distributed training is initialized"""
    return dist.is_initialized()


class SSLMonitor:
    """Monitor SSL training progress and metrics (single GPU version)"""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.monitor_dir = os.path.join(results_dir, 'monitoring')
        os.makedirs(self.monitor_dir, exist_ok=True)

        # Initialize history
        self.history = defaultdict(list)
        self.best_metrics = {
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }

        # File paths
        self.json_path = os.path.join(self.monitor_dir, 'training_history.json')
        self.csv_path = os.path.join(self.monitor_dir, 'training_history.csv')
        self.report_path = os.path.join(self.monitor_dir, 'training_report.txt')

    def update(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict] = None):
        """Update monitoring with new metrics"""

        # ---------- 基本信息 ----------
        self.history['epoch'].append(epoch)
        self.history['timestamp'].append(datetime.now().isoformat())

        # ---------- 训练指标 ----------
        for key, value in train_metrics.items():
            self.history.setdefault(f'train_{key}', []).append(value)

        # ⚠️ 如果某些已有 train_* 列这次没收到值，补 None
        current_len = len(self.history['epoch'])
        for key in list(self.history.keys()):
            if key.startswith('train_') and len(self.history[key]) < current_len:
                self.history[key].append(None)

        # ---------- 验证指标 ----------
        if val_metrics:
            for key, value in val_metrics.items():
                self.history.setdefault(f'val_{key}', []).append(value)

            # 更新 best
            if val_metrics['total_loss'] < self.best_metrics['best_val_loss']:
                self.best_metrics['best_val_loss'] = val_metrics['total_loss']
                self.best_metrics['best_epoch'] = epoch
        else:
            # ⚠️ 给所有已存在的 val_* 列补 None
            for key in self.history:
                if key.startswith('val_') and len(self.history[key]) < current_len:
                    self.history[key].append(None)

        # ---------- 终极保险：全局等长 ----------
        for key, lst in self.history.items():
            if len(lst) < current_len:
                lst.extend([None] * (current_len - len(lst)))

        # ---------- 保存 ----------
        self.save_history()

    def save_history(self):
        """Save training history to files"""

        # Save to JSON
        with open(self.json_path, 'w') as f:
            json.dump({
                'history': dict(self.history),
                'best_metrics': self.best_metrics
            }, f, indent=2)

        # Save to CSV
        if self.history['epoch']:
            df = pd.DataFrame(dict(self.history))
            df.to_csv(self.csv_path, index=False)

    def get_history(self) -> Dict:
        """Get the training history"""
        return dict(self.history)

    def generate_report(self):
        """Generate text report of training progress"""

        with open(self.report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SSL PRETRAINING REPORT\n")
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

            if 'val_total_loss' in self.history and self.history['val_total_loss']:
                f.write(f"Best validation loss: {self.best_metrics['best_val_loss']:.4f} ")
                f.write(f"(epoch {self.best_metrics['best_epoch']})\n")
                f.write(f"Current validation loss: {self.history['val_total_loss'][-1]:.4f}\n")

            # Latest metrics
            f.write("\nLATEST METRICS\n")
            f.write("-" * 40 + "\n")

            # Training metrics
            f.write("Training:\n")
            if 'train_total_loss' in self.history:
                f.write(f"  Total loss: {self.history['train_total_loss'][-1]:.4f}\n")
            if 'train_inpainting_loss' in self.history:
                f.write(f"  Inpainting loss: {self.history['train_inpainting_loss'][-1]:.4f}\n")
            if 'train_rotation_loss' in self.history:
                f.write(f"  Rotation loss: {self.history['train_rotation_loss'][-1]:.4f}\n")
            if 'train_contrastive_loss' in self.history:
                f.write(f"  Contrastive loss: {self.history['train_contrastive_loss'][-1]:.4f}\n")

            # Validation metrics
            if 'val_total_loss' in self.history and self.history['val_total_loss']:
                f.write("\nValidation:\n")
                f.write(f"  Total loss: {self.history['val_total_loss'][-1]:.4f}\n")
                f.write(f"  Inpainting loss: {self.history['val_inpainting_loss'][-1]:.4f}\n")
                f.write(f"  Rotation loss: {self.history['val_rotation_loss'][-1]:.4f}\n")
                f.write(f"  Contrastive loss: {self.history['val_contrastive_loss'][-1]:.4f}\n")

            # Training dynamics
            f.write("\nTRAINING DYNAMICS\n")
            f.write("-" * 40 + "\n")

            if len(self.history['epoch']) > 1:
                # Calculate trends
                train_losses = np.array(self.history['train_total_loss'])
                if len(train_losses) >= 10:
                    recent_trend = np.polyfit(range(10), train_losses[-10:], 1)[0]
                    f.write(f"Recent training loss trend: {'↓' if recent_trend < 0 else '↑'} ")
                    f.write(f"({recent_trend:.6f} per epoch)\n")

                # Learning rate
                if 'train_lr' in self.history:
                    f.write(f"Current learning rate: {self.history['train_lr'][-1]:.6f}\n")

                # Gradient norm
                if 'train_grad_norm' in self.history:
                    grad_norms = self.history['train_grad_norm']
                    f.write(f"Average gradient norm: {np.mean(grad_norms[-10:]):.4f}\n")
                    f.write(f"Max gradient norm: {np.max(grad_norms):.4f}\n")

        print(f"📊 Report saved to: {self.report_path}")

    def plot_metrics(self):
        """Generate plots for training metrics"""

        if not self.history['epoch']:
            return

        epochs = self.history['epoch']

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))

        # 1. Total loss
        ax1 = plt.subplot(2, 2, 1)
        if 'train_total_loss' in self.history:
            ax1.plot(epochs, self.history['train_total_loss'], 'b-', label='Train', alpha=0.7)
        if 'val_total_loss' in self.history and self.history['val_total_loss']:
            val_epochs = [e for e, v in zip(epochs, self.history['val_total_loss']) if v is not None]
            val_losses = [v for v in self.history['val_total_loss'] if v is not None]
            ax1.plot(val_epochs, val_losses, 'r-', label='Val', marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Total SSL Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Component losses
        ax2 = plt.subplot(2, 2, 2)
        if 'train_inpainting_loss' in self.history:
            ax2.plot(epochs, self.history['train_inpainting_loss'], label='Inpainting')
        if 'train_rotation_loss' in self.history:
            ax2.plot(epochs, self.history['train_rotation_loss'], label='Rotation')
        if 'train_contrastive_loss' in self.history:
            ax2.plot(epochs, self.history['train_contrastive_loss'], label='Contrastive')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('SSL Component Losses (Training)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Learning rate
        ax3 = plt.subplot(2, 2, 3)
        if 'train_lr' in self.history:
            ax3.plot(epochs, self.history['train_lr'], 'g-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        # 4. Gradient norm
        ax4 = plt.subplot(2, 2, 4)
        if 'train_grad_norm' in self.history:
            ax4.plot(epochs, self.history['train_grad_norm'], 'purple', alpha=0.7)
            # Add moving average
            if len(epochs) >= 5:
                window = min(10, len(epochs) // 5)
                grad_norms = np.array(self.history['train_grad_norm'])
                ma = np.convolve(grad_norms, np.ones(window) / window, mode='valid')
                ma_epochs = epochs[window - 1:]
                ax4.plot(ma_epochs, ma, 'darkviolet', linewidth=2, label=f'{window}-epoch MA')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Gradient Norm')
        ax4.set_title('Gradient Norm Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.monitor_dir, 'training_metrics.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📈 Plots saved to: {plot_path}")

        # Additional plot for validation metrics if available
        if 'val_total_loss' in self.history and any(v is not None for v in self.history['val_total_loss']):
            self._plot_validation_details()

    def _plot_validation_details(self):
        """Plot detailed validation metrics"""

        fig = plt.figure(figsize=(12, 8))

        # Get validation epochs
        val_data = []
        for i, epoch in enumerate(self.history['epoch']):
            if i < len(self.history.get('val_total_loss', [])) and self.history['val_total_loss'][i] is not None:
                val_data.append({
                    'epoch': epoch,
                    'total': self.history['val_total_loss'][i],
                    'inpainting': self.history.get('val_inpainting_loss', [None])[i],
                    'rotation': self.history.get('val_rotation_loss', [None])[i],
                    'contrastive': self.history.get('val_contrastive_loss', [None])[i],
                })

        if not val_data:
            return

        val_df = pd.DataFrame(val_data)

        # Plot validation component losses
        ax = plt.subplot(1, 1, 1)
        ax.plot(val_df['epoch'], val_df['total'], 'k-', linewidth=2, label='Total', marker='o')
        if 'inpainting' in val_df.columns:
            ax.plot(val_df['epoch'], val_df['inpainting'], '--', label='Inpainting', marker='s')
        if 'rotation' in val_df.columns:
            ax.plot(val_df['epoch'], val_df['rotation'], '--', label='Rotation', marker='^')
        if 'contrastive' in val_df.columns:
            ax.plot(val_df['epoch'], val_df['contrastive'], '--', label='Contrastive', marker='v')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.monitor_dir, 'validation_details.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


class SSLMonitorDistributed(SSLMonitor):
    """Monitor SSL training progress for distributed training (only on main process)"""

    def __init__(self, results_dir: str):
        # Only create directories and files on main process
        is_main = not is_dist() or dist.get_rank() == 0

        if is_main:
            super().__init__(results_dir)
            # Add distributed training info to history
            if is_dist():
                self.history['world_size'] = [dist.get_world_size()]
        else:
            # Dummy initialization for non-main processes
            self.results_dir = results_dir
            self.monitor_dir = None
            self.history = defaultdict(list)
            self.best_metrics = {
                'best_val_loss': float('inf'),
                'best_epoch': 0
            }
            self.json_path = None
            self.csv_path = None
            self.report_path = None

    def update(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict] = None):
        """Update monitoring with new metrics (only on main process)"""
        is_main = not is_dist() or dist.get_rank() == 0
        if is_main:
            super().update(epoch, train_metrics, val_metrics)

    def save_history(self):
        """Save training history to files (only on main process)"""
        is_main = not is_dist() or dist.get_rank() == 0
        if is_main:
            super().save_history()

    def generate_report(self):
        """Generate text report of training progress (only on main process)"""
        is_main = not is_dist() or dist.get_rank() == 0
        if is_main:
            # Modify report to include distributed training info
            with open(self.report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("SSL PRETRAINING REPORT - DISTRIBUTED VERSION\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if is_dist():
                    f.write(f"World Size: {dist.get_world_size()} GPUs\n")
                f.write("=" * 80 + "\n\n")

                if not self.history['epoch']:
                    f.write("No training data available yet.\n")
                    return

                # Training summary
                f.write("TRAINING SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total epochs: {len(self.history['epoch'])}\n")
                f.write(f"Current epoch: {self.history['epoch'][-1]}\n")
                if is_dist():
                    f.write(f"Distributed training: Yes ({dist.get_world_size()} GPUs)\n")

                # Rest of the report generation...
                if 'val_total_loss' in self.history and self.history['val_total_loss']:
                    f.write(f"Best validation loss: {self.best_metrics['best_val_loss']:.4f} ")
                    f.write(f"(epoch {self.best_metrics['best_epoch']})\n")
                    val_losses = [v for v in self.history['val_total_loss'] if v is not None]
                    if val_losses:
                        f.write(f"Current validation loss: {val_losses[-1]:.4f}\n")

                # Latest metrics
                f.write("\nLATEST METRICS\n")
                f.write("-" * 40 + "\n")

                # Training metrics
                f.write("Training:\n")
                if 'train_total_loss' in self.history:
                    f.write(f"  Total loss: {self.history['train_total_loss'][-1]:.4f}\n")
                if 'train_inpainting_loss' in self.history:
                    f.write(f"  Inpainting loss: {self.history['train_inpainting_loss'][-1]:.4f}\n")
                if 'train_rotation_loss' in self.history:
                    f.write(f"  Rotation loss: {self.history['train_rotation_loss'][-1]:.4f}\n")
                if 'train_contrastive_loss' in self.history:
                    f.write(f"  Contrastive loss: {self.history['train_contrastive_loss'][-1]:.4f}\n")

                # Validation metrics
                if 'val_total_loss' in self.history and self.history['val_total_loss']:
                    val_losses = [v for v in self.history['val_total_loss'] if v is not None]
                    if val_losses:
                        f.write("\nValidation:\n")
                        f.write(f"  Total loss: {val_losses[-1]:.4f}\n")
                        if 'val_inpainting_loss' in self.history:
                            inp_losses = [v for v in self.history['val_inpainting_loss'] if v is not None]
                            if inp_losses:
                                f.write(f"  Inpainting loss: {inp_losses[-1]:.4f}\n")
                        if 'val_rotation_loss' in self.history:
                            rot_losses = [v for v in self.history['val_rotation_loss'] if v is not None]
                            if rot_losses:
                                f.write(f"  Rotation loss: {rot_losses[-1]:.4f}\n")
                        if 'val_contrastive_loss' in self.history:
                            con_losses = [v for v in self.history['val_contrastive_loss'] if v is not None]
                            if con_losses:
                                f.write(f"  Contrastive loss: {con_losses[-1]:.4f}\n")

                # Training dynamics
                f.write("\nTRAINING DYNAMICS\n")
                f.write("-" * 40 + "\n")

                if len(self.history['epoch']) > 1:
                    # Calculate trends
                    train_losses = np.array(self.history['train_total_loss'])
                    if len(train_losses) >= 10:
                        recent_trend = np.polyfit(range(10), train_losses[-10:], 1)[0]
                        f.write(f"Recent training loss trend: {'↓' if recent_trend < 0 else '↑'} ")
                        f.write(f"({recent_trend:.6f} per epoch)\n")

                    # Learning rate
                    if 'train_lr' in self.history:
                        f.write(f"Current learning rate: {self.history['train_lr'][-1]:.6f}\n")

                    # Gradient norm
                    if 'train_grad_norm' in self.history:
                        grad_norms = self.history['train_grad_norm']
                        f.write(f"Average gradient norm: {np.mean(grad_norms[-10:]):.4f}\n")
                        f.write(f"Max gradient norm: {np.max(grad_norms):.4f}\n")

            print(f"📊 Report saved to: {self.report_path}")

    def plot_metrics(self):
        """Generate plots for training metrics (only on main process)"""
        is_main = not is_dist() or dist.get_rank() == 0
        if is_main:
            super().plot_metrics()