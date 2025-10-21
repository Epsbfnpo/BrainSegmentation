"""
Dice Score Monitoring Module - Dual-Branch Cross-Domain Graph Alignment Version
Tracks dice score evolution with distributed training support and dual-branch metrics
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
import torch.distributed as dist


class DiceMonitor:
    """Monitor dice score evolution and detect anomalies - Dual-Branch Version"""

    def __init__(self, results_dir: str, drop_threshold: float = 0.3, window_size: int = 5,
                 graph_prior_enabled: bool = False, cross_domain_enabled: bool = False,
                 dual_branch_enabled: bool = False):
        """Initialize DiceMonitor with dual-branch tracking"""
        self.results_dir = results_dir
        self.drop_threshold = drop_threshold
        self.window_size = window_size
        self.graph_prior_enabled = graph_prior_enabled
        self.cross_domain_enabled = cross_domain_enabled
        self.dual_branch_enabled = dual_branch_enabled

        # Only create directories on main process
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if is_main:
            self.monitor_dir = os.path.join(results_dir, 'dice_monitoring')
            os.makedirs(self.monitor_dir, exist_ok=True)
            if graph_prior_enabled:
                self.graph_dir = os.path.join(results_dir, 'graph_analysis')
                os.makedirs(self.graph_dir, exist_ok=True)
            if dual_branch_enabled:
                self.dual_branch_dir = os.path.join(results_dir, 'dual_branch_analysis')
                os.makedirs(self.dual_branch_dir, exist_ok=True)
        else:
            self.monitor_dir = None
            self.graph_dir = None
            self.dual_branch_dir = None

        # Initialize tracking
        self.dice_history = {
            'epochs': [],
            'dice_scores': [],
            'seg_losses': [],
            'domain_losses': [],
            'domain_accs': [],
            'learning_rates': [],
            'alphas': [],
            'timestamps': [],
            'per_class_dice': [],
            'drops_detected': [],
            # Graph-related metrics (legacy)
            'graph_losses': [],
            'graph_spec_losses': [],
            'graph_edge_losses': [],
            'graph_sym_losses': [],
            'graph_struct_losses': [],
            # Cross-domain graph metrics
            'graph_spec_src_losses': [],
            'graph_edge_src_losses': [],
            'graph_spec_tgt_losses': [],
            'graph_edge_tgt_losses': [],
            # NEW: Dynamic spectral metrics
            'dyn_spec_losses': [],
            'dyn_lambda_values': [],
            'dyn_conflict_counts': [],
            'qap_mismatch_scores': [],
            # Validation metrics
            'structural_violations': [],
            'symmetry_scores': [],
            'adjacency_errors': [],
            'adjacency_errors_src': [],
            # NEW: Small structure tracking
            'bottom_30_dice': [],
            'bottom_10_dice': [],
            # NEW: Exponential moving averages for conflict detection
            'forbidden_ema': [],
            'required_ema': [],
            'conflict_signals': []
        }

        # File paths (only for main process)
        if is_main:
            self.report_path = os.path.join(self.monitor_dir, 'dice_report.txt')
            self.json_path = os.path.join(self.monitor_dir, 'dice_history.json')
            self.csv_path = os.path.join(self.monitor_dir, 'dice_history.csv')
            if graph_prior_enabled:
                self.graph_report_path = os.path.join(self.graph_dir, 'graph_analysis_report.txt')
            if cross_domain_enabled:
                self.crossdomain_report_path = os.path.join(self.graph_dir, 'crossdomain_alignment_report.txt')
            if dual_branch_enabled:
                self.dual_branch_report_path = os.path.join(self.dual_branch_dir, 'dual_branch_report.txt')
        else:
            self.report_path = None
            self.json_path = None
            self.csv_path = None
            self.graph_report_path = None
            self.crossdomain_report_path = None
            self.dual_branch_report_path = None

        # EMA parameters
        self.ema_alpha = 0.3  # Smoothing factor for exponential moving average
        self.forbidden_ema_value = 0.0
        self.required_ema_value = 0.0

    def add_dice_score(self, epoch: int, dice_score: float,
                       train_metrics: Dict, val_metrics: Dict) -> None:
        """Add new dice score and related metrics including dual-branch metrics"""
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if not is_main:
            return

        self.dice_history['epochs'].append(epoch)
        self.dice_history['dice_scores'].append(dice_score)
        self.dice_history['seg_losses'].append(train_metrics.get('seg_loss', 0))
        self.dice_history['domain_losses'].append(train_metrics.get('domain_loss', 0))
        self.dice_history['domain_accs'].append(train_metrics.get('domain_acc', 0))
        self.dice_history['learning_rates'].append(train_metrics.get('lr', 0))
        self.dice_history['alphas'].append(train_metrics.get('alpha', 0))
        self.dice_history['timestamps'].append(datetime.now().isoformat())
        self.dice_history['per_class_dice'].append(val_metrics.get('dice_per_class', []))

        # Add legacy graph metrics (for backward compatibility)
        if self.graph_prior_enabled:
            graph_loss_val = train_metrics.get('graph_loss', train_metrics.get('graph_total', 0))
            self.dice_history['graph_losses'].append(graph_loss_val)
            self.dice_history['graph_spec_losses'].append(train_metrics.get('graph_spec', train_metrics.get('graph_spec_src', 0)))
            self.dice_history['graph_edge_losses'].append(train_metrics.get('graph_edge', train_metrics.get('graph_edge_src', 0)))
            self.dice_history['graph_sym_losses'].append(train_metrics.get('graph_sym', 0))
            self.dice_history['graph_struct_losses'].append(train_metrics.get('graph_struct', 0))

            # Cross-domain metrics
            if self.cross_domain_enabled:
                self.dice_history['graph_spec_src_losses'].append(train_metrics.get('graph_spec_src', 0))
                self.dice_history['graph_edge_src_losses'].append(train_metrics.get('graph_edge_src', 0))
                self.dice_history['graph_spec_tgt_losses'].append(train_metrics.get('graph_spec_tgt', 0))
                self.dice_history['graph_edge_tgt_losses'].append(train_metrics.get('graph_edge_tgt', 0))

            # Compute structural violations from validation metrics
            violations = self._compute_structural_violations(val_metrics)
            self.dice_history['structural_violations'].append(violations)

            # Update EMA for conflict detection
            if violations:
                forbidden_current = violations.get('forbidden_present', 0)
                required_current = violations.get('required_missing', 0)

                self.forbidden_ema_value = self.ema_alpha * forbidden_current + (1 - self.ema_alpha) * self.forbidden_ema_value
                self.required_ema_value = self.ema_alpha * required_current + (1 - self.ema_alpha) * self.required_ema_value

                self.dice_history['forbidden_ema'].append(self.forbidden_ema_value)
                self.dice_history['required_ema'].append(self.required_ema_value)

                # Detect conflicts (EMA rising)
                conflict_signal = 0
                if len(self.dice_history['forbidden_ema']) >= 3:
                    recent_ema = self.dice_history['forbidden_ema'][-3:]
                    if recent_ema[-1] > recent_ema[0] * 1.1:  # 10% increase
                        conflict_signal = 1
                self.dice_history['conflict_signals'].append(conflict_signal)

            # Compute symmetry scores
            symmetry = self._compute_symmetry_scores(val_metrics)
            self.dice_history['symmetry_scores'].append(symmetry)

            # Compute adjacency errors
            adj_errors = self._compute_adjacency_errors(val_metrics)
            self.dice_history['adjacency_errors'].append(adj_errors)

            # Store source domain alignment errors if available
            if 'adjacency_errors_src' in val_metrics and val_metrics['adjacency_errors_src']:
                self.dice_history['adjacency_errors_src'].append(val_metrics['adjacency_errors_src'])

        # Dual-branch specific metrics
        if self.dual_branch_enabled:
            self.dice_history['dyn_spec_losses'].append(train_metrics.get('dyn_spec', 0))
            self.dice_history['dyn_lambda_values'].append(train_metrics.get('dyn_lambda', 0))
            self.dice_history['dyn_conflict_counts'].append(train_metrics.get('dyn_conflicts', 0))

            # QAP mismatch score (could be computed in trainer)
            qap_score = train_metrics.get('qap_mismatch', 0)
            self.dice_history['qap_mismatch_scores'].append(qap_score)

        # Track small structure performance
        if val_metrics.get('dice_per_class'):
            dice_per_class = np.array(val_metrics['dice_per_class'])
            sorted_dice = np.sort(dice_per_class)

            # Bottom 30% (approximately 26 classes for 87 total)
            bottom_30_count = int(len(dice_per_class) * 0.3)
            bottom_30_avg = np.mean(sorted_dice[:bottom_30_count]) if bottom_30_count > 0 else 0
            self.dice_history['bottom_30_dice'].append(bottom_30_avg)

            # Bottom 10 classes
            bottom_10_avg = np.mean(sorted_dice[:10]) if len(sorted_dice) >= 10 else np.mean(sorted_dice)
            self.dice_history['bottom_10_dice'].append(bottom_10_avg)

        # Save to JSON after each update
        self.save_history()

        # Check for dice drop
        if self.check_dice_drop():
            self.dice_history['drops_detected'].append({
                'epoch': epoch,
                'dice_score': dice_score,
                'timestamp': datetime.now().isoformat()
            })

    def _compute_structural_violations(self, val_metrics: Dict) -> Dict:
        """Compute structural violation metrics from validation results"""
        violations = {
            'required_missing': 0,
            'forbidden_present': 0,
            'containment_violated': 0,
            'exclusivity_violated': 0
        }

        if 'structural_violations' in val_metrics:
            violations.update(val_metrics['structural_violations'])

        return violations

    def _compute_symmetry_scores(self, val_metrics: Dict) -> Dict:
        """Compute symmetry scores for left-right paired structures"""
        symmetry = {
            'mean_symmetry': 0.0,
            'min_symmetry': 1.0,
            'max_asymmetry': 0.0
        }

        if 'symmetry_scores' in val_metrics:
            scores = val_metrics['symmetry_scores']
            if scores:
                symmetry['mean_symmetry'] = np.mean(scores)
                symmetry['min_symmetry'] = np.min(scores)
                symmetry['max_asymmetry'] = 1.0 - np.min(scores)

        return symmetry

    def _compute_adjacency_errors(self, val_metrics: Dict) -> Dict:
        """Compute adjacency prediction errors"""
        adj_errors = {
            'mean_adj_error': 0.0,
            'max_adj_error': 0.0,
            'spectral_distance': 0.0
        }

        if 'adjacency_errors' in val_metrics:
            adj_errors.update(val_metrics['adjacency_errors'])

        return adj_errors

    def check_dice_drop(self) -> bool:
        """Check if there's a significant dice score drop"""
        if len(self.dice_history['dice_scores']) < self.window_size + 1:
            return False

        dice_scores = np.array(self.dice_history['dice_scores'])

        # Calculate moving average
        if len(dice_scores) >= self.window_size:
            recent_avg = np.mean(dice_scores[-self.window_size:])
            prev_avg = np.mean(dice_scores[-(self.window_size + 1):-1])

            # Check for significant drop
            if prev_avg > 0:
                drop_ratio = (prev_avg - recent_avg) / prev_avg
                if drop_ratio > self.drop_threshold:
                    return True

        # Also check for absolute drop
        if len(dice_scores) >= 2:
            current = dice_scores[-1]
            previous = dice_scores[-2]
            if previous > 0 and (previous - current) / previous > self.drop_threshold:
                return True

        return False

    def save_history(self) -> None:
        """Save history to JSON and CSV files"""
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if not is_main:
            return

        # Save to JSON
        with open(self.json_path, 'w') as f:
            json.dump(self.dice_history, f, indent=2, default=str)

        # Save to CSV
        if self.dice_history['epochs']:
            df_data = {
                'epoch': self.dice_history['epochs'],
                'dice_score': self.dice_history['dice_scores'],
                'seg_loss': self.dice_history['seg_losses'],
                'domain_loss': self.dice_history['domain_losses'],
                'domain_acc': self.dice_history['domain_accs'],
                'learning_rate': self.dice_history['learning_rates'],
                'alpha': self.dice_history['alphas'],
                'timestamp': self.dice_history['timestamps']
            }

            # Add graph metrics if available
            if self.graph_prior_enabled and self.dice_history['graph_losses']:
                df_data.update({
                    'graph_loss': self.dice_history['graph_losses'],
                    'graph_spec': self.dice_history['graph_spec_losses'],
                    'graph_edge': self.dice_history['graph_edge_losses'],
                    'graph_sym': self.dice_history['graph_sym_losses'],
                    'graph_struct': self.dice_history['graph_struct_losses']
                })

                # Add cross-domain metrics if available
                if self.cross_domain_enabled and self.dice_history['graph_spec_src_losses']:
                    df_data.update({
                        'graph_spec_src': self.dice_history['graph_spec_src_losses'],
                        'graph_edge_src': self.dice_history['graph_edge_src_losses'],
                        'graph_spec_tgt': self.dice_history['graph_spec_tgt_losses'],
                        'graph_edge_tgt': self.dice_history['graph_edge_tgt_losses']
                    })

                # Add dual-branch metrics if available
                if self.dual_branch_enabled and self.dice_history['dyn_spec_losses']:
                    df_data.update({
                        'dyn_spec': self.dice_history['dyn_spec_losses'],
                        'dyn_lambda': self.dice_history['dyn_lambda_values'],
                        'dyn_conflicts': self.dice_history['dyn_conflict_counts']
                    })

            df = pd.DataFrame(df_data)
            df.to_csv(self.csv_path, index=False)

    def generate_report(self) -> None:
        """Generate detailed dice monitoring report with dual-branch metrics"""
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if not is_main or not self.dice_history['epochs']:
            return

        with open(self.report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DICE SCORE MONITORING REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Add distributed training info
            if dist.is_initialized():
                f.write(f"World Size: {dist.get_world_size()} GPUs\n")

            if self.graph_prior_enabled:
                f.write("Graph Prior Regularization: ENABLED\n")
                if self.cross_domain_enabled:
                    f.write("Cross-Domain Alignment: ENABLED\n")
                if self.dual_branch_enabled:
                    f.write("Dual-Branch Architecture: ENABLED\n")

            f.write("=" * 80 + "\n\n")

            # Summary statistics
            dice_scores = np.array(self.dice_history['dice_scores'])
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total epochs monitored: {len(dice_scores)}\n")
            f.write(f"Best dice score: {np.max(dice_scores):.4f} (epoch {self.dice_history['epochs'][np.argmax(dice_scores)]})\n")
            f.write(f"Current dice score: {dice_scores[-1]:.4f}\n")
            f.write(f"Average dice score: {np.mean(dice_scores):.4f}\n")
            f.write(f"Std dev: {np.std(dice_scores):.4f}\n")
            f.write(f"Number of drops detected: {len(self.dice_history['drops_detected'])}\n\n")

            # Dice evolution
            f.write("DICE SCORE EVOLUTION\n")
            f.write("-" * 40 + "\n")

            if self.dual_branch_enabled and self.dice_history['dyn_lambda_values']:
                f.write("Epoch | Dice Score | Seg Loss | Graph(P) | Graph(D) | λ_dyn | LR\n")
                f.write("-" * 75 + "\n")

                for i in range(len(self.dice_history['epochs'])):
                    epoch = self.dice_history['epochs'][i]
                    dice = self.dice_history['dice_scores'][i]
                    seg_loss = self.dice_history['seg_losses'][i]
                    graph_prior = self.dice_history['graph_losses'][i] if i < len(self.dice_history['graph_losses']) else 0
                    graph_dyn = self.dice_history['dyn_spec_losses'][i] if i < len(self.dice_history['dyn_spec_losses']) else 0
                    lambda_dyn = self.dice_history['dyn_lambda_values'][i] if i < len(self.dice_history['dyn_lambda_values']) else 0
                    lr = self.dice_history['learning_rates'][i]
                    drop_marker = " *DROP*" if any(d['epoch'] == epoch for d in self.dice_history['drops_detected']) else ""

                    f.write(f"{epoch:5d} | {dice:10.4f} | {seg_loss:8.4f} | {graph_prior:8.4f} | {graph_dyn:8.4f} | {lambda_dyn:5.3f} | {lr:.2e}{drop_marker}\n")

            elif self.cross_domain_enabled and self.dice_history['graph_spec_src_losses']:
                f.write("Epoch | Dice Score | Seg Loss | Graph(S) | Graph(T) | LR\n")
                f.write("-" * 65 + "\n")

                for i in range(len(self.dice_history['epochs'])):
                    epoch = self.dice_history['epochs'][i]
                    dice = self.dice_history['dice_scores'][i]
                    seg_loss = self.dice_history['seg_losses'][i]
                    graph_src = (self.dice_history['graph_spec_src_losses'][i] +
                                self.dice_history['graph_edge_src_losses'][i]) if i < len(self.dice_history['graph_spec_src_losses']) else 0
                    graph_tgt = (self.dice_history['graph_spec_tgt_losses'][i] +
                                self.dice_history['graph_edge_tgt_losses'][i]) if i < len(self.dice_history['graph_spec_tgt_losses']) else 0
                    lr = self.dice_history['learning_rates'][i]
                    drop_marker = " *DROP*" if any(d['epoch'] == epoch for d in self.dice_history['drops_detected']) else ""

                    f.write(f"{epoch:5d} | {dice:10.4f} | {seg_loss:8.4f} | {graph_src:8.4f} | {graph_tgt:8.4f} | {lr:.2e}{drop_marker}\n")

            elif self.graph_prior_enabled and self.dice_history['graph_losses']:
                f.write("Epoch | Dice Score | Seg Loss | Graph Loss | LR\n")
                f.write("-" * 60 + "\n")

                for i in range(len(self.dice_history['epochs'])):
                    epoch = self.dice_history['epochs'][i]
                    dice = self.dice_history['dice_scores'][i]
                    seg_loss = self.dice_history['seg_losses'][i]
                    graph_loss = self.dice_history['graph_losses'][i] if i < len(self.dice_history['graph_losses']) else 0
                    lr = self.dice_history['learning_rates'][i]
                    drop_marker = " *DROP*" if any(d['epoch'] == epoch for d in self.dice_history['drops_detected']) else ""

                    f.write(f"{epoch:5d} | {dice:10.4f} | {seg_loss:8.4f} | {graph_loss:10.4f} | {lr:.2e}{drop_marker}\n")

            # Dual-branch analysis
            if self.dual_branch_enabled and self.dice_history['dyn_spec_losses']:
                f.write("\n\nDUAL-BRANCH ANALYSIS\n")
                f.write("-" * 40 + "\n")

                dyn_spec = np.array(self.dice_history['dyn_spec_losses'])
                dyn_lambda = np.array(self.dice_history['dyn_lambda_values'])
                dyn_conflicts = np.array(self.dice_history['dyn_conflict_counts'])

                f.write("PRIOR BRANCH (Stable Structural Anchoring):\n")
                if self.dice_history['graph_losses']:
                    graph_losses = np.array(self.dice_history['graph_losses'])
                    f.write(f"  Average loss: {np.mean(graph_losses):.4f} (std: {np.std(graph_losses):.4f})\n")
                    f.write(f"  Recent trend (last 10): {np.mean(graph_losses[-10:]):.4f}\n")

                f.write("\nDYNAMIC BRANCH (Flexible Spectral Alignment):\n")
                active_idx = np.where(dyn_lambda > 0)[0]
                if len(active_idx) > 0:
                    f.write(f"  Activation epoch: {self.dice_history['epochs'][active_idx[0]]}\n")
                    f.write(f"  Current λ_dyn: {dyn_lambda[-1]:.4f}\n")
                    f.write(f"  Average loss (active epochs): {np.mean(dyn_spec[active_idx]):.4f}\n")
                    f.write(f"  Total conflict suppressions: {np.sum(dyn_conflicts)}\n")
                else:
                    f.write("  Not yet active\n")

            # Cross-domain alignment analysis
            if self.cross_domain_enabled and self.dice_history['graph_spec_src_losses']:
                f.write("\n\nCROSS-DOMAIN ALIGNMENT ANALYSIS\n")
                f.write("-" * 40 + "\n")

                spec_src = np.array(self.dice_history['graph_spec_src_losses'])
                edge_src = np.array(self.dice_history['graph_edge_src_losses'])
                spec_tgt = np.array(self.dice_history['graph_spec_tgt_losses'])
                edge_tgt = np.array(self.dice_history['graph_edge_tgt_losses'])

                f.write("SOURCE Domain Alignment (Primary):\n")
                f.write(f"  Spectral loss: {np.mean(spec_src):.4f} (std: {np.std(spec_src):.4f})\n")
                f.write(f"  Edge loss: {np.mean(edge_src):.4f} (std: {np.std(edge_src):.4f})\n")
                f.write(f"  Combined: {np.mean(spec_src + edge_src):.4f}\n")

                f.write("\nTARGET Domain Regularization (Secondary):\n")
                f.write(f"  Spectral loss: {np.mean(spec_tgt):.4f} (std: {np.std(spec_tgt):.4f})\n")
                f.write(f"  Edge loss: {np.mean(edge_tgt):.4f} (std: {np.std(edge_tgt):.4f})\n")
                f.write(f"  Combined: {np.mean(spec_tgt + edge_tgt):.4f}\n")

                # Alignment progress
                f.write("\nAlignment Progress:\n")
                if len(spec_src) >= 10:
                    early = spec_src[:10].mean() + edge_src[:10].mean()
                    recent = spec_src[-10:].mean() + edge_src[-10:].mean()
                    improvement = (early - recent) / early * 100 if early > 0 else 0
                    f.write(f"  Early epochs (1-10): {early:.4f}\n")
                    f.write(f"  Recent epochs (last 10): {recent:.4f}\n")
                    f.write(f"  Improvement: {improvement:.1f}%\n")

            # Graph prior analysis
            if self.graph_prior_enabled and self.dice_history['graph_losses']:
                f.write("\n\nGRAPH PRIOR ANALYSIS\n")
                f.write("-" * 40 + "\n")

                graph_losses = np.array(self.dice_history['graph_losses'])
                spec_losses = np.array(self.dice_history['graph_spec_losses'])
                edge_losses = np.array(self.dice_history['graph_edge_losses'])
                sym_losses = np.array(self.dice_history['graph_sym_losses'])

                f.write(f"Average graph loss: {np.mean(graph_losses):.4f}\n")
                f.write(f"  - Spectral: {np.mean(spec_losses):.4f}\n")
                f.write(f"  - Edge: {np.mean(edge_losses):.4f}\n")
                f.write(f"  - Symmetry: {np.mean(sym_losses):.4f}\n")

            # Validation graph metrics
            if self.dice_history['adjacency_errors_src']:
                f.write("\n\nVALIDATION GRAPH METRICS (Cross-Domain)\n")
                f.write("-" * 40 + "\n")

                latest_src_errors = self.dice_history['adjacency_errors_src'][-1]
                f.write("Source Domain Alignment:\n")
                f.write(f"  Mean adjacency error: {latest_src_errors.get('mean_abs_error_src', 0):.4f}\n")
                f.write(f"  Spectral distance: {latest_src_errors.get('spectral_distance_src', 0):.4f}\n")

            if self.dice_history['structural_violations']:
                latest_violations = self.dice_history['structural_violations'][-1]
                f.write("\nStructural Violations:\n")
                for key, value in latest_violations.items():
                    f.write(f"  {key}: {value}\n")

            if self.dice_history['symmetry_scores']:
                latest_symmetry = self.dice_history['symmetry_scores'][-1]
                f.write("\nSymmetry Scores:\n")
                f.write(f"  Mean symmetry: {latest_symmetry.get('mean_symmetry', 0):.3f}\n")
                f.write(f"  Min symmetry: {latest_symmetry.get('min_symmetry', 0):.3f}\n")
                f.write(f"  Max asymmetry: {latest_symmetry.get('max_asymmetry', 0):.3f}\n")

            # Small structure analysis
            if self.dice_history['bottom_30_dice']:
                f.write("\n\nSMALL STRUCTURE PERFORMANCE\n")
                f.write("-" * 40 + "\n")

                bottom_30 = np.array(self.dice_history['bottom_30_dice'])
                bottom_10 = np.array(self.dice_history['bottom_10_dice'])

                f.write("Bottom 30% classes:\n")
                f.write(f"  Current dice: {bottom_30[-1]:.4f}\n")
                f.write(f"  Best dice: {np.max(bottom_30):.4f}\n")
                f.write(f"  Average: {np.mean(bottom_30):.4f}\n")

                f.write("\nBottom 10 classes:\n")
                f.write(f"  Current dice: {bottom_10[-1]:.4f}\n")
                f.write(f"  Best dice: {np.max(bottom_10):.4f}\n")
                f.write(f"  Average: {np.mean(bottom_10):.4f}\n")

            # Drop analysis
            if self.dice_history['drops_detected']:
                f.write("\n\nDICE SCORE DROPS DETECTED\n")
                f.write("-" * 40 + "\n")
                for drop in self.dice_history['drops_detected']:
                    f.write(f"Epoch {drop['epoch']}: Dice = {drop['dice_score']:.4f} "
                            f"(Time: {drop['timestamp']})\n")

            # Correlation analysis
            f.write("\n\nCORRELATION ANALYSIS\n")
            f.write("-" * 40 + "\n")

            if len(dice_scores) > 5:
                seg_losses = np.array(self.dice_history['seg_losses'])
                dice_seg_corr = np.corrcoef(dice_scores, seg_losses)[0, 1]
                f.write(f"Dice vs Seg Loss correlation: {dice_seg_corr:.3f}\n")

                if self.dual_branch_enabled and len(self.dice_history['dyn_spec_losses']) > 5:
                    dyn_losses = np.array(self.dice_history['dyn_spec_losses'])
                    if len(dyn_losses) == len(dice_scores):
                        dice_dyn_corr = np.corrcoef(dice_scores, dyn_losses)[0, 1]
                        f.write(f"Dice vs Dynamic Spectral correlation: {dice_dyn_corr:.3f}\n")

                if self.cross_domain_enabled and len(self.dice_history['graph_spec_src_losses']) > 5:
                    src_losses = np.array(self.dice_history['graph_spec_src_losses']) + np.array(self.dice_history['graph_edge_src_losses'])
                    dice_src_corr = np.corrcoef(dice_scores[:len(src_losses)], src_losses)[0, 1]
                    f.write(f"Dice vs Source Alignment correlation: {dice_src_corr:.3f}\n")
                elif self.graph_prior_enabled and len(self.dice_history['graph_losses']) > 5:
                    graph_losses = np.array(self.dice_history['graph_losses'])
                    dice_graph_corr = np.corrcoef(dice_scores, graph_losses)[0, 1]
                    f.write(f"Dice vs Graph Loss correlation: {dice_graph_corr:.3f}\n")

            # Recent trend
            f.write("\n\nRECENT TREND\n")
            f.write("-" * 40 + "\n")

            if len(dice_scores) >= 10:
                recent_dice = dice_scores[-10:]
                trend = np.polyfit(range(10), recent_dice, 1)[0]
                f.write(f"Last 10 epochs trend: {'↑' if trend > 0 else '↓'} ({trend:.5f} per epoch)\n")
                f.write(f"Recent average: {np.mean(recent_dice):.4f}\n")

            # Per-class analysis
            if self.dice_history['per_class_dice'] and any(self.dice_history['per_class_dice']):
                f.write("\n\nPER-CLASS DICE ANALYSIS (Latest)\n")
                f.write("-" * 40 + "\n")

                latest_per_class = self.dice_history['per_class_dice'][-1]
                if latest_per_class:
                    sorted_indices = np.argsort(latest_per_class)

                    f.write("Worst 10 classes:\n")
                    for idx in sorted_indices[:10]:
                        f.write(f"  Class {idx}: {latest_per_class[idx]:.4f}\n")

                    f.write("\nBest 5 classes:\n")
                    for idx in sorted_indices[-5:]:
                        f.write(f"  Class {idx}: {latest_per_class[idx]:.4f}\n")

        print(f"📄 Dice monitoring report saved to: {self.report_path}")

    def generate_cross_domain_report(self) -> None:
        """Generate specialized cross-domain alignment report"""
        if not self.cross_domain_enabled or not self.crossdomain_report_path:
            return

        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if not is_main or not self.dice_history['epochs']:
            return

        with open(self.crossdomain_report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CROSS-DOMAIN GRAPH ALIGNMENT REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            if not self.dice_history['graph_spec_src_losses']:
                f.write("No cross-domain alignment data available.\n")
                return

            epochs = np.array(self.dice_history['epochs'])
            dice_scores = np.array(self.dice_history['dice_scores'])
            spec_src = np.array(self.dice_history['graph_spec_src_losses'])
            edge_src = np.array(self.dice_history['graph_edge_src_losses'])
            spec_tgt = np.array(self.dice_history['graph_spec_tgt_losses'])
            edge_tgt = np.array(self.dice_history['graph_edge_tgt_losses'])

            f.write("ALIGNMENT OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total epochs with alignment: {len(spec_src)}\n")
            f.write(f"Latest dice score: {dice_scores[-1]:.4f}\n")
            f.write(f"Best dice score: {np.max(dice_scores):.4f}\n\n")

            # Detailed source alignment
            f.write("SOURCE DOMAIN ALIGNMENT (dHCP)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Spectral Alignment:\n")
            f.write(f"  Initial: {spec_src[0]:.4f}\n")
            f.write(f"  Current: {spec_src[-1]:.4f}\n")
            f.write(f"  Best: {np.min(spec_src):.4f}\n")
            f.write(f"  Mean: {np.mean(spec_src):.4f}\n")

            f.write(f"\nEdge Consistency:\n")
            f.write(f"  Initial: {edge_src[0]:.4f}\n")
            f.write(f"  Current: {edge_src[-1]:.4f}\n")
            f.write(f"  Best: {np.min(edge_src):.4f}\n")
            f.write(f"  Mean: {np.mean(edge_src):.4f}\n")

            # Target regularization
            f.write("\nTARGET DOMAIN REGULARIZATION (PPREMO/PREBO)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Spectral Alignment:\n")
            f.write(f"  Initial: {spec_tgt[0]:.4f}\n")
            f.write(f"  Current: {spec_tgt[-1]:.4f}\n")
            f.write(f"  Best: {np.min(spec_tgt):.4f}\n")
            f.write(f"  Mean: {np.mean(spec_tgt):.4f}\n")

            f.write(f"\nEdge Consistency:\n")
            f.write(f"  Initial: {edge_tgt[0]:.4f}\n")
            f.write(f"  Current: {edge_tgt[-1]:.4f}\n")
            f.write(f"  Best: {np.min(edge_tgt):.4f}\n")
            f.write(f"  Mean: {np.mean(edge_tgt):.4f}\n")

            # Convergence analysis
            f.write("\nCONVERGENCE ANALYSIS\n")
            f.write("-" * 40 + "\n")

            window = min(20, len(spec_src) // 4)
            if len(spec_src) > window * 2:
                early_src = (spec_src[:window] + edge_src[:window]).mean()
                late_src = (spec_src[-window:] + edge_src[-window:]).mean()
                src_improvement = (early_src - late_src) / early_src * 100

                early_tgt = (spec_tgt[:window] + edge_tgt[:window]).mean()
                late_tgt = (spec_tgt[-window:] + edge_tgt[-window:]).mean()
                tgt_improvement = (early_tgt - late_tgt) / early_tgt * 100

                f.write(f"Source alignment improvement: {src_improvement:.1f}%\n")
                f.write(f"Target alignment improvement: {tgt_improvement:.1f}%\n")

                # Check if source is dominating
                src_ratio = late_src / (late_tgt + 1e-8)
                f.write(f"Current source/target ratio: {src_ratio:.2f}\n")
                if src_ratio > 5:
                    f.write("  ⚠️ Source alignment may be dominating\n")
                elif src_ratio < 0.2:
                    f.write("  ⚠️ Target regularization may be too strong\n")
                else:
                    f.write("  ✓ Balanced alignment\n")

            # Validation metrics correlation
            if self.dice_history['adjacency_errors_src']:
                f.write("\nVALIDATION METRICS\n")
                f.write("-" * 40 + "\n")

                src_val_errors = []
                for err_dict in self.dice_history['adjacency_errors_src']:
                    if 'spectral_distance_src' in err_dict:
                        src_val_errors.append(err_dict['spectral_distance_src'])

                if src_val_errors:
                    f.write(f"Source spectral distance (validation):\n")
                    f.write(f"  Initial: {src_val_errors[0]:.4f}\n")
                    f.write(f"  Current: {src_val_errors[-1]:.4f}\n")
                    f.write(f"  Best: {np.min(src_val_errors):.4f}\n")

            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")

            if len(spec_src) > 10:
                recent_src = spec_src[-5:].mean() + edge_src[-5:].mean()
                recent_tgt = spec_tgt[-5:].mean() + edge_tgt[-5:].mean()

                if recent_src > 0.5:
                    f.write("• Source alignment loss is high - consider:\n")
                    f.write("  - Increasing warmup epochs\n")
                    f.write("  - Reducing lambda_spec_src and lambda_edge_src\n")
                    f.write("  - Checking if source/target domains are too different\n")

                if recent_tgt < 0.01:
                    f.write("• Target regularization is very weak - consider:\n")
                    f.write("  - Slightly increasing lambda_spec_tgt and lambda_edge_tgt\n")
                    f.write("  - This may help preserve target-specific structures\n")

                # Check dice correlation
                if len(dice_scores) == len(spec_src):
                    src_total = spec_src + edge_src
                    corr = np.corrcoef(dice_scores, src_total)[0, 1]
                    if corr > 0.7:
                        f.write(f"• High positive correlation ({corr:.2f}) between dice and source loss\n")
                        f.write("  - Source alignment may be hindering segmentation\n")
                        f.write("  - Consider reducing source alignment weights\n")

        print(f"📊 Cross-domain report saved to: {self.crossdomain_report_path}")

    def generate_dual_branch_report(self) -> None:
        """Generate specialized dual-branch analysis report"""
        if not self.dual_branch_enabled or not self.dual_branch_report_path:
            return

        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if not is_main or not self.dice_history['epochs']:
            return

        with open(self.dual_branch_report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DUAL-BRANCH GRAPH ALIGNMENT ANALYSIS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            epochs = np.array(self.dice_history['epochs'])
            dice_scores = np.array(self.dice_history['dice_scores'])

            f.write("OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total epochs analyzed: {len(epochs)}\n")
            f.write(f"Best dice score: {np.max(dice_scores):.4f} (epoch {epochs[np.argmax(dice_scores)]})\n")
            f.write(f"Current dice score: {dice_scores[-1]:.4f}\n\n")

            # Prior branch analysis
            if self.dice_history['graph_losses']:
                graph_losses = np.array(self.dice_history['graph_losses'])
                f.write("PRIOR BRANCH (Structural Anchoring)\n")
                f.write("-" * 40 + "\n")
                f.write(f"Average graph loss: {np.mean(graph_losses):.4f}\n")
                f.write(f"Loss trend (last 10 epochs): {np.mean(graph_losses[-10:]):.4f}\n")

                if self.dice_history['graph_spec_src_losses']:
                    spec_src = np.array(self.dice_history['graph_spec_src_losses'])
                    edge_src = np.array(self.dice_history['graph_edge_src_losses'])
                    f.write(f"\nSource alignment:\n")
                    f.write(f"  Spectral: {np.mean(spec_src[-10:]):.4f} (std: {np.std(spec_src[-10:]):.4f})\n")
                    f.write(f"  Edge: {np.mean(edge_src[-10:]):.4f} (std: {np.std(edge_src[-10:]):.4f})\n")

            # Dynamic branch analysis
            if self.dice_history['dyn_spec_losses']:
                dyn_spec = np.array(self.dice_history['dyn_spec_losses'])
                dyn_lambda = np.array(self.dice_history['dyn_lambda_values'])
                dyn_conflicts = np.array(self.dice_history['dyn_conflict_counts'])

                f.write("\nDYNAMIC BRANCH (Flexible Spectral)\n")
                f.write("-" * 40 + "\n")

                # Find when dynamic branch became active
                active_idx = np.where(dyn_lambda > 0)[0]
                if len(active_idx) > 0:
                    start_epoch = epochs[active_idx[0]]
                    f.write(f"Activation epoch: {start_epoch}\n")
                    f.write(f"Current λ_dyn: {dyn_lambda[-1]:.4f}\n")
                    f.write(f"Average loss (last 10): {np.mean(dyn_spec[-10:]):.4f}\n")
                    f.write(f"Total conflict suppressions: {np.sum(dyn_conflicts)}\n")
                else:
                    f.write("Dynamic branch not yet active\n")

            # Branch interaction analysis
            f.write("\nBRANCH INTERACTION\n")
            f.write("-" * 40 + "\n")

            if self.dice_history['conflict_signals']:
                conflicts = np.array(self.dice_history['conflict_signals'])
                conflict_epochs = epochs[np.where(conflicts > 0)[0]]
                f.write(f"Conflict events: {len(conflict_epochs)}\n")
                if len(conflict_epochs) > 0:
                    f.write(f"  Epochs with conflicts: {conflict_epochs.tolist()}\n")

            if self.dice_history['forbidden_ema']:
                forbidden_ema = np.array(self.dice_history['forbidden_ema'])
                f.write(f"\nForbidden edges EMA:\n")
                f.write(f"  Current: {forbidden_ema[-1]:.2f}\n")
                f.write(f"  Trend: {'↑' if forbidden_ema[-1] > forbidden_ema[-10] else '↓'}\n")

            # Small structure focus
            if self.dice_history['bottom_30_dice']:
                bottom_30 = np.array(self.dice_history['bottom_30_dice'])
                bottom_10 = np.array(self.dice_history['bottom_10_dice'])

                f.write("\nSMALL STRUCTURE PERFORMANCE\n")
                f.write("-" * 40 + "\n")
                f.write(f"Bottom 30% classes:\n")
                f.write(f"  Current dice: {bottom_30[-1]:.4f}\n")
                f.write(f"  Best dice: {np.max(bottom_30):.4f}\n")
                f.write(f"  Improvement: {bottom_30[-1] - bottom_30[0]:.4f}\n")

                f.write(f"\nBottom 10 classes:\n")
                f.write(f"  Current dice: {bottom_10[-1]:.4f}\n")
                f.write(f"  Best dice: {np.max(bottom_10):.4f}\n")
                f.write(f"  Improvement: {bottom_10[-1] - bottom_10[0]:.4f}\n")

            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")

            if self.dice_history['dyn_spec_losses'] and len(dyn_spec) > 20:
                recent_dyn = dyn_spec[-10:].mean()
                if recent_dyn > 0.1:
                    f.write("• Dynamic loss is high - consider:\n")
                    f.write("  - Reducing dyn_top_k to focus on lower frequencies\n")
                    f.write("  - Increasing dyn_ramp_epochs for slower integration\n")

                if np.sum(dyn_conflicts[-10:]) > 3:
                    f.write("• Frequent conflicts detected - consider:\n")
                    f.write("  - Strengthening restricted mask constraints\n")
                    f.write("  - Reducing lambda_dyn temporarily\n")

            if self.dice_history['bottom_10_dice']:
                if bottom_10[-1] < 0.5:
                    f.write("• Small structures underperforming - consider:\n")
                    f.write("  - Increasing qap_mismatch_g for stronger penalties\n")
                    f.write("  - Adjusting class weights more aggressively\n")

        print(f"📊 Dual-branch report saved to: {self.dual_branch_report_path}")

    def plot_dice_evolution(self) -> None:
        """Generate plots for dice score evolution with dual-branch metrics"""
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if not is_main or not self.dice_history['epochs']:
            return

        epochs = np.array(self.dice_history['epochs'])
        dice_scores = np.array(self.dice_history['dice_scores'])
        seg_losses = np.array(self.dice_history['seg_losses'])

        # Determine number of subplots based on available data
        n_plots = 3  # Base: dice, seg loss, domain/other
        if self.dual_branch_enabled and self.dice_history['dyn_spec_losses']:
            n_plots = 6  # Add dual-branch specific plots
        elif self.cross_domain_enabled and self.dice_history['graph_spec_src_losses']:
            n_plots = 5  # Add source/target alignment plots

        # Create figure with subplots
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
        axes = list(axes) if n_plots > 1 else [axes]

        # Plot 1: Dice score evolution
        ax1 = axes[0]
        ax1.plot(epochs, dice_scores, 'b-', linewidth=2, label='Dice Score')

        # Mark drops
        for drop in self.dice_history['drops_detected']:
            ax1.axvline(x=drop['epoch'], color='r', linestyle='--', alpha=0.5)

        # Add moving average
        if len(dice_scores) >= self.window_size:
            ma = np.convolve(dice_scores, np.ones(self.window_size) / self.window_size, mode='valid')
            ax1.plot(epochs[self.window_size - 1:], ma, 'g--', alpha=0.7, label=f'{self.window_size}-epoch MA')

        # Mark training stages if dual-branch
        if self.dual_branch_enabled and self.dice_history['dyn_lambda_values']:
            dyn_lambda = np.array(self.dice_history['dyn_lambda_values'])
            active_idx = np.where(dyn_lambda > 0)[0]
            if len(active_idx) > 0:
                stage_a_end = epochs[active_idx[0]]
                ax1.axvline(x=stage_a_end, color='g', linestyle='--', alpha=0.5, label='Dynamic Start')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Dice Score')
        ax1.set_title('Dice Score Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Segmentation loss
        ax2 = axes[1]
        ax2.plot(epochs, seg_losses, 'r-', linewidth=2, label='Seg Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Segmentation Loss')
        ax2.set_title('Segmentation Loss Evolution')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Graph losses or domain accuracy
        ax3 = axes[2]
        if self.graph_prior_enabled and self.dice_history['graph_losses']:
            graph_losses = np.array(self.dice_history['graph_losses'])
            ax3.plot(epochs, graph_losses, 'g-', linewidth=2, label='Graph Loss')
            ax3.set_ylabel('Graph Loss')
            ax3.set_title('Graph Prior Loss Evolution')
        else:
            dom_accs = np.array(self.dice_history['domain_accs'])
            ax3.plot(epochs, dom_accs, 'g-', linewidth=2)
            ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Perfect confusion')
            ax3.fill_between(epochs, 0.4, 0.6, alpha=0.2, color='green')
            ax3.set_ylabel('Domain Accuracy')
            ax3.set_title('Domain Classifier Performance')

        ax3.set_xlabel('Epoch')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Additional plots for dual-branch
        if self.dual_branch_enabled and n_plots >= 6:
            # Plot 4: Dual-branch losses
            ax4 = axes[3]
            if self.dice_history['graph_losses']:
                graph_losses = np.array(self.dice_history['graph_losses'])
                ax4.plot(epochs, graph_losses, 'r-', label='Prior Branch', linewidth=2)
            if self.dice_history['dyn_spec_losses']:
                dyn_spec = np.array(self.dice_history['dyn_spec_losses'])
                ax4.plot(epochs[:len(dyn_spec)], dyn_spec, 'b-', label='Dynamic Branch', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Dual-Branch Loss Evolution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Plot 5: Conflict signals
            ax5 = axes[4]
            if self.dice_history['forbidden_ema']:
                forbidden_ema = np.array(self.dice_history['forbidden_ema'])
                ax5.plot(epochs[:len(forbidden_ema)], forbidden_ema, 'r-', label='Forbidden EMA', linewidth=2)
            if self.dice_history['conflict_signals']:
                conflicts = np.array(self.dice_history['conflict_signals'])
                conflict_epochs = epochs[np.where(conflicts > 0)[0]]
                for ce in conflict_epochs:
                    ax5.axvline(x=ce, color='orange', alpha=0.3)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Violations')
            ax5.set_title('Conflict Detection')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # Plot 6: Small structure performance
            ax6 = axes[5]
            if self.dice_history['bottom_30_dice']:
                bottom_30 = np.array(self.dice_history['bottom_30_dice'])
                bottom_10 = np.array(self.dice_history['bottom_10_dice'])
                ax6.plot(epochs[:len(bottom_30)], bottom_30, 'b-', label='Bottom 30%', linewidth=2)
                ax6.plot(epochs[:len(bottom_10)], bottom_10, 'r-', label='Bottom 10', linewidth=2)
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Dice Score')
            ax6.set_title('Small Structure Performance')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # Additional plots for cross-domain (if not dual-branch)
        elif self.cross_domain_enabled and self.dice_history['graph_spec_src_losses'] and n_plots >= 5:
            spec_src = np.array(self.dice_history['graph_spec_src_losses'])
            edge_src = np.array(self.dice_history['graph_edge_src_losses'])
            spec_tgt = np.array(self.dice_history['graph_spec_tgt_losses'])
            edge_tgt = np.array(self.dice_history['graph_edge_tgt_losses'])

            # Plot 4: Source domain alignment
            ax4 = axes[3]
            ax4.plot(epochs[:len(spec_src)], spec_src, 'b-', label='Spectral', linewidth=2)
            ax4.plot(epochs[:len(edge_src)], edge_src, 'r-', label='Edge', linewidth=2)
            ax4.plot(epochs[:len(spec_src)], spec_src + edge_src, 'g--', label='Total', linewidth=2, alpha=0.7)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Source Domain Alignment (dHCP)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Plot 5: Target domain regularization
            ax5 = axes[4]
            ax5.plot(epochs[:len(spec_tgt)], spec_tgt, 'b-', label='Spectral', linewidth=2)
            ax5.plot(epochs[:len(edge_tgt)], edge_tgt, 'r-', label='Edge', linewidth=2)
            ax5.plot(epochs[:len(spec_tgt)], spec_tgt + edge_tgt, 'g--', label='Total', linewidth=2, alpha=0.7)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Loss')
            ax5.set_title('Target Domain Regularization (PPREMO/PREBO)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.dual_branch_enabled and self.dual_branch_dir:
            plot_path = os.path.join(self.dual_branch_dir, 'dual_branch_evolution.png')
        else:
            plot_path = os.path.join(self.monitor_dir, 'dice_evolution.png')

        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"📈 Dice evolution plot saved to: {plot_path}")

    def get_report_path(self) -> str:
        """Get path to the report file"""
        return self.report_path

    def get_latest_dice(self) -> Optional[float]:
        """Get the latest dice score"""
        if self.dice_history['dice_scores']:
            return self.dice_history['dice_scores'][-1]
        return None

    def get_small_structure_performance(self, num_classes: int = 30) -> Dict:
        """Analyze performance on small structures (bottom N classes by dice)"""
        if not self.dice_history['per_class_dice'] or not self.dice_history['per_class_dice'][-1]:
            return {}

        latest_per_class = self.dice_history['per_class_dice'][-1]
        sorted_indices = np.argsort(latest_per_class)

        small_class_indices = sorted_indices[:num_classes]
        small_class_dice = [latest_per_class[idx] for idx in small_class_indices]

        return {
            'mean_dice': np.mean(small_class_dice),
            'min_dice': np.min(small_class_dice),
            'max_dice': np.max(small_class_dice),
            'std_dice': np.std(small_class_dice),
            'indices': small_class_indices.tolist(),
            'scores': small_class_dice
        }