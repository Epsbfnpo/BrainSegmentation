"""Utility for tracking Dice scores and graph-alignment diagnostics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.distributed as dist


@dataclass
class _DropEvent:
    epoch: int
    dice: float
    timestamp: str


class DiceMonitor:
    """Persist training/validation Dice trends and graph-alignment metrics."""

    def __init__(
        self,
        results_dir: str,
        drop_threshold: float = 0.3,
        window_size: int = 5,
        graph_prior_enabled: bool = False,
        cross_domain_enabled: bool = False,
    ) -> None:
        self.results_dir = results_dir
        self.drop_threshold = drop_threshold
        self.window_size = window_size
        self.graph_prior_enabled = graph_prior_enabled
        self.cross_domain_enabled = cross_domain_enabled

        is_main = (not dist.is_initialized()) or dist.get_rank() == 0
        if is_main:
            self.monitor_dir = os.path.join(results_dir, "dice_monitoring")
            os.makedirs(self.monitor_dir, exist_ok=True)
            if graph_prior_enabled:
                self.graph_dir = os.path.join(results_dir, "graph_analysis")
                os.makedirs(self.graph_dir, exist_ok=True)
            else:
                self.graph_dir = None
        else:
            self.monitor_dir = None
            self.graph_dir = None

        self.report_path = (
            os.path.join(self.monitor_dir, "dice_report.txt") if is_main else None
        )
        self.json_path = (
            os.path.join(self.monitor_dir, "dice_history.json") if is_main else None
        )
        self.csv_path = (
            os.path.join(self.monitor_dir, "dice_history.csv") if is_main else None
        )
        self.crossdomain_report_path = (
            os.path.join(self.graph_dir, "crossdomain_alignment_report.txt")
            if is_main and self.graph_dir and cross_domain_enabled
            else None
        )

        self.dice_history: Dict[str, List] = {
            "epochs": [],
            "dice_scores": [],
            "seg_losses": [],
            "domain_losses": [],
            "domain_accs": [],
            "learning_rates": [],
            "alphas": [],
            "timestamps": [],
            "per_class_dice": [],
            "drops_detected": [],
            "graph_losses": [],
            "graph_spec_losses": [],
            "graph_edge_losses": [],
            "graph_sym_losses": [],
            "graph_struct_losses": [],
            "graph_spec_src_losses": [],
            "graph_edge_src_losses": [],
            "graph_spec_tgt_losses": [],
            "graph_edge_tgt_losses": [],
            "structural_violations": [],
            "symmetry_scores": [],
            "adjacency_errors": [],
            "adjacency_errors_src": [],
            "bottom_30_dice": [],
            "bottom_10_dice": [],
            "forbidden_ema": [],
            "required_ema": [],
            "conflict_signals": [],
        }

        self._forbidden_ema = 0.0
        self._required_ema = 0.0
        self._ema_alpha = 0.3

    # ------------------------------------------------------------------
    def add_dice_score(
        self, epoch: int, dice_score: float, train_metrics: Dict, val_metrics: Dict
    ) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        self.dice_history["epochs"].append(epoch)
        self.dice_history["dice_scores"].append(dice_score)
        self.dice_history["seg_losses"].append(train_metrics.get("seg_loss", 0.0))
        self.dice_history["domain_losses"].append(train_metrics.get("domain_loss", 0.0))
        self.dice_history["domain_accs"].append(train_metrics.get("domain_acc", 0.0))
        self.dice_history["learning_rates"].append(train_metrics.get("lr", 0.0))
        self.dice_history["alphas"].append(train_metrics.get("alpha", 0.0))
        self.dice_history["timestamps"].append(datetime.now().isoformat())
        self.dice_history["per_class_dice"].append(
            val_metrics.get("dice_per_class", [])
        )

        if self.graph_prior_enabled:
            graph_loss = train_metrics.get("graph_loss", train_metrics.get("graph_total", 0.0))
            self.dice_history["graph_losses"].append(graph_loss)
            self.dice_history["graph_spec_losses"].append(
                train_metrics.get("graph_spec", train_metrics.get("graph_spec_src", 0.0))
            )
            self.dice_history["graph_edge_losses"].append(
                train_metrics.get("graph_edge", train_metrics.get("graph_edge_src", 0.0))
            )
            self.dice_history["graph_sym_losses"].append(train_metrics.get("graph_sym", 0.0))
            self.dice_history["graph_struct_losses"].append(train_metrics.get("graph_struct", 0.0))

            if self.cross_domain_enabled:
                self.dice_history["graph_spec_src_losses"].append(
                    train_metrics.get("graph_spec_src", 0.0)
                )
                self.dice_history["graph_edge_src_losses"].append(
                    train_metrics.get("graph_edge_src", 0.0)
                )
                self.dice_history["graph_spec_tgt_losses"].append(
                    train_metrics.get("graph_spec_tgt", 0.0)
                )
                self.dice_history["graph_edge_tgt_losses"].append(
                    train_metrics.get("graph_edge_tgt", 0.0)
                )

            violations = self._compute_structural_violations(val_metrics)
            self.dice_history["structural_violations"].append(violations)

            self._forbidden_ema = (
                self._ema_alpha * violations.get("forbidden_present", 0)
                + (1 - self._ema_alpha) * self._forbidden_ema
            )
            self._required_ema = (
                self._ema_alpha * violations.get("required_missing", 0)
                + (1 - self._ema_alpha) * self._required_ema
            )
            self.dice_history["forbidden_ema"].append(self._forbidden_ema)
            self.dice_history["required_ema"].append(self._required_ema)

            conflict_signal = int(
                len(self.dice_history["forbidden_ema"]) >= 3
                and self.dice_history["forbidden_ema"][-1]
                > 1.1 * self.dice_history["forbidden_ema"][-3]
            )
            self.dice_history["conflict_signals"].append(conflict_signal)

            self.dice_history["symmetry_scores"].append(
                self._compute_symmetry_scores(val_metrics)
            )
            self.dice_history["adjacency_errors"].append(
                self._compute_adjacency_errors(val_metrics)
            )
            if val_metrics.get("adjacency_errors_src"):
                self.dice_history["adjacency_errors_src"].append(
                    val_metrics["adjacency_errors_src"]
                )

        dice_per_class = val_metrics.get("dice_per_class")
        if dice_per_class:
            sorted_vals = np.sort(np.asarray(dice_per_class, dtype=np.float32))
            bottom_30 = int(len(sorted_vals) * 0.3)
            bottom_30 = max(bottom_30, 1)
            self.dice_history["bottom_30_dice"].append(float(sorted_vals[:bottom_30].mean()))
            count = min(len(sorted_vals), 10)
            self.dice_history["bottom_10_dice"].append(float(sorted_vals[:count].mean()))
        else:
            self.dice_history["bottom_30_dice"].append(0.0)
            self.dice_history["bottom_10_dice"].append(0.0)

        self.save_history()

        if self.check_dice_drop():
            self.dice_history["drops_detected"].append(
                _DropEvent(epoch, dice_score, datetime.now().isoformat()).__dict__
            )

    # ------------------------------------------------------------------
    def check_dice_drop(self) -> bool:
        scores = self.dice_history["dice_scores"]
        if len(scores) < self.window_size + 1:
            return False
        recent = np.array(scores[-self.window_size :], dtype=np.float32)
        previous = np.array(scores[-(self.window_size + 1) : -1], dtype=np.float32)
        if not len(previous):
            return False
        drop = previous.mean() - recent.mean()
        return drop > self.drop_threshold

    # ------------------------------------------------------------------
    def save_history(self) -> None:
        if not self.json_path:
            return
        with open(self.json_path, "w") as f:
            json.dump(self.dice_history, f, indent=2)

        if self.csv_path:
            rows = []
            for idx, epoch in enumerate(self.dice_history["epochs"]):
                row = {key: self.dice_history[key][idx] for key in self._scalar_keys() if idx < len(self.dice_history[key])}
                row["epoch"] = epoch
                row["dice_score"] = self.dice_history["dice_scores"][idx]
                rows.append(row)
            if rows:
                pd.DataFrame(rows).to_csv(self.csv_path, index=False)

    # ------------------------------------------------------------------
    def generate_report(self) -> None:
        if not self.report_path:
            return
        with open(self.report_path, "w") as f:
            if not self.dice_history["epochs"]:
                f.write("No dice history recorded.\n")
                return
            best_idx = int(np.argmax(self.dice_history["dice_scores"]))
            f.write("Dice Monitor Summary\n")
            f.write("=" * 60 + "\n")
            f.write(
                f"Best Dice: {self.dice_history['dice_scores'][best_idx]:.4f} at epoch {self.dice_history['epochs'][best_idx]}\n"
            )
            f.write(
                f"Last Dice: {self.dice_history['dice_scores'][-1]:.4f} (epoch {self.dice_history['epochs'][-1]})\n"
            )
            if self.dice_history["drops_detected"]:
                f.write("\nRecent Drops:\n")
                for event in self.dice_history["drops_detected"][-3:]:
                    f.write(
                        f"  Epoch {event['epoch']}: dice {event['dice']:.4f} at {event['timestamp']}\n"
                    )
            if self.graph_prior_enabled and self.dice_history["graph_losses"]:
                f.write("\nGraph alignment (last epoch):\n")
                f.write(
                    f"  Spectral: {self.dice_history['graph_spec_losses'][-1]:.4f}\n"
                )
                f.write(
                    f"  Edge: {self.dice_history['graph_edge_losses'][-1]:.4f}\n"
                )
                f.write(
                    f"  Symmetry: {self.dice_history['graph_sym_losses'][-1]:.4f}\n"
                )

    # ------------------------------------------------------------------
    def generate_cross_domain_report(self) -> None:
        if not self.crossdomain_report_path or not self.dice_history["graph_spec_src_losses"]:
            return
        with open(self.crossdomain_report_path, "w") as f:
            f.write("Cross-domain graph alignment summary\n")
            f.write("=" * 60 + "\n")
            last_src = self.dice_history["graph_spec_src_losses"][-5:]
            last_tgt = self.dice_history["graph_spec_tgt_losses"][-5:]
            f.write(f"Source spectral (last 5 avg): {np.mean(last_src):.4f}\n")
            f.write(f"Target spectral (last 5 avg): {np.mean(last_tgt):.4f}\n")
            last_src_edge = self.dice_history["graph_edge_src_losses"][-5:]
            last_tgt_edge = self.dice_history["graph_edge_tgt_losses"][-5:]
            f.write(f"Source edge (last 5 avg): {np.mean(last_src_edge):.4f}\n")
            f.write(f"Target edge (last 5 avg): {np.mean(last_tgt_edge):.4f}\n")

    # ------------------------------------------------------------------
    def plot_dice_evolution(self) -> None:
        if not self.monitor_dir or not self.dice_history["epochs"]:
            return
        epochs = np.array(self.dice_history["epochs"], dtype=np.float32)
        dice = np.array(self.dice_history["dice_scores"], dtype=np.float32)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, dice, label="Dice", linewidth=2)
        if self.graph_prior_enabled and self.dice_history["graph_spec_losses"]:
            plt.twinx()
            plt.plot(
                epochs[: len(self.dice_history["graph_spec_losses"])],
                self.dice_history["graph_spec_losses"],
                color="orange",
                label="Spectral loss",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.title("Dice evolution")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(self.monitor_dir, "dice_evolution.png"))
        plt.close()

    # ------------------------------------------------------------------
    def _compute_structural_violations(self, val_metrics: Dict) -> Dict:
        return val_metrics.get(
            "structural_violations",
            {
                "required_missing": 0,
                "forbidden_present": 0,
                "containment_violated": 0,
                "exclusivity_violated": 0,
            },
        )

    def _compute_symmetry_scores(self, val_metrics: Dict) -> Dict:
        return val_metrics.get(
            "symmetry_scores",
            {"mean_symmetry": 0.0, "median_symmetry": 0.0},
        )

    def _compute_adjacency_errors(self, val_metrics: Dict) -> Dict:
        return val_metrics.get(
            "adjacency_errors",
            {"required": 0.0, "forbidden": 0.0},
        )

    def _scalar_keys(self) -> List[str]:
        return [
            "seg_losses",
            "domain_losses",
            "domain_accs",
            "learning_rates",
            "alphas",
            "graph_losses",
            "graph_spec_losses",
            "graph_edge_losses",
            "graph_sym_losses",
            "graph_struct_losses",
            "graph_spec_src_losses",
            "graph_edge_src_losses",
            "graph_spec_tgt_losses",
            "graph_edge_tgt_losses",
            "bottom_30_dice",
            "bottom_10_dice",
            "forbidden_ema",
            "required_ema",
            "conflict_signals",
        ]

