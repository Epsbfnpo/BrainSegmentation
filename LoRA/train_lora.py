#!/usr/bin/env python3
"""Target-only training entrypoint with production diagnostics."""

import argparse
import json
import os
import random
import signal
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from monai.networks.nets import SwinUNETR
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, TaskType

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib may be unavailable on some systems
    plt = None

from age_aware_modules import SimplifiedDAUnetModule
from data_loader_age_aware import get_target_dataloaders
from trainer_lora import (CombinedSegmentationLoss, ExponentialMovingAverage,
                          train_epoch, validate_epoch)


def parse_slurm_timelimit(raw: Optional[str]) -> Optional[float]:
    """Parse SLURM time limit strings into seconds."""
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    days = 0
    time_part = raw
    if "-" in raw:
        day_str, time_part = raw.split("-", 1)
        try:
            days = int(day_str)
        except ValueError:
            return None
    parts = time_part.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            hours, minutes, seconds = 0, int(parts[0]), int(parts[1])
        elif len(parts) == 1:
            # Interpret single integer as minutes
            hours, minutes, seconds = 0, int(parts[0]), 0
        else:
            return None
    except ValueError:
        return None
    total_minutes = (days * 24 * 60) + (hours * 60) + minutes
    return float(total_minutes * 60 + seconds)


def compute_job_deadline(buffer_seconds: float) -> Optional[float]:
    """Return UNIX timestamp when training should finish (buffer already subtracted)."""
    end_time_env = os.environ.get("SLURM_JOB_END_TIME")
    if end_time_env:
        try:
            end_time = float(end_time_env)
            return end_time - buffer_seconds
        except ValueError:
            pass
    start_env = os.environ.get("SLURM_JOB_START_TIME")
    limit_env = os.environ.get("SLURM_JOB_TIME_LIMIT") or os.environ.get("SLURM_TIMELIMIT")
    if start_env and limit_env:
        try:
            start_time = float(start_env)
        except ValueError:
            start_time = None
        time_limit = parse_slurm_timelimit(limit_env)
        if start_time is not None and time_limit is not None:
            return start_time + time_limit - buffer_seconds
    return None


def record_resume_checkpoint(results_dir: Path, checkpoint_path: Path) -> None:
    resume_file = results_dir / "resume_from.txt"
    checkpoint_path = checkpoint_path.resolve()
    resume_file.write_text(str(checkpoint_path), encoding="utf-8")


def init_distributed(args) -> bool:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", init_method="env://", timeout=timedelta(minutes=args.dist_timeout))
        torch.cuda.set_device(args.local_rank)
        return True
    args.rank = 0
    args.world_size = 1
    args.local_rank = 0
    return False


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args) -> bool:
    return getattr(args, "rank", 0) == 0


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sanitize_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (int, float)):
        return float(value)
    return value


def load_metrics_history(path: Path) -> Dict[str, list]:
    if not path.exists():
        return {"train": [], "val": []}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"train": [], "val": []}
    if not isinstance(data, dict):
        return {"train": [], "val": []}
    data.setdefault("train", [])
    data.setdefault("val", [])
    return data


def save_metrics_history(history: Dict[str, list], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def update_metrics_history(history: Dict[str, list],
                           split: str,
                           metrics: Dict,
                           path: Path,
                           is_main: bool) -> None:
    split = "val" if split not in ("train", "val") else split
    sanitized = {k: _sanitize_value(v) for k, v in metrics.items() if v is not None}
    sanitized.setdefault("timestamp", time.time())
    history.setdefault(split, []).append(sanitized)
    if is_main:
        save_metrics_history(history, path)


def _extract_series(entries, key: str):
    series = []
    for entry in entries:
        epoch = entry.get("epoch")
        value = entry.get(key)
        if epoch is None or value is None:
            continue
        series.append((epoch, value))
    series.sort(key=lambda item: item[0])
    if not series:
        return [], []
    epochs, values = zip(*series)
    return list(epochs), list(values)


def _plot_series(entries, keys, outfile: Path, title: str, ylabel: str):
    if plt is None or not entries:
        return False
    plotted = False
    plt.figure(figsize=(10, 5))
    for key in keys:
        xs, ys = _extract_series(entries, key)
        if xs and ys:
            plt.plot(xs, ys, label=key)
            plotted = True
    if not plotted:
        plt.close()
        return False
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    return True


def generate_training_plots(history: Dict[str, list], results_dir: Path) -> bool:
    if plt is None:
        print("⚠️  matplotlib is not available; skipping training plot generation.")
        return False
    plot_dir = results_dir / "analysis"
    train_entries = history.get("train", [])
    val_entries = history.get("val", [])
    generated = False
    if _plot_series(
        train_entries,
        ["loss", "seg", "dice", "ce", "focal"],
        plot_dir / "train_seg_losses.png",
        "Training segmentation losses",
        "Loss",
    ):
        generated = True

    val_metric_groups = [
        (["dice", "cldice", "cbdice"], "Overlap Metrics (Dice)", "Score (0-1)"),
        (["hd95", "assd", "clce"], "Distance and CE Metrics", "Value (lower is better)"),
        (["rve", "val_adj_mae", "val_spec_dist"], "Geometric and Spectral Errors", "Error"),
        (["val_sym_score"], "Symmetry Score", "Score (0-1)"),
    ]

    if val_entries:
        for keys, title, ylabel in val_metric_groups:
            safe_filename = (
                title.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("&", "and")
            )
            if _plot_series(val_entries, keys, plot_dir / f"val_{safe_filename}.png", title, ylabel):
                generated = True

    if train_entries or val_entries:
        if plt is not None:
            plt.figure(figsize=(10, 5))
            plotted = False
            xs, ys = _extract_series(train_entries, "dice")
            if xs and ys:
                plt.plot(xs, ys, label="train_dice")
                plotted = True
            xs, ys = _extract_series(val_entries, "dice")
            if xs and ys:
                plt.plot(xs, ys, label="val_dice")
                plotted = True
            if plotted:
                plt.title("Dice over epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Dice")
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plot_dir.mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(plot_dir / "dice_history.png")
                generated = True
            plt.close()
    if not generated:
        print("⚠️  No training metrics available for plotting yet.")
    return generated


def load_class_mapping(path: Optional[Path]) -> Dict[int, int]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    mapping = {}
    data = payload.get("index_to_raw_label") if isinstance(payload, dict) else None
    if isinstance(data, dict):
        for key, value in data.items():
            try:
                mapping[int(key)] = int(value)
            except (TypeError, ValueError):
                continue
    return mapping


def save_per_class_report(per_class_scores, class_mapping: Dict[int, int], results_dir: Path):
    if not per_class_scores:
        return None
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, score in enumerate(per_class_scores):
        raw_label = class_mapping.get(idx, idx + 1)
        records.append({
            "remapped_index": int(idx),
            "raw_label": int(raw_label),
            "dice": float(score),
        })
    json_path = analysis_dir / "best_model_per_class_dice.json"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    if plt is None:
        return {"json": json_path}
    labels = [str(item["raw_label"]) for item in records]
    values = [item["dice"] for item in records]
    width = max(16.0, len(values) * 0.15)
    plt.figure(figsize=(width, 6))
    plt.bar(range(len(values)), values, color="#2878B5")
    plt.xticks(range(len(values)), labels, rotation=90, fontsize=6)
    plt.xlabel("Raw label ID")
    plt.ylabel("Dice")
    plt.title("Best-model Dice per raw region")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    png_path = analysis_dir / "best_model_per_class_dice.png"
    plt.savefig(png_path)
    plt.close()
    return {"json": json_path, "png": png_path}


def load_model_weights_only(model: torch.nn.Module, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("state_dict", payload)
    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    target.load_state_dict(state, strict=True)


def build_model(args, device: torch.device) -> SimplifiedDAUnetModule:
    backbone = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_swin_checkpoint,
    ).to(device)
    wrapper = SimplifiedDAUnetModule(
        backbone,
        num_classes=args.out_channels,
        volume_stats_path=None,
        foreground_only=args.foreground_only,
        enhanced_class_weights=args.enhanced_class_weights,
        use_age_conditioning=False,
        debug_mode=args.debug_mode,
    )
    return wrapper


def get_model_state(model: torch.nn.Module) -> dict:
    return model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()


def save_checkpoint(path: Path,
                    *,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler,
                    epoch: int,
                    global_step: int,
                    best_dice: float,
                    ema_state: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_dice": best_dice,
            "state_dict": get_model_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema_state_dict": ema_state,
        },
        path,
    )


def load_checkpoint(path: Path,
                    *,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler):
    payload = torch.load(path, map_location="cpu")
    model_state = payload.get("state_dict", payload)
    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    target.load_state_dict(model_state, strict=True)

    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])

    return (
        payload.get("epoch", 0),
        payload.get("global_step", 0),
        payload.get("best_dice", 0.0),
        payload.get("ema_state_dict"),
    )


def register_signal_handlers(flag_container: dict, *, is_main: bool) -> None:
    flag_container.setdefault("triggered", False)
    flag_container.setdefault("stop_requested", False)
    flag_container.setdefault("signal", None)
    flag_container.setdefault("timestamp", None)

    def _handler(signum, frame):
        if is_main:
            print(
                f"⚠️  Received signal {signum}; requesting graceful shutdown after checkpoint",
                flush=True,
            )
        flag_container["triggered"] = True
        flag_container["stop_requested"] = True
        flag_container["signal"] = signum
        flag_container["timestamp"] = time.time()

    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, _handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Target-only age-aware segmentation training")
    parser.add_argument("--split_json", required=True, type=str, help="Target dataset split JSON")
    parser.add_argument("--results_dir", default="./results", type=str)
    parser.add_argument("--log_dir", default=None, type=str, help="TensorBoard log directory")
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--cache_rate", default=0.0, type=float)
    parser.add_argument("--cache_num_workers", default=4, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=87, type=int)
    parser.add_argument("--feature_size", default=48, type=int)
    parser.add_argument("--roi_x", default=96, type=int)
    parser.add_argument("--roi_y", default=96, type=int)
    parser.add_argument("--roi_z", default=96, type=int)
    parser.add_argument("--apply_spacing", dest="apply_spacing", action="store_true", default=True)
    parser.add_argument("--no_apply_spacing", dest="apply_spacing", action="store_false")
    parser.add_argument("--apply_orientation", dest="apply_orientation", action="store_true", default=True)
    parser.add_argument("--no_apply_orientation", dest="apply_orientation", action="store_false")
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[0.8, 0.8, 0.8])
    parser.add_argument("--foreground_only", action="store_true", default=True)
    parser.add_argument("--enhanced_class_weights", action="store_true", default=True)
    parser.add_argument("--pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--laterality_pairs_json", type=str, default=None)
    parser.add_argument("--use_label_crop", dest="use_label_crop", action="store_true", default=True)
    parser.add_argument("--no_label_crop", dest="use_label_crop", action="store_false")
    parser.add_argument("--label_crop_samples", type=int, default=1)
    parser.add_argument("--enable_weighted_sampling", action="store_true", default=False)
    parser.add_argument("--loss_config", type=str, default="dice_focal", choices=["dice_ce", "dice_focal", "dice_ce_focal"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--class_map_json", type=str, default=None,
                        help="JSON file containing index_to_raw_label mapping for reporting")
    parser.add_argument("--lr_min", type=float, default=1e-7)
    parser.add_argument("--lr_warmup_epochs", type=int, default=120)
    parser.add_argument("--lr_warmup_start_factor", type=float, default=0.1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--ema_decay", type=float, default=0.0)
    parser.add_argument("--dist_timeout", type=int, default=180)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--use_swin_checkpoint", dest="use_swin_checkpoint", action="store_true", default=True)
    parser.add_argument("--no_swin_checkpoint", dest="use_swin_checkpoint", action="store_false")
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_sliding_window", dest="use_sliding_window", action="store_true", default=True)
    parser.add_argument("--no_eval_sliding_window", dest="use_sliding_window", action="store_false")
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--sw_overlap", type=float, default=0.25)
    parser.add_argument("--multi_scale_eval", action="store_true", default=False)
    parser.add_argument("--eval_scales", type=float, nargs="*", default=[1.0])
    parser.add_argument("--eval_tta", action="store_true", default=False)
    parser.add_argument("--tta_flip_axes", type=int, nargs="*", default=[0])
    parser.add_argument("--grad_clip", type=float, default=12.0)
    parser.add_argument("--save_interval", type=int, default=50, help="Save checkpoint every N epochs")
    parser.add_argument("--max_keep_ckpt", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--epoch_time_buffer", type=float, default=600.0,
                        help="Minimum seconds required to start a new epoch")
    parser.add_argument("--slurm_time_buffer", type=float, default=300.0,
                        help="Seconds to reserve before job termination when estimating deadline")
    parser.add_argument("--debug_mode", action="store_true", default=False)
    parser.add_argument("--debug_step_limit", type=int, default=2)
    parser.add_argument("--debug_val_limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "metrics_history.json"
    history = load_metrics_history(history_path)

    distributed = init_distributed(args)
    is_main = is_main_process(args)

    log_dir = Path(args.log_dir) if args.log_dir else results_dir / "tensorboard"
    writer: Optional[SummaryWriter] = SummaryWriter(log_dir=str(log_dir)) if is_main else None

    signal_state = {"triggered": False, "stop_requested": False, "signal": None, "timestamp": None}
    register_signal_handlers(signal_state, is_main=is_main)

    job_deadline = compute_job_deadline(float(args.slurm_time_buffer))
    if job_deadline is not None and is_main:
        remaining = job_deadline - time.time()
        print(f"⏱️  Detected SLURM deadline in {max(0.0, remaining):.1f} seconds (buffer applied)")

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        args.use_amp = False
    class_map_path: Optional[Path] = Path(args.class_map_json).resolve() if args.class_map_json else None
    class_mapping = load_class_mapping(class_map_path) if class_map_path is not None else {}

    train_loader, val_loader = get_target_dataloaders(
        args,
        is_distributed=distributed,
        world_size=args.world_size,
        rank=args.rank,
    )

    model = build_model(args, device)
    if args.pretrained_checkpoint:
        if is_main:
            print(f"📦 Loading pretrained weights for LoRA base from {args.pretrained_checkpoint}")
        load_model_weights_only(model, Path(args.pretrained_checkpoint))

    class_weights = model.get_class_weights()

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["qkv", "proj", "linear1", "linear2"],
        modules_to_save=["out"],
    )

    model = get_peft_model(model, peft_config)

    if is_main:
        model.print_trainable_parameters()

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    ema_helper: Optional[ExponentialMovingAverage] = None
    if args.ema_decay > 0.0:
        if is_main:
            print(f"📈 EMA enabled with decay={args.ema_decay}")
        ema_helper = ExponentialMovingAverage(model, decay=args.ema_decay)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_epochs = max(0, int(args.lr_warmup_epochs))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(args.lr_warmup_start_factor),
            total_iters=warmup_epochs,
        )
        cosine_iters = max(1, args.epochs - warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_iters,
            eta_min=float(args.lr_min),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=float(args.lr_min),
        )

    loss_fn = CombinedSegmentationLoss(
        num_classes=args.out_channels,
        class_weights=class_weights,
        foreground_only=args.foreground_only,
        loss_config=args.loss_config,
        focal_gamma=args.focal_gamma,
    )

    best_dice = 0.0
    global_step = 0
    start_epoch = 1
    checkpoint_history: deque[Path] = deque()
    last_checkpoint_path: Optional[Path] = None
    last_completed_epoch = 0
    time_limit_exhausted = False
    latest_model_path = results_dir / "latest_model.pt"
    final_model_path = results_dir / "final_model.pt"
    best_checkpoint_path = results_dir / "best_model.pt"
    best_model_epoch: Optional[int] = None

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            resume_epoch, resume_step, resume_best, ema_state = load_checkpoint(
                resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            start_epoch = resume_epoch + 1
            global_step = resume_step
            best_dice = resume_best
            if ema_helper is not None and ema_state is not None:
                ema_helper.load_state_dict(ema_state)
            if is_main:
                print(f"🔁 Resumed from {resume_path} at epoch {resume_epoch}, global_step {global_step}, best dice {best_dice:.4f}")
        elif is_main:
            print(f"⚠️  Resume checkpoint {resume_path} not found; starting from scratch")

    if is_main:
        print(f"📂 Results directory: {results_dir}")
        print(f"📝 Logging to: {log_dir}")
        if class_mapping:
            print(f"🗺️  Loaded class mapping for {len(class_mapping)} regions from {class_map_path}")
        else:
            print("⚠️  No class map found; per-class reports will use remapped indices only")

    for epoch in range(start_epoch, args.epochs + 1):
        if signal_state.get("stop_requested"):
            if is_main:
                print(
                    f"🛑 Stop requested by external signal before epoch {epoch:03d}; exiting loop",
                    flush=True,
                )
            time_limit_exhausted = True
            break
        if job_deadline is not None:
            time_remaining = job_deadline - time.time()
            if time_remaining < float(args.epoch_time_buffer):
                if is_main:
                    print(
                        f"⏳ Remaining time {max(0.0, time_remaining):.1f}s is below buffer "
                        f"({args.epoch_time_buffer}s); stopping before epoch {epoch:03d}",
                        flush=True,
                    )
                time_limit_exhausted = True
                break
            elif is_main and epoch == start_epoch:
                print(f"🔁 Starting epoch loop with ~{time_remaining:.1f}s available", flush=True)
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        start_time = time.time()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=device,
            epoch=epoch,
            use_amp=args.use_amp,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            writer=writer,
            global_step=global_step,
            is_main=is_main,
            log_interval=args.log_interval,
            debug_mode=args.debug_mode,
            debug_step_limit=args.debug_step_limit,
            ema_helper=ema_helper,
        )
        scheduler.step()
        global_step = int(train_metrics.get("global_step", global_step))
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics["lr"] = current_lr
        duration = time.time() - start_time
        last_completed_epoch = epoch

        if is_main:
            seg_msg = (
                f"loss={train_metrics['loss']:.4f} seg={train_metrics['seg']:.4f} "
                f"dice={train_metrics.get('dice', 0.0):.4f} ce={train_metrics.get('ce', 0.0):.4f} "
                f"focal={train_metrics.get('focal', 0.0):.4f}"
            )
            print(
                f"Epoch {epoch:03d}: {seg_msg} "
                f"grad={train_metrics.get('grad_norm', 0.0):.3f} time={duration:.1f}s lr={current_lr:.6f}",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar("train/lr", current_lr, epoch)
            update_metrics_history(history, "train", train_metrics, history_path, True)

        stop_after_epoch = bool(signal_state.get("stop_requested"))
        if stop_after_epoch and is_main:
            print(
                f"🛑 Stop requested; finishing after epoch {epoch:03d}",
                flush=True,
            )

        if (not stop_after_epoch) and (epoch % args.eval_interval == 0 or epoch == args.epochs):
            if ema_helper is not None:
                ema_helper.apply_shadow()
            val_metrics = validate_epoch(
                model,
                val_loader,
                device=device,
                num_classes=args.out_channels,
                foreground_only=args.foreground_only,
                use_sliding_window=args.use_sliding_window,
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
                sw_batch_size=args.sw_batch_size,
                sw_overlap=args.sw_overlap,
                multi_scale=args.multi_scale_eval,
                eval_scales=args.eval_scales,
                eval_tta=args.eval_tta,
                tta_axes=args.tta_flip_axes,
                debug_mode=args.debug_mode,
                debug_step_limit=args.debug_val_limit,
                is_main=is_main,
            )
            val_metrics["epoch"] = epoch
            if ema_helper is not None:
                ema_helper.restore()
            improved = val_metrics["dice"] > best_dice
            if improved:
                best_dice = val_metrics["dice"]
                best_model_epoch = epoch
            if is_main:
                extra_msgs = []
                extra_msgs.append(f"hd95={val_metrics.get('hd95', 0.0):.2f}")
                extra_msgs.append(f"assd={val_metrics.get('assd', 0.0):.2f}")
                extra_msgs.append(f"rve={val_metrics.get('rve', 0.0):.3f}")
                extra_msgs.append(f"cldice={val_metrics.get('cldice', 0.0):.3f}")
                extra_msgs.append(f"cbdice={val_metrics.get('cbdice', 0.0):.3f}")
                extra_msgs.append(f"clce={val_metrics.get('clce', 0.0):.3f}")
                adj_errors = val_metrics.get("adjacency_errors")
                if adj_errors:
                    extra_msgs.append(f"adj={adj_errors.get('mean_adj_error', 0.0):.3f}")
                    extra_msgs.append(f"spec={adj_errors.get('spectral_distance', 0.0):.3f}")
                struct = val_metrics.get("structural_violations")
                if struct:
                    extra_msgs.append(f"req_miss={struct.get('required_missing', 0.0):.2f}")
                    extra_msgs.append(f"forb={struct.get('forbidden_present', 0.0):.2f}")
                sym_scores = val_metrics.get("symmetry_scores")
                if sym_scores:
                    extra_msgs.append(f"sym={sym_scores[0]:.3f}")
                msg_extra = " ".join(extra_msgs)
                print(f"  Validation dice={val_metrics['dice']:.4f} {msg_extra}".strip())
                if writer is not None:
                    writer.add_scalar("val/dice", val_metrics["dice"], epoch)
                    writer.add_scalar("val/hd95", val_metrics.get("hd95", 0.0), epoch)
                    writer.add_scalar("val/assd", val_metrics.get("assd", 0.0), epoch)
                    writer.add_scalar("val/cldice", val_metrics.get("cldice", 0.0), epoch)
                    writer.add_scalar("val/cbdice", val_metrics.get("cbdice", 0.0), epoch)
                    writer.add_scalar("val/rve", val_metrics.get("rve", 0.0), epoch)
                    writer.add_scalar("val/clce", val_metrics.get("clce", 0.0), epoch)
                    if adj_errors:
                        writer.add_scalar("val/adjacency_mae", adj_errors.get("mean_adj_error", 0.0), epoch)
                        writer.add_scalar("val/spectral_gap", adj_errors.get("spectral_distance", 0.0), epoch)
                    if struct:
                        writer.add_scalar("val/required_missing", struct.get("required_missing", 0.0), epoch)
                        writer.add_scalar("val/forbidden_present", struct.get("forbidden_present", 0.0), epoch)
                    if sym_scores:
                        writer.add_scalar("val/symmetry_score", sym_scores[0], epoch)
                if improved:
                    best_path = best_checkpoint_path
                    save_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        best_dice=best_dice,
                        ema_state=ema_helper.state_dict() if ema_helper is not None else None,
                    )
                    record_resume_checkpoint(results_dir, best_path)
                    last_checkpoint_path = best_path
                    print(f"  ✅ New best checkpoint saved to {best_path}")
                update_metrics_history(history, "val", val_metrics, history_path, True)

        if is_main and args.save_interval > 0 and epoch % args.save_interval == 0:
            ckpt_path = results_dir / f"checkpoint_epoch{epoch:03d}.pt"
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_dice=best_dice,
                ema_state=ema_helper.state_dict() if ema_helper is not None else None,
            )
            checkpoint_history.append(ckpt_path)
            record_resume_checkpoint(results_dir, ckpt_path)
            last_checkpoint_path = ckpt_path
            while len(checkpoint_history) > args.max_keep_ckpt:
                old = checkpoint_history.popleft()
                if old.exists():
                    old.unlink()
                    print(f"  🧹 Removed old checkpoint {old}")

        if signal_state.get("triggered") and is_main:
            emergency_path = results_dir / f"checkpoint_signal_epoch{epoch:03d}.pt"
            save_checkpoint(
                emergency_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_dice=best_dice,
                ema_state=ema_helper.state_dict() if ema_helper is not None else None,
            )
            print(f"  💾 Signal-triggered checkpoint saved to {emergency_path}")
            checkpoint_history.append(emergency_path)
            record_resume_checkpoint(results_dir, emergency_path)
            last_checkpoint_path = emergency_path
            signal_state["triggered"] = False
            time_limit_exhausted = True

        if stop_after_epoch:
            time_limit_exhausted = True
            break

    if writer is not None:
        writer.close()

    if signal_state.get("stop_requested"):
        time_limit_exhausted = True

    if time_limit_exhausted and is_main:
        resume_epoch = last_completed_epoch if last_completed_epoch > 0 else max(0, start_epoch - 1)
        save_checkpoint(
            latest_model_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=resume_epoch,
            global_step=global_step,
            best_dice=best_dice,
            ema_state=ema_helper.state_dict() if ema_helper is not None else None,
        )
        record_resume_checkpoint(results_dir, latest_model_path)
        print(f"💾 Latest checkpoint saved to {latest_model_path}")
        print("⛔ Time buffer reached; exiting gracefully for resubmission")
    elif not time_limit_exhausted:
        final_epoch = last_completed_epoch if last_completed_epoch > 0 else args.epochs
        if is_main:
            save_checkpoint(
                final_model_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=final_epoch,
                global_step=global_step,
                best_dice=best_dice,
                ema_state=ema_helper.state_dict() if ema_helper is not None else None,
            )
            if latest_model_path.exists():
                latest_model_path.unlink()
            print(f"🏁 Final checkpoint saved to {final_model_path}")

        analysis_checkpoint = None
        analysis_epoch = final_epoch
        if best_checkpoint_path.exists():
            analysis_checkpoint = best_checkpoint_path
            if best_model_epoch is not None:
                analysis_epoch = best_model_epoch
        elif final_model_path.exists():
            analysis_checkpoint = final_model_path

        if analysis_checkpoint is not None and analysis_checkpoint.exists():
            if is_main:
                print(f"📊 Running final evaluation with {analysis_checkpoint}")
            load_model_weights_only(model, analysis_checkpoint)
            final_eval = validate_epoch(
                model,
                val_loader,
                device=device,
                num_classes=args.out_channels,
                foreground_only=args.foreground_only,
                use_sliding_window=args.use_sliding_window,
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
                sw_batch_size=args.sw_batch_size,
                sw_overlap=args.sw_overlap,
                multi_scale=args.multi_scale_eval,
                eval_scales=args.eval_scales,
                eval_tta=args.eval_tta,
                tta_axes=args.tta_flip_axes,
                debug_mode=False,
                debug_step_limit=1,
                is_main=is_main,
                return_per_class=True,
            )
            final_eval["epoch"] = analysis_epoch
            if is_main:
                per_class = final_eval.get("per_class_dice")
                locations = save_per_class_report(per_class, class_mapping, results_dir) if per_class else None
                msg = f"  ✅ Final analysis dice={final_eval['dice']:.4f}"
                if locations and "json" in locations:
                    msg += f" | per-class report: {locations['json']}"
                if locations and "png" in locations:
                    msg += f" | bar chart: {locations['png']}"
                print(msg)
        if is_main:
            plots_generated = generate_training_plots(history, results_dir)
            save_metrics_history(history, history_path)
            if plots_generated:
                print(f"📈 Training curves written to {results_dir / 'analysis'}")

    if distributed:
        dist.barrier()
    cleanup_distributed()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
