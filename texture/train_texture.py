#!/usr/bin/env python3
"""Training entry-point for the texture-focused adaptation pipeline."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import signal
import time
from collections import Counter, deque
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional

import torch
import torch.distributed as dist
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader_texture import create_texture_dataloaders
from texture_model import TextureAwareModel, TextureBranchConfig
from trainer_texture import (
    build_dice_metric,
    build_loss,
    load_checkpoint,
    save_checkpoint,
    train_epoch,
    validate,
)

_DEF_RESULTS = "./results/texture_adapt"
_REASON_TO_CODE = {"time_limit": 1, "signal": 2}
_CODE_TO_REASON = {v: k for k, v in _REASON_TO_CODE.items()}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Texture-centric domain adaptation for brain segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument("--source_split_json", type=str, required=True, help="JSON split for source domain")
    parser.add_argument("--target_split_json", type=str, required=True, help="JSON split for target domain")
    parser.add_argument("--results_dir", type=str, default=_DEF_RESULTS, help="Directory to store outputs")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_rate", type=float, default=0.0)
    parser.add_argument("--cache_num_workers", type=int, default=4)
    parser.add_argument("--roi_x", type=int, default=96)
    parser.add_argument("--roi_y", type=int, default=96)
    parser.add_argument("--roi_z", type=int, default=96)
    parser.add_argument("--apply_spacing", action="store_true", default=False)
    parser.add_argument("--apply_orientation", action="store_true", default=False)
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[0.8, 0.8, 0.8])

    # Optimisation
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision")
    parser.set_defaults(amp=True)

    # Model configuration
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=87)
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--texture_embed_dim", type=int, default=128)
    parser.add_argument("--texture_stats_proj_dim", type=int, default=128)
    parser.add_argument("--texture_domain_hidden", type=int, default=128)
    parser.add_argument("--grl_lambda", type=float, default=1.0)
    parser.add_argument(
        "--foreground_only",
        dest="foreground_only",
        action="store_true",
        help="Use foreground-only labels (background mapped to -1, outputs cover 87 brain regions)",
    )
    parser.add_argument(
        "--include_background",
        dest="foreground_only",
        action="store_false",
        help="Retain background as an explicit prediction channel",
    )
    parser.set_defaults(foreground_only=True)

    # Loss weights
    parser.add_argument("--domain_loss_weight", type=float, default=0.5)
    parser.add_argument("--embed_align_weight", type=float, default=0.1)
    parser.add_argument("--stats_align_weight", type=float, default=0.1)
    parser.add_argument(
        "--domain_warmup_epochs",
        type=int,
        default=10,
        help="Number of epochs to ramp the adversarial losses from 0 to the configured weight",
    )
    parser.add_argument(
        "--align_warmup_epochs",
        type=int,
        default=10,
        help="Number of epochs to warm up the embedding/statistics alignment penalties",
    )

    # Checkpointing / resume
    parser.add_argument("--pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--auto_resume", action="store_true", help="Automatically resume from latest checkpoint")
    parser.add_argument("--max_keep_checkpoints", type=int, default=5, help="Number of epoch checkpoints to keep")

    # Debug / logging controls
    parser.add_argument(
        "--debug_interval",
        type=int,
        default=50,
        help="Steps between detailed training debug logs (0 disables per-step debug logging)",
    )
    parser.add_argument(
        "--debug_val_batches",
        type=int,
        default=2,
        help="Number of validation batches to log with detailed debug info",
    )
    parser.add_argument(
        "--debug_sample_count",
        type=int,
        default=4,
        help="Number of dataset samples to inspect prior to training for debugging",
    )

    # Distributed + scheduling
    parser.add_argument("--dist_timeout", type=int, default=120, help="Distributed init timeout in minutes")
    parser.add_argument("--job_time_limit", type=int, default=115, help="Job time limit in minutes (2h minus buffer)")
    parser.add_argument("--time_buffer_minutes", type=int, default=5, help="Minutes reserved as safety buffer")

    return parser


def _log_message(log_path: str, message: str, *, is_main: bool) -> None:
    if not is_main:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message, flush=True)


def _infer_texture_dim(loader, *, is_main: bool) -> int:
    if not is_main:
        return 0
    if hasattr(loader, "dataset") and len(loader.dataset) > 0:
        sample = loader.dataset[0]
    else:
        sample = next(iter(loader))
    stats = sample["texture_stats"]
    if isinstance(stats, torch.Tensor):
        dim = stats.numel()
    else:
        dim = len(stats)
    return int(dim)


def _find_latest_checkpoint(results_dir: str) -> Optional[str]:
    candidates = []
    latest_path = os.path.join(results_dir, "checkpoint_latest.pth")
    if os.path.isfile(latest_path):
        candidates.append(latest_path)
    pattern = os.path.join(results_dir, "checkpoint_epoch_*.pth")
    candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    candidates = sorted(candidates, key=os.path.getmtime, reverse=True)
    return candidates[0]


def _prune_old_checkpoints(results_dir: str, keep: int, *, is_main: bool) -> None:
    if not is_main or keep <= 0:
        return
    pattern = os.path.join(results_dir, "checkpoint_epoch_*.pth")
    checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for path in checkpoints[keep:]:
        try:
            os.remove(path)
        except OSError:
            continue


def _setup_distributed(dist_timeout: int) -> Dict[str, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    distributed = world_size > 1

    if distributed:
        # Ensure NCCL asynchronous errors surface as Python exceptions rather than silent hangs.
        if "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ:
            legacy_setting = os.environ.get("NCCL_ASYNC_ERROR_HANDLING")
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = legacy_setting or "1"
        torch.cuda.set_device(local_rank)
        timeout = timedelta(minutes=dist_timeout)
        dist.init_process_group(backend="nccl", timeout=timeout)

    return {"rank": rank, "world_size": world_size, "local_rank": local_rank, "distributed": distributed}


def _safe_barrier(stage: str, *, debug_fn: Optional[Callable[[str], None]] = None) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if debug_fn:
        debug_fn(f"enter barrier: {stage}")
    try:
        try:
            work = dist.barrier(async_op=True)
        except TypeError:
            dist.barrier()
        else:
            work.wait()
    except Exception as exc:
        if debug_fn:
            debug_fn(f"barrier failure at {stage}: {exc}")
        raise
    else:
        if debug_fn:
            debug_fn(f"exit barrier: {stage}")


def _warmup_scale(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    progress = max(0, epoch - 1)
    return float(min(1.0, progress / float(warmup_epochs)))


def _cleanup_distributed(debug_fn: Optional[Callable[[str], None]] = None):
    if dist.is_available() and dist.is_initialized():
        _safe_barrier("cleanup", debug_fn=debug_fn)
        dist.destroy_process_group()


class TimeManager:
    """Utility to keep training within the 2-hour job window."""

    def __init__(self, job_time_limit_minutes: int, buffer_minutes: int = 5):
        self.job_start_time = time.time()
        self.job_time_limit = job_time_limit_minutes * 60
        self.buffer_time = buffer_minutes * 60
        self.train_times = deque(maxlen=8)
        self.val_times = deque(maxlen=4)
        self.epoch_times = deque(maxlen=6)

    def _estimate(self, values: deque, default: float) -> float:
        if not values:
            return default
        return sum(values) / len(values) * 1.2  # add 20% safety margin

    def record_train(self, duration: float) -> None:
        self.train_times.append(duration)

    def record_val(self, duration: float) -> None:
        self.val_times.append(duration)

    def record_epoch(self, duration: float) -> None:
        self.epoch_times.append(duration)

    def remaining(self) -> float:
        elapsed = time.time() - self.job_start_time
        return self.job_time_limit - self.buffer_time - elapsed

    def should_stop(self) -> bool:
        return self.remaining() <= 0

    def can_start_epoch(self, will_validate: bool) -> bool:
        if self.should_stop():
            return False
        train_est = self._estimate(self.train_times, default=600.0)
        val_est = self._estimate(self.val_times, default=360.0) if will_validate else 0.0
        epoch_est = self._estimate(self.epoch_times, default=train_est + val_est)
        required = max(train_est + val_est, epoch_est)
        return self.remaining() > required

    def can_run_validation(self) -> bool:
        if self.should_stop():
            return False
        required = self._estimate(self.val_times, default=360.0)
        return self.remaining() > required

    def summary(self) -> Dict[str, float]:
        return {
            "remaining_minutes": max(0.0, self.remaining() / 60.0),
            "avg_train_minutes": self._estimate(self.train_times, 0.0) / 60.0 if self.train_times else 0.0,
            "avg_val_minutes": self._estimate(self.val_times, 0.0) / 60.0 if self.val_times else 0.0,
            "avg_epoch_minutes": self._estimate(self.epoch_times, 0.0) / 60.0 if self.epoch_times else 0.0,
        }


def _as_cpu_tensor(data) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    if hasattr(data, "as_tensor"):
        return data.as_tensor().detach().cpu()
    return torch.as_tensor(data)


def _summarize_module(module: torch.nn.Module) -> Dict[str, float]:
    total = 0
    sum_sq = 0.0
    max_abs = 0.0
    with torch.no_grad():
        for param in module.parameters():
            data = param.detach().float()
            total += data.numel()
            sum_sq += float(data.pow(2).sum().item())
            max_abs = max(max_abs, float(data.abs().max().item()))
    rms = math.sqrt(sum_sq / max(1, total)) if total > 0 else 0.0
    return {"total_params": total, "rms": rms, "max_abs": max_abs}


def _topk_histogram(hist, k: int = 5):
    if not hist:
        return []
    values = [(idx, float(val)) for idx, val in enumerate(hist)]
    values = [item for item in values if item[1] > 0]
    if not values:
        return []
    values.sort(key=lambda item: item[1], reverse=True)
    total = sum(val for _, val in values)
    top = []
    for idx, val in values[:k]:
        frac = val / total if total > 0 else 0.0
        top.append((idx, val, frac))
    return top


def _inspect_samples(dataset, count: int, *, prefix: str, debug: Callable[[str], None]) -> None:
    if dataset is None or count <= 0:
        return
    length = len(dataset) if hasattr(dataset, "__len__") else None
    debug(f"{prefix} dataset length={length if length is not None else 'unknown'}")
    domain_counter: Counter = Counter()
    fg_fracs = []
    unique_labels: set[int] = set()
    inspected = 0
    for idx in range(min(count, length or count)):
        try:
            sample = dataset[idx]
        except Exception as exc:
            debug(f"{prefix} sample[{idx}] failed to load for debug inspection: {exc}")
            continue
        inspected += 1
        label_tensor = _as_cpu_tensor(sample.get("label"))
        sample_unique = torch.unique(label_tensor).to(dtype=torch.long)
        unique_labels.update(sample_unique.tolist())
        fg_fraction = float((label_tensor >= 0).float().mean().item())
        fg_fracs.append(fg_fraction)
        domain_value = sample.get("domain")
        if domain_value is not None:
            try:
                domain_int = int(_as_cpu_tensor(domain_value).item())
            except Exception:
                domain_int = int(domain_value)
            domain_counter[domain_int] += 1
        texture_stats = sample.get("texture_stats")
        if texture_stats is not None:
            stats_tensor = _as_cpu_tensor(texture_stats).float()
            stats_mean = float(stats_tensor.mean().item())
            stats_std = float(stats_tensor.std(unbiased=False).item())
            stats_min = float(stats_tensor.min().item())
            stats_max = float(stats_tensor.max().item())
            stats_summary = (
                f"mean={stats_mean:.4f}, std={stats_std:.4f}, min={stats_min:.4f}, max={stats_max:.4f}"
            )
        else:
            stats_summary = "missing"
        debug(
            (
                f"{prefix}[{idx}]: domain={domain_value}, fg_frac={fg_fraction:.3f}, "
                f"unique_labels={sorted(sample_unique.tolist())[:12]}"
                f"{'...' if sample_unique.numel() > 12 else ''}, texture_stats={stats_summary}"
            )
        )
    if fg_fracs:
        fg_avg = float(sum(fg_fracs) / len(fg_fracs))
        fg_min = float(min(fg_fracs))
        fg_max = float(max(fg_fracs))
        debug(
            (
                f"{prefix} foreground fraction stats -> mean={fg_avg:.3f}, min={fg_min:.3f}, "
                f"max={fg_max:.3f}"
            )
        )
    if domain_counter:
        debug(f"{prefix} domain counts (first {inspected} samples) -> {dict(domain_counter)}")
    if unique_labels:
        sorted_labels = sorted(unique_labels)
        preview = sorted_labels[:20]
        suffix = "..." if len(sorted_labels) > 20 else ""
        debug(f"{prefix} union of labels across inspected samples: {preview}{suffix}")


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    if args.align_warmup_epochs < 0:
        args.align_warmup_epochs = args.domain_warmup_epochs

    dist_info = _setup_distributed(args.dist_timeout)
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    local_rank = dist_info["local_rank"]
    distributed = dist_info["distributed"]
    is_main = rank == 0

    debug: Callable[[str], None] = lambda msg: None

    try:
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        set_determinism(seed=args.seed)

        os.makedirs(args.results_dir, exist_ok=True)
        log_path = os.path.join(args.results_dir, "training.log")
        history_path = os.path.join(args.results_dir, "metrics_history.json")

        _log_message(log_path, "Starting texture adaptation run", is_main=is_main)
        _log_message(log_path, json.dumps(vars(args), indent=2), is_main=is_main)

        roi_size = (args.roi_x, args.roi_y, args.roi_z)

        rank_trace_path = os.path.join(args.results_dir, f"rank_{rank:02d}_trace.log")

        def debug(msg: str) -> None:
            line = f"[rank {rank}] {msg}"
            print(line, flush=True)
            try:
                os.makedirs(os.path.dirname(rank_trace_path), exist_ok=True)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(rank_trace_path, "a", encoding="utf-8") as trace_file:
                    trace_file.write(f"{timestamp} {line}\n")
            except OSError:
                # Avoid crashing training due to logging issues on shared filesystems.
                pass

        debug("starting dataloader creation")

        train_loader, val_loader, train_sampler, _ = create_texture_dataloaders(
            args.source_split_json,
            args.target_split_json,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            cache_rate=args.cache_rate,
            cache_num_workers=args.cache_num_workers,
            roi_size=roi_size,
            apply_spacing=args.apply_spacing,
            target_spacing=args.target_spacing,
            apply_orientation=args.apply_orientation,
            foreground_only=args.foreground_only,
            num_classes=args.out_channels,
            distributed=distributed,
            distribute_val=False,
            world_size=world_size,
            rank=rank,
            seed=args.seed,
        )

        debug("dataloaders ready")
        try:
            train_steps = len(train_loader)
        except TypeError:
            train_steps = -1
        try:
            val_steps = len(val_loader)
        except TypeError:
            val_steps = -1
        debug(
            "dataset stats -> train_samples=%d, val_samples=%d, train_steps_per_epoch=%d, val_steps=%d"
            % (
                len(train_loader.dataset) if hasattr(train_loader, "dataset") else -1,
                len(val_loader.dataset) if hasattr(val_loader, "dataset") else -1,
                train_steps,
                val_steps,
            )
        )

        if is_main:
            _inspect_samples(
                getattr(train_loader, "dataset", None),
                args.debug_sample_count,
                prefix="train",
                debug=debug,
            )
            _inspect_samples(
                getattr(val_loader, "dataset", None),
                max(0, min(args.debug_sample_count, 2)),
                prefix="val",
                debug=debug,
            )

        texture_stats_dim = _infer_texture_dim(train_loader, is_main=is_main)
        if distributed and world_size > 1:
            tensor = torch.tensor([texture_stats_dim], device=device, dtype=torch.int32)
            dist.broadcast(tensor, src=0)
            texture_stats_dim = int(tensor[0].item())
        _log_message(log_path, f"Inferred texture feature dimension: {texture_stats_dim}", is_main=is_main)
        debug("texture feature dimension inferred")

        branch_cfg = TextureBranchConfig(
            embed_dim=args.texture_embed_dim,
            stats_projection_dim=args.texture_stats_proj_dim,
            domain_hidden=args.texture_domain_hidden,
            grl_lambda=args.grl_lambda,
        )

        model = TextureAwareModel(
            img_size=roi_size,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            dropout=args.dropout,
            texture_stats_dim=texture_stats_dim,
            branch_cfg=branch_cfg,
        )
        model = model.to(device)
        debug("model constructed and moved to device")

        seg_summary = _summarize_module(model.segmenter)
        debug(
            "segmenter parameter summary before loading weights -> params=%d, rms=%.4e, max=%.4e"
            % (seg_summary["total_params"], seg_summary["rms"], seg_summary["max_abs"])
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

        debug("optimizer and scheduler ready")

        if distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        include_background = not args.foreground_only
        if args.foreground_only and args.out_channels != 87 and is_main:
            _log_message(
                log_path,
                (
                    "foreground_only mode expects 87 foreground classes; "
                    f"received out_channels={args.out_channels}"
                ),
                is_main=is_main,
            )

        loss_fn = build_loss(
            num_classes=args.out_channels,
            include_background=include_background,
            foreground_only=args.foreground_only,
        )
        dice_metric = build_dice_metric(
            num_classes=args.out_channels,
            include_background=include_background,
            foreground_only=args.foreground_only,
        )

        start_epoch = 1
        best_metric: Optional[float] = None

        if args.pretrained_checkpoint and os.path.isfile(args.pretrained_checkpoint):
            ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            if isinstance(model, DDP):
                model.module.segmenter.load_state_dict(state_dict, strict=False)
            else:
                model.segmenter.load_state_dict(state_dict, strict=False)
            _log_message(log_path, f"Loaded pretrained weights from {args.pretrained_checkpoint}", is_main=is_main)
            if isinstance(model, DDP):
                seg_summary = _summarize_module(model.module.segmenter)
            else:
                seg_summary = _summarize_module(model.segmenter)
            debug(
                "segmenter parameter summary after loading weights -> params=%d, rms=%.4e, max=%.4e"
                % (seg_summary["total_params"], seg_summary["rms"], seg_summary["max_abs"])
            )
        elif args.pretrained_checkpoint:
            _log_message(log_path, f"⚠️ Pretrained checkpoint not found at {args.pretrained_checkpoint}", is_main=is_main)

        resume_path = None
        if args.resume_checkpoint:
            resume_path = args.resume_checkpoint
        elif args.auto_resume:
            resume_path = _find_latest_checkpoint(args.results_dir)

        if resume_path and os.path.isfile(resume_path):
            _log_message(log_path, f"Resuming from {resume_path}", is_main=is_main)
            checkpoint = load_checkpoint(model, optimizer, resume_path, map_location=device)
            resumed_epoch = checkpoint.get("epoch", 0)
            best_metric = checkpoint.get("best_metric", best_metric)
            start_epoch = resumed_epoch + 1
            if resumed_epoch > 0:
                for _ in range(resumed_epoch):
                    scheduler.step()
        elif resume_path:
            _log_message(log_path, f"⚠️ Resume checkpoint not found at {resume_path}", is_main=is_main)

        if distributed and world_size > 1:
            best_tensor = torch.tensor([-1.0 if best_metric is None else best_metric], device=device)
            dist.broadcast(best_tensor, src=0)
            best_value = float(best_tensor[0].item())
            if best_value >= 0:
                best_metric = best_value

        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

        metrics_history = []
        if is_main and os.path.isfile(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    metrics_history = json.load(f)
            except Exception:
                metrics_history = []
            else:
                metrics_history = [m for m in metrics_history if m.get("epoch", 0) < start_epoch]

        time_manager = TimeManager(args.job_time_limit, buffer_minutes=args.time_buffer_minutes)
        current_epoch = start_epoch - 1
        stop_reason: Optional[str] = None

        def save_latest(reason: str) -> None:
            if not is_main:
                return
            latest_path = os.path.join(args.results_dir, "checkpoint_latest.pth")
            save_checkpoint(model, optimizer, max(1, current_epoch), latest_path, best_metric=best_metric)
            _log_message(log_path, f"💾 Saved latest checkpoint due to {reason}", is_main=is_main)

        def global_decision(value: Optional[bool]) -> bool:
            """Broadcast a boolean decision from rank 0 to all workers."""

            if distributed and world_size > 1:
                tensor = torch.zeros(1, device=device, dtype=torch.int32)
                if is_main:
                    tensor[0] = 1 if value else 0
                dist.broadcast(tensor, src=0)
                return bool(tensor.item())
            assert value is not None
            return bool(value)

        def sync_stop(reason: Optional[str]) -> Optional[str]:
            nonlocal stop_reason
            if reason and not stop_reason:
                stop_reason = reason
            if distributed and world_size > 1:
                code = _REASON_TO_CODE.get(stop_reason or reason, 0)
                code_tensor = torch.tensor([code], device=device, dtype=torch.int32)
                dist.all_reduce(code_tensor, op=dist.ReduceOp.MAX)
                final_code = int(code_tensor.item())
                if final_code > 0:
                    stop_reason = _CODE_TO_REASON.get(final_code, stop_reason or "time_limit")
                    return stop_reason
                return None
            return stop_reason if stop_reason else None

        def handle_signal(signum, frame):  # type: ignore[override]
            nonlocal stop_reason
            stop_reason = stop_reason or "signal"
            save_latest("signal")

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        _log_message(log_path, "Commencing training loop", is_main=is_main)
        debug("entered training loop")

        base_model_ref = model.module if hasattr(model, "module") else model
        prev_seg_summary = _summarize_module(getattr(base_model_ref, "segmenter", base_model_ref))
        prev_domain_summary = _summarize_module(getattr(base_model_ref, "domain_classifier", base_model_ref))
        prev_texture_summary = _summarize_module(getattr(base_model_ref, "texture_projection", base_model_ref))
        prev_stats_summary = _summarize_module(getattr(base_model_ref, "stats_projector", base_model_ref))
        prev_encoder_summary = _summarize_module(getattr(base_model_ref, "texture_encoder", base_model_ref))

        exit_reason = "completed"
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            shared_reason = sync_stop(stop_reason)
            if shared_reason:
                exit_reason = shared_reason
                break

            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            will_validate = True
            debug("before can_start_epoch decision")
            can_start = global_decision(time_manager.can_start_epoch(will_validate) if is_main else None)
            debug(f"after can_start_epoch decision -> {can_start}")
            if not can_start:
                exit_reason = "time_limit"
                stop_reason = stop_reason or "time_limit"
                save_latest("time limit before epoch")
                shared_reason = sync_stop(stop_reason)
                exit_reason = shared_reason or exit_reason
                break

            epoch_start = time.time()
            train_start = time.time()
            debug("starting train_epoch")
            domain_scale = _warmup_scale(epoch, args.domain_warmup_epochs)
            align_scale = _warmup_scale(epoch, args.align_warmup_epochs)
            current_domain_weight = args.domain_loss_weight * domain_scale
            current_embed_weight = args.embed_align_weight * align_scale
            current_stats_weight = args.stats_align_weight * align_scale
            current_grl_lambda = args.grl_lambda * domain_scale
            train_stats = train_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device,
                amp=args.amp and device.type == "cuda",
                scaler=scaler,
                domain_loss_weight=current_domain_weight,
                embed_align_weight=current_embed_weight,
                stats_align_weight=current_stats_weight,
                grl_lambda=current_grl_lambda,
                is_distributed=distributed,
                world_size=world_size,
                use_tqdm=is_main,
                progress_desc=f"Train {epoch}/{args.epochs}",
                debug_interval=args.debug_interval if is_main else 0,
                debug_fn=debug if is_main else None,
                num_classes=args.out_channels,
                collect_stats=is_main,
            )
            debug("finished train_epoch")
            train_duration = time.time() - train_start
            if is_main:
                time_manager.record_train(train_duration)
                debug(
                    "epoch %d train summary -> total=%.4f, seg=%.4f, domain=%.4f, align=%.4f, stats=%.4f, dom_acc=%.3f, batches=%d"
                    % (
                        epoch,
                        train_stats["total_loss"],
                        train_stats["seg_loss"],
                        train_stats["domain_loss"],
                        train_stats["align_loss"],
                        train_stats["stats_align_loss"],
                        train_stats["domain_acc"],
                        train_stats["num_batches"],
                    )
                )

                if train_stats.get("label_hist") and train_stats.get("pred_hist"):
                    label_top = _topk_histogram(train_stats["label_hist"])
                    pred_top = _topk_histogram(train_stats["pred_hist"])
                    fg_fraction = train_stats.get("fg_fraction", 0.0)
                    debug(
                        (
                            f"epoch {epoch} train label dist top -> "
                            f"{[(idx, int(val), frac) for idx, val, frac in label_top]}"
                            f", pred dist top -> {[(idx, int(val), frac) for idx, val, frac in pred_top]}"
                            f", fg_fraction={fg_fraction:.3f}"
                        )
                    )

                base_model_ref = model.module if hasattr(model, "module") else model
                seg_summary = _summarize_module(getattr(base_model_ref, "segmenter", base_model_ref))
                domain_summary = _summarize_module(getattr(base_model_ref, "domain_classifier", base_model_ref))
                texture_summary = _summarize_module(getattr(base_model_ref, "texture_projection", base_model_ref))
                stats_summary = _summarize_module(getattr(base_model_ref, "stats_projector", base_model_ref))
                encoder_summary = _summarize_module(getattr(base_model_ref, "texture_encoder", base_model_ref))

                debug(
                    (
                        f"epoch {epoch} parameter stats -> seg_rms={seg_summary['rms']:.4e} (Δ{seg_summary['rms'] - prev_seg_summary['rms']:.2e}), "
                        f"domain_rms={domain_summary['rms']:.4e} (Δ{domain_summary['rms'] - prev_domain_summary['rms']:.2e}), "
                        f"texproj_rms={texture_summary['rms']:.4e} (Δ{texture_summary['rms'] - prev_texture_summary['rms']:.2e}), "
                        f"statsproj_rms={stats_summary['rms']:.4e} (Δ{stats_summary['rms'] - prev_stats_summary['rms']:.2e}), "
                        f"encoder_rms={encoder_summary['rms']:.4e} (Δ{encoder_summary['rms'] - prev_encoder_summary['rms']:.2e})"
                    )
                )

                prev_seg_summary = seg_summary
                prev_domain_summary = domain_summary
                prev_texture_summary = texture_summary
                prev_stats_summary = stats_summary
                prev_encoder_summary = encoder_summary

            debug("checking should_stop")
            should_stop = global_decision(time_manager.should_stop() if is_main else None)
            debug(f"should_stop -> {should_stop}")
            if should_stop:
                exit_reason = "time_limit"
                stop_reason = stop_reason or "time_limit"
                save_latest("time limit after train")
                shared_reason = sync_stop(stop_reason)
                exit_reason = shared_reason or exit_reason
                break

            val_loss = 0.0
            val_dice = 0.0
            val_duration = 0.0
            val_extra: Dict[str, object] = {}
            if will_validate:
                debug("before can_run_validation decision")
                can_validate = global_decision(time_manager.can_run_validation() if is_main else None)
                debug(f"after can_run_validation -> {can_validate}")
                if not can_validate:
                    exit_reason = "time_limit"
                    stop_reason = stop_reason or "time_limit"
                    save_latest("time limit before val")
                    shared_reason = sync_stop(stop_reason)
                    exit_reason = shared_reason or exit_reason
                    break
                val_start = time.time()
                if is_main:
                    debug("starting validation on main rank")
                    val_loss, val_dice, val_extra = validate(
                        model,
                        val_loader,
                        loss_fn,
                        dice_metric,
                        device,
                        roi_size=roi_size,
                        sw_batch_size=max(1, args.val_batch_size),
                        amp=args.amp and device.type == "cuda",
                        num_classes=args.out_channels,
                        is_distributed=False,
                        world_size=1,
                        foreground_only=args.foreground_only,
                        use_tqdm=True,
                        progress_desc=f"Val {epoch}/{args.epochs}",
                        debug_batches=args.debug_val_batches,
                        debug_fn=debug,
                        collect_stats=True,
                    )
                    debug("finished validation on main rank")
                val_duration = time.time() - val_start
                if is_main:
                    time_manager.record_val(val_duration)
                    debug(
                        "epoch %d val summary -> loss=%.4f, dice=%.4f, duration=%.2fs"
                        % (epoch, val_loss, val_dice, val_duration)
                    )
                    if val_extra:
                        label_top = _topk_histogram(val_extra.get("label_hist", []))
                        pred_top = _topk_histogram(val_extra.get("pred_hist", []))
                        dice_scores = val_extra.get("per_class_dice", [])
                        top_dice = []
                        if dice_scores:
                            pairs = [(idx, float(score)) for idx, score in enumerate(dice_scores)]
                            pairs.sort(key=lambda item: item[1], reverse=True)
                            top_dice = pairs[:5]
                        debug(
                            (
                                f"epoch {epoch} val label dist top -> "
                                f"{[(idx, int(val), frac) for idx, val, frac in label_top]}"
                                f", pred dist top -> {[(idx, int(val), frac) for idx, val, frac in pred_top]}"
                                f", dice top -> {[(idx, round(score, 4)) for idx, score in top_dice]}"
                            )
                        )
                if distributed and world_size > 1:
                    debug("broadcasting validation metrics")
                    metrics_tensor = torch.zeros(2, device=device, dtype=torch.float32)
                    if is_main:
                        metrics_tensor[0] = float(val_loss)
                        metrics_tensor[1] = float(val_dice)
                    dist.broadcast(metrics_tensor, src=0)
                    val_loss = float(metrics_tensor[0].item())
                    val_dice = float(metrics_tensor[1].item())

            scheduler.step()
            debug("scheduler stepped")
            if is_main:
                current_lr = optimizer.param_groups[0]["lr"]
                remaining = time_manager.summary()
                debug(
                    "epoch %d lr=%.6e, remaining=%.2f min (avg train=%.2f, avg val=%.2f)"
                    % (
                        epoch,
                        current_lr,
                        remaining["remaining_minutes"],
                        remaining["avg_train_minutes"],
                        remaining["avg_val_minutes"],
                    )
                )

            epoch_duration = time.time() - epoch_start
            if is_main:
                time_manager.record_epoch(epoch_duration)

            if is_main:
                _log_message(
                    log_path,
                    (
                        f"Epoch {epoch}/{args.epochs}: total_loss={train_stats['total_loss']:.4f}, "
                        f"seg_loss={train_stats['seg_loss']:.4f}, domain_loss={train_stats['domain_loss']:.4f}, "
                        f"align_loss={train_stats['align_loss']:.4f}, stats_align_loss={train_stats['stats_align_loss']:.4f}, "
                        f"domain_acc={train_stats['domain_acc']:.3f}, val_loss={val_loss:.4f}, val_dice={val_dice:.4f}, "
                        f"domain_w={current_domain_weight:.3f}, align_w={current_embed_weight:.3f}, stats_w={current_stats_weight:.3f}"
                    ),
                    is_main=is_main,
                )

                metrics_entry: Dict[str, object] = {
                    "epoch": epoch,
                    **train_stats,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "lr": optimizer.param_groups[0]["lr"],
                    "domain_weight": current_domain_weight,
                    "embed_align_weight": current_embed_weight,
                    "stats_align_weight": current_stats_weight,
                    "epoch_minutes": epoch_duration / 60.0,
                    "train_minutes": train_duration / 60.0,
                    "val_minutes": val_duration / 60.0,
                    "time_remaining_minutes": time_manager.summary()["remaining_minutes"],
                }
                if val_extra:
                    for key, value in val_extra.items():
                        metrics_entry[f"val_{key}"] = value
                metrics_history.append(metrics_entry)
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(metrics_history, f, indent=2)

                if best_metric is None or val_dice > best_metric:
                    best_metric = val_dice
                    best_path = os.path.join(args.results_dir, "best_model.pth")
                    save_checkpoint(model, optimizer, epoch, best_path, best_metric=best_metric)
                    _log_message(log_path, f"✅ New best model saved (dice={best_metric:.4f})", is_main=is_main)
                debug("metrics logged and best model check complete")

                if epoch % max(1, args.save_every) == 0:
                    ckpt_path = os.path.join(args.results_dir, f"checkpoint_epoch_{epoch:03d}.pth")
                    save_checkpoint(model, optimizer, epoch, ckpt_path, best_metric=best_metric)
                    _prune_old_checkpoints(args.results_dir, args.max_keep_checkpoints, is_main=is_main)

            if stop_reason:
                if distributed and world_size > 1:
                    debug("stop_reason set before final barrier")
                    _safe_barrier("pre-exit", debug_fn=debug)
                shared_reason = sync_stop(stop_reason)
                exit_reason = shared_reason or stop_reason
                break

            if distributed and world_size > 1:
                debug("epoch barrier before next loop")
                _safe_barrier(f"epoch-{epoch}-end", debug_fn=debug)

            shared_reason = sync_stop(stop_reason)
            if shared_reason:
                exit_reason = shared_reason
                break

            debug("epoch loop completed")

        if exit_reason != "time_limit" and exit_reason != "signal":
            if is_main:
                final_path = os.path.join(args.results_dir, "last_model.pth")
                save_checkpoint(model, optimizer, max(start_epoch, current_epoch), final_path, best_metric=best_metric)
                _log_message(log_path, f"Training complete. Final model saved to {final_path}", is_main=is_main)
        else:
            _log_message(log_path, f"Training stopped early due to {exit_reason}", is_main=is_main)

    finally:
        _cleanup_distributed(debug)


if __name__ == "__main__":
    main()
