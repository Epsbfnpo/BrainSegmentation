#!/usr/bin/env python3
"""Training entry-point for the texture-focused adaptation pipeline."""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Optional

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

    # Checkpointing / resume
    parser.add_argument("--pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--auto_resume", action="store_true", help="Automatically resume from latest checkpoint")
    parser.add_argument("--max_keep_checkpoints", type=int, default=5, help="Number of epoch checkpoints to keep")

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
        torch.cuda.set_device(local_rank)
        timeout = timedelta(minutes=dist_timeout)
        dist.init_process_group(backend="nccl", timeout=timeout)

    return {"rank": rank, "world_size": world_size, "local_rank": local_rank, "distributed": distributed}


def _cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
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


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    dist_info = _setup_distributed(args.dist_timeout)
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    local_rank = dist_info["local_rank"]
    distributed = dist_info["distributed"]
    is_main = rank == 0

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

        def debug(msg: str) -> None:
            print(f"[rank {rank}] {msg}", flush=True)

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
            train_stats = train_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device,
                amp=args.amp and device.type == "cuda",
                scaler=scaler,
                domain_loss_weight=args.domain_loss_weight,
                embed_align_weight=args.embed_align_weight,
                stats_align_weight=args.stats_align_weight,
                grl_lambda=args.grl_lambda,
                is_distributed=distributed,
                world_size=world_size,
                use_tqdm=is_main,
                progress_desc=f"Train {epoch}/{args.epochs}",
            )
            debug("finished train_epoch")
            train_duration = time.time() - train_start
            if is_main:
                time_manager.record_train(train_duration)

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
                    val_loss, val_dice = validate(
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
                    )
                    debug("finished validation on main rank")
                val_duration = time.time() - val_start
                if is_main:
                    time_manager.record_val(val_duration)
                if distributed and world_size > 1:
                    debug("broadcasting validation metrics")
                    metrics_list = [val_loss, val_dice]
                    dist.broadcast_object_list(metrics_list, src=0)
                    val_loss, val_dice = metrics_list
                    # ensure all workers exit the broadcast phase before
                    # progressing to logging or stopping decisions
                    dist.barrier()
                    debug("post-validation barrier complete")

            scheduler.step()
            debug("scheduler stepped")

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
                        f"domain_acc={train_stats['domain_acc']:.3f}, val_loss={val_loss:.4f}, val_dice={val_dice:.4f}"
                    ),
                    is_main=is_main,
                )

                metrics_history.append(
                    {
                        "epoch": epoch,
                        **train_stats,
                        "val_loss": val_loss,
                        "val_dice": val_dice,
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch_minutes": epoch_duration / 60.0,
                        "train_minutes": train_duration / 60.0,
                        "val_minutes": val_duration / 60.0,
                        "time_remaining_minutes": time_manager.summary()["remaining_minutes"],
                    }
                )
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
                    dist.barrier()
                shared_reason = sync_stop(stop_reason)
                exit_reason = shared_reason or stop_reason
                break

            if distributed and world_size > 1:
                debug("epoch barrier before next loop")
                dist.barrier()

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
        _cleanup_distributed()


if __name__ == "__main__":
    main()
