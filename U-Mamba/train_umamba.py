#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from data_loader import get_loader
from model import UMamba3D
from trainer import CombinedLoss, ExponentialMovingAverage, train_epoch, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Target-only U-Mamba training")
    parser.add_argument("--split_json", required=True, help="Path to training/validation split JSON")
    parser.add_argument("--results_dir", default="./results/umamba", help="Directory to store checkpoints")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--roi_x", type=int, default=128)
    parser.add_argument("--roi_y", type=int, default=128)
    parser.add_argument("--roi_z", type=int, default=128)
    parser.add_argument("--out_channels", type=int, default=87)
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[0.8, 0.8, 0.8])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_rate", type=float, default=0.2)
    parser.add_argument("--cache_num_workers", type=int, default=4)
    parser.add_argument("--slurm_time_buffer", type=float, default=300.0)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def parse_slurm_timelimit(value: str) -> float | None:
    try:
        parts = value.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return float(hours * 3600 + minutes * 60 + seconds)
        if len(parts) == 2:
            hours, minutes = map(int, parts)
            return float(hours * 3600 + minutes * 60)
        if len(parts) == 1:
            minutes = int(parts[0])
            return float(minutes * 60)
    except ValueError:
        return None
    return None


def compute_job_deadline(buffer_seconds: float) -> float | None:
    end_env = os.environ.get("SLURM_JOB_END_TIME")
    if end_env:
        try:
            return float(end_env) - buffer_seconds
        except ValueError:
            pass
    start_env = os.environ.get("SLURM_JOB_START_TIME")
    limit_env = os.environ.get("SLURM_JOB_TIME_LIMIT") or os.environ.get("SLURM_TIMELIMIT")
    if start_env and limit_env:
        try:
            start_time = float(start_env)
        except ValueError:
            start_time = None
        limit_seconds = parse_slurm_timelimit(limit_env)
        if start_time is not None and limit_seconds is not None:
            return start_time + limit_seconds - buffer_seconds
    return None


def register_signal_handlers(flag: dict, *, is_main: bool) -> None:
    flag.setdefault("triggered", False)
    flag.setdefault("stop_requested", False)
    flag.setdefault("signal", None)

    def _handler(signum, frame):  # pragma: no cover - signal handling
        flag["triggered"] = True
        flag["signal"] = signum
        flag["stop_requested"] = True
        if is_main:
            print(f"⚠️  Received signal {signum}; will stop after current epoch.")

    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, _handler)


def init_distributed() -> tuple[bool, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        return True, rank, world_size
    return False, 0, 1


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer, epoch: int, best_dice: float) -> None:
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_dice": best_dice,
    }
    torch.save(state, path)


def main() -> None:
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    is_distributed, rank, world_size = init_distributed()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}") if torch.cuda.is_available() else torch.device("cpu")
    is_main = (rank == 0)

    model = UMamba3D(in_channels=1, out_channels=args.out_channels).to(device)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = CombinedLoss(args.out_channels).to(device)
    ema = ExponentialMovingAverage(model)

    train_loader, val_loader = get_loader(
        args,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )

    start_epoch = 1
    best_dice = 0.0
    resume_path = Path(args.resume) if args.resume else Path(args.results_dir) / "latest_model.pt"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_dice", 0.0)
        target_state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(target_state)
        optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        if is_main:
            print(f"🔄 Resumed from epoch {start_epoch-1}, best dice {best_dice:.4f}")

    deadline = compute_job_deadline(args.slurm_time_buffer)
    signal_state = {"triggered": False, "stop_requested": False, "signal": None}
    register_signal_handlers(signal_state, is_main=is_main)

    for epoch in range(start_epoch, args.epochs + 1):
        if is_distributed and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)

        loss = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler, ema)
        scheduler.step()

        if is_main:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:04d} | Loss {loss:.4f} | LR {lr:.6f}")

        if is_main and (epoch % args.eval_interval == 0):
            ema.apply_shadow()
            dice = validate(model, val_loader, device, args)
            ema.restore()
            print(f"  >> Val Dice: {dice:.4f}")
            if dice > best_dice:
                best_dice = dice
                save_checkpoint(Path(args.results_dir) / "best_model.pt", model, optimizer, epoch, best_dice)
                print("  🔥 New best model saved")

        if is_main:
            save_checkpoint(Path(args.results_dir) / "latest_model.pt", model, optimizer, epoch, best_dice)

        if signal_state.get("stop_requested"):
            if is_main:
                print("🛑 Stop requested by external signal; exiting after checkpoint.")
            break

        if deadline and time.time() > deadline:
            if is_main:
                print("⏳ Approaching SLURM time limit; saving checkpoint and exiting.")
            break

    if is_main:
        save_checkpoint(Path(args.results_dir) / "final_model.pt", model, optimizer, epoch, best_dice)


if __name__ == "__main__":
    main()
