from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from arch.mednext import MedNeXt
from data_loader import get_loader
from utils import CombinedLoss, ExponentialMovingAverage, train_one_epoch, validate


def parse_args():
    parser = argparse.ArgumentParser(description="MedNeXt baseline training on PPREMO/PREBO")
    parser.add_argument("--split_json", required=True, help="Path to split JSON with training/validation entries")
    parser.add_argument("--results_dir", default="./results_mednext", help="Directory to store checkpoints")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=87)
    parser.add_argument("--roi_x", type=int, default=128)
    parser.add_argument("--roi_y", type=int, default=128)
    parser.add_argument("--roi_z", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_rate", type=float, default=0.2)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--slurm_time_buffer", type=float, default=300.0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main = dist.get_rank() == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    model = MedNeXt(
        in_channels=args.in_channels,
        n_channels=32,
        n_classes=args.out_channels,
        exp_r=2,
        drop_path_rate=0.05,
        deep_supervision=True,
        dim="3d",
        grn=True,
    ).to(device)

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = CombinedLoss(args.out_channels).to(device)
    ema = ExponentialMovingAverage(model, decay=0.999)

    train_loader, val_loader = get_loader(
        args,
        is_distributed=dist.is_initialized(),
        rank=dist.get_rank() if dist.is_initialized() else 0,
        world_size=dist.get_world_size() if dist.is_initialized() else 1,
    )

    best_dice = 0.0
    job_deadline = None
    if "SLURM_JOB_END_TIME" in os.environ:
        job_deadline = float(os.environ["SLURM_JOB_END_TIME"]) - args.slurm_time_buffer

    for epoch in range(1, args.epochs + 1):
        if job_deadline and time.time() > job_deadline:
            if is_main:
                torch.save({"epoch": epoch, "state_dict": model.state_dict()}, os.path.join(args.results_dir, "latest_model.pt"))
                print("⏳ SLURM time limit reached. Saving checkpoint and exiting.")
            break

        if dist.is_initialized():
            train_loader.sampler.set_epoch(epoch)

        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, ema)
        scheduler.step()

        if is_main:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f} - LR: {lr:.6f}")

            if epoch % args.val_interval == 0:
                ema.apply_shadow()
                dice, hd95 = validate(model, val_loader, device, (args.roi_x, args.roi_y, args.roi_z))
                ema.restore()
                print(f"  Validation Dice: {dice:.4f}, HD95: {hd95:.4f}")

                if dice > best_dice:
                    best_dice = dice
                    torch.save(model.state_dict(), os.path.join(args.results_dir, "best_model.pt"))
                    print("  🔥 New best model saved!")

    if is_main:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "final_model.pt"))
        print("Training complete. Final model saved.")


if __name__ == "__main__":
    main()
