import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from arch.MedNextV1 import MedNeXt
from data_loader import get_loader
from utils import CombinedLoss, ExponentialMovingAverage, train_one_epoch, validate


def init_distributed():
    if "RANK" not in os.environ:
        return False, torch.device("cuda"), True, None

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = dist.get_rank() == 0
    return True, device, is_main, local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_json", required=True, help="Path to dataset split JSON")
    parser.add_argument("--results_dir", default="./results_mednext", help="Directory to save checkpoints")
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
    parser.add_argument("--slurm_time_buffer", type=float, default=300)
    args = parser.parse_args()

    is_distributed, device, is_main, local_rank = init_distributed()
    os.makedirs(args.results_dir, exist_ok=True)

    model = MedNeXt(
        in_channels=args.in_channels,
        n_channels=32,
        n_classes=args.out_channels,
        exp_r=2,
        kernel_size=3,
        deep_supervision=True,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        dim="3d",
        grn=True,
    ).to(device)

    if is_main:
        print("🚀 Initializing MedNeXt from scratch (Target-only training)")

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = CombinedLoss(args.out_channels).to(device)
    ema = ExponentialMovingAverage(model, decay=0.999)

    train_loader, val_loader = get_loader(
        args,
        is_distributed=is_distributed,
        rank=dist.get_rank() if is_distributed else 0,
        world_size=dist.get_world_size() if is_distributed else 1,
    )

    best_dice = 0.0
    job_deadline = None
    if "SLURM_JOB_END_TIME" in os.environ:
        job_deadline = float(os.environ["SLURM_JOB_END_TIME"]) - args.slurm_time_buffer

    for epoch in range(1, args.epochs + 1):
        if job_deadline and time.time() > job_deadline:
            if is_main:
                torch.save({"epoch": epoch, "state_dict": model.state_dict()}, os.path.join(args.results_dir, "latest_model.pt"))
                print("⏳ SLURM time limit reached. Saved latest checkpoint.")
            break

        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, ema)
        scheduler.step()

        if is_main:
            print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

            if epoch % args.val_interval == 0:
                ema.apply_shadow()
                dice, hd95 = validate(model, val_loader, device, (args.roi_x, args.roi_y, args.roi_z))
                ema.restore()

                print(f"  Validation - Dice: {dice:.4f}, HD95: {hd95:.2f}")

                if dice > best_dice:
                    best_dice = dice
                    torch.save(model.state_dict(), os.path.join(args.results_dir, "best_model.pt"))
                    print("  🔥 New best model saved!")

    if is_main:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "final_model.pt"))


if __name__ == "__main__":
    main()
