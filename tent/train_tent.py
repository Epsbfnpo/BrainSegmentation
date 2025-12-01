#!/usr/bin/env python3
import argparse
import os
import time
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

# Local imports (copies in tent folder)
from age_aware_modules import SimplifiedDAUnetModule
from data_loader_age_aware import get_target_dataloaders
from tent_core import configure_model_for_tent, softmax_entropy


def parse_args():
    parser = argparse.ArgumentParser(description="TENT Adaptation")
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--results_dir", default="./results_tent")
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate for TENT")
    parser.add_argument("--epochs", default=50, type=int, help="Number of TENT adaptation epochs")
    parser.add_argument("--save_interval", default=10, type=int, help="How often to save checkpoints")
    parser.add_argument("--eval_interval", default=5, type=int, help="Validation interval in epochs")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--roi_x", default=96, type=int)
    parser.add_argument("--roi_y", default=96, type=int)
    parser.add_argument("--roi_z", default=96, type=int)
    parser.add_argument("--out_channels", default=87, type=int)
    parser.add_argument("--feature_size", default=48, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--slurm_time_buffer", default=300, type=float)
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[0.8, 0.8, 0.8])
    parser.add_argument("--cache_rate", default=0.0, type=float)
    parser.add_argument("--cache_num_workers", default=4, type=int)
    parser.add_argument("--foreground_only", action="store_true", default=True)
    parser.add_argument("--use_label_crop", action="store_true", default=True)
    parser.add_argument("--label_crop_samples", default=1, type=int)
    parser.add_argument("--enable_weighted_sampling", action="store_true", default=False)
    parser.add_argument("--volume_stats", default=None)
    parser.add_argument("--laterality_pairs_json", default=None)
    parser.add_argument("--apply_spacing", action="store_true", default=True)
    parser.add_argument("--apply_orientation", action="store_true", default=True)
    parser.add_argument("--use_swin_checkpoint", action="store_true", default=True)
    return parser.parse_args()


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def save_checkpoint(path, model, optimizer, epoch, best_dice):
    state = {
        "epoch": epoch,
        "best_dice": best_dice,
        "state_dict": model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)


def validate(model, loader, device, args):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    with torch.no_grad():
        for batch in loader:
            val_images = batch["image"].to(device)
            val_labels = batch["label"].to(device)

            val_outputs = sliding_window_inference(
                val_images, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.25
            )

            val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
            val_outputs_onehot = F.one_hot(
                val_outputs.squeeze(1).long(), num_classes=args.out_channels
            ).permute(0, 4, 1, 2, 3)

            val_labels_onehot = F.one_hot(
                val_labels.squeeze(1).long(), num_classes=args.out_channels
            ).permute(0, 4, 1, 2, 3)

            dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)

    result = dice_metric.aggregate().item()
    dice_metric.reset()
    return result


def main():
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    is_distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train_loader, val_loader = get_target_dataloaders(
        args, is_distributed=is_distributed, world_size=world_size, rank=rank
    )

    backbone = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=1,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_swin_checkpoint,
    ).to(device)
    model = SimplifiedDAUnetModule(backbone, num_classes=args.out_channels).to(device)

    if args.pretrained_checkpoint:
        ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        new_state = {}
        for k, v in state.items():
            key = k.replace("module.", "")
            if not key.startswith("backbone."):
                key = "backbone." + key
            new_state[key] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if rank == 0:
            print(f"Loaded source weights from {args.pretrained_checkpoint}")
            if missing:
                print(f"Missing keys: {missing}")
            if unexpected:
                print(f"Unexpected keys: {unexpected}")

    tent_params, param_names = configure_model_for_tent(model)
    if rank == 0:
        print(f"TENT Active: Updating {len(tent_params)} affine parameters in Norm layers.")
        if param_names:
            print(" - " + "\n - ".join(param_names))

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    optimizer = torch.optim.AdamW(tent_params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 1
    best_dice = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt.get("best_dice", 0.0)
        if is_distributed:
            model.module.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if rank == 0:
            print(f"Resumed from epoch {start_epoch - 1}")

    job_deadline = None
    if "SLURM_JOB_END_TIME" in os.environ:
        job_deadline = float(os.environ["SLURM_JOB_END_TIME"]) - args.slurm_time_buffer
        if rank == 0:
            print(f"Job deadline set. Will stop at timestamp {job_deadline}")

    writer = SummaryWriter(log_dir=os.path.join(args.results_dir, "logs")) if rank == 0 else None

    for epoch in range(start_epoch, args.epochs + 1):
        if job_deadline and time.time() > job_deadline:
            if rank == 0:
                print("Time limit reached. Saving and exiting for requeue.")
            save_checkpoint(
                os.path.join(args.results_dir, "latest_model.pt"), model, optimizer, epoch - 1, best_dice
            )
            break

        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        steps = 0

        for batch in train_loader:
            img = batch["image"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(img)
                loss = softmax_entropy(logits).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / steps if steps > 0 else 0.0
        if rank == 0:
            print(f"Epoch {epoch}: Entropy Loss = {avg_loss:.6f}")
            if writer:
                writer.add_scalar("train/entropy", avg_loss, epoch)

        if rank == 0 and (epoch % args.eval_interval == 0 or epoch == args.epochs):
            dice = validate(model, val_loader, device, args)
            print(f"Epoch {epoch} Val Dice: {dice:.4f}")
            if writer:
                writer.add_scalar("val/dice", dice, epoch)

            if dice > best_dice:
                best_dice = dice
                save_checkpoint(
                    os.path.join(args.results_dir, "best_model.pt"), model, optimizer, epoch, best_dice
                )

        if rank == 0 and (epoch % args.save_interval == 0 or epoch == args.epochs):
            save_checkpoint(
                os.path.join(args.results_dir, "latest_model.pt"), model, optimizer, epoch, best_dice
            )

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
