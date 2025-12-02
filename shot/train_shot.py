#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from torch.utils.tensorboard import SummaryWriter

from age_aware_modules import SimplifiedDAUnetModule
from data_loader_age_aware import get_target_dataloaders
from shot_core import configure_model_for_shot, shot_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Source-free SHOT adaptation")
    parser.add_argument("--split_json", required=True, help="Path to target train/val split JSON")
    parser.add_argument("--results_dir", default="./results_shot")
    parser.add_argument("--pretrained_checkpoint", required=True, help="Checkpoint trained on source domain")

    # Optimisation
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--diversity_weight", default=1.0, type=float, help="β in L = Ent - β*Div")
    parser.add_argument("--eval_interval", default=5, type=int, help="Validate every N epochs")
    parser.add_argument("--save_interval", default=1, type=int, help="Write latest checkpoint every N epochs")

    # Model
    parser.add_argument("--roi_x", default=96, type=int)
    parser.add_argument("--roi_y", default=96, type=int)
    parser.add_argument("--roi_z", default=96, type=int)
    parser.add_argument("--out_channels", default=87, type=int)
    parser.add_argument("--feature_size", default=48, type=int)
    parser.add_argument("--use_swin_checkpoint", action="store_true", help="Enable Swin gradient checkpointing")

    # Data handling
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

    # Housekeeping
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--slurm_time_buffer", default=300, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def init_distributed() -> Tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_dice: float) -> None:
    state = {
        "epoch": epoch,
        "best_dice": best_dice,
        "state_dict": model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)


def load_weights_exact(model: SimplifiedDAUnetModule, checkpoint_path: str) -> None:
    if is_main_process():
        print(f"📦 Loading weights from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    msg = model.backbone.load_state_dict(state_dict, strict=False)
    if is_main_process():
        print(f"   Missing keys: {len(msg.missing_keys)}")
        print(f"   Unexpected keys: {len(msg.unexpected_keys)}")
        if msg.missing_keys:
            print(f"   Sample missing: {msg.missing_keys[:3]}")


def validate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, args: argparse.Namespace) -> float:
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
            val_outputs_onehot = F.one_hot(val_outputs.squeeze(1).long(), num_classes=args.out_channels).permute(0, 4, 1, 2, 3)
            val_labels_onehot = F.one_hot(val_labels.squeeze(1).long(), num_classes=args.out_channels).permute(0, 4, 1, 2, 3)

            dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)

    dice = dice_metric.aggregate().to(device)
    dice_metric.reset()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(dice, op=dist.ReduceOp.SUM)
        dice = dice / dist.get_world_size()

    return dice.item()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    is_distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_target_dataloaders(args, is_distributed=is_distributed, world_size=world_size, rank=rank)

    backbone = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=1,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_swin_checkpoint,
    ).to(device)

    model = SimplifiedDAUnetModule(backbone, num_classes=args.out_channels).to(device)

    if args.pretrained_checkpoint:
        load_weights_exact(model, args.pretrained_checkpoint)

    params_to_update = configure_model_for_shot(model, verbose=is_main_process())

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(params_to_update, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 1
    best_dice = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_dice", 0.0)
        (model.module if is_distributed else model).load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if is_main_process():
            print(f"Resumed from epoch {start_epoch - 1}")

    job_deadline = None
    if "SLURM_JOB_END_TIME" in os.environ:
        job_deadline = float(os.environ["SLURM_JOB_END_TIME"]) - args.slurm_time_buffer
        if is_main_process():
            print(f"Job deadline set. Will stop at timestamp {job_deadline}")

    writer = SummaryWriter(log_dir=os.path.join(args.results_dir, "logs")) if is_main_process() else None

    for epoch in range(start_epoch, args.epochs + 1):
        if job_deadline and time.time() > job_deadline:
            if is_main_process():
                print("Time limit reached. Saving and exiting for requeue.")
            break

        if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_total_loss = 0.0
        epoch_ent = 0.0
        epoch_div = 0.0
        steps = 0

        for batch in train_loader:
            img = batch["image"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(img)
                loss, ent, div = shot_loss(logits, diversity_weight=args.diversity_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_total_loss += loss.item()
            epoch_ent += ent.item()
            epoch_div += div.item()
            steps += 1

        if is_main_process():
            avg_loss = epoch_total_loss / max(steps, 1)
            avg_ent = epoch_ent / max(steps, 1)
            avg_div = epoch_div / max(steps, 1)

            print(f"Epoch {epoch}: Total={avg_loss:.4f} (Ent={avg_ent:.4f} - Div={avg_div:.4f})")
            if writer:
                writer.add_scalar("train/loss", avg_loss, epoch)
                writer.add_scalar("train/ent", avg_ent, epoch)
                writer.add_scalar("train/div", avg_div, epoch)

        should_validate = (epoch % max(args.eval_interval, 1) == 0)
        if should_validate:
            dice = validate(model, val_loader, device, args)
            if is_main_process():
                print(f"Epoch {epoch} Val Dice: {dice:.4f}")
                if writer:
                    writer.add_scalar("val/dice", dice, epoch)

                if dice > best_dice:
                    best_dice = dice
                    save_checkpoint(os.path.join(args.results_dir, "best_model.pt"), model, optimizer, epoch, best_dice)

        if epoch % max(args.save_interval, 1) == 0 and is_main_process():
            save_checkpoint(os.path.join(args.results_dir, "latest_model.pt"), model, optimizer, epoch, best_dice)

    if is_main_process() and (not job_deadline or time.time() < job_deadline):
        save_checkpoint(os.path.join(args.results_dir, "final_model.pt"), model, optimizer, epoch, best_dice)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
