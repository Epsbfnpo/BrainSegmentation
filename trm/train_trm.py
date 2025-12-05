#!/usr/bin/env python3
"""Transfer Risk Map baseline training entrypoint."""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader import get_target_dataloaders
from modules import SwinUNETRWrapper, load_pretrained_weights
from trainer_trm import reduce_tensor, train_epoch, validate_epoch
from trm_core import TransferRiskManager


def init_distributed(args) -> bool:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", init_method="env://")
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


def save_checkpoint(state: Dict, path: Path, *, is_main: bool) -> None:
    if not is_main:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Transfer Risk Map training")
    parser.add_argument("--split_json", required=True, help="Path to target split JSON")
    parser.add_argument("--results_dir", default="trm_runs", help="Directory to store logs and checkpoints")
    parser.add_argument("--pretrained_checkpoint", required=True, help="Checkpoint from source model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Epochs to accumulate P(y|z) before freezing")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--roi_x", type=int, default=96)
    parser.add_argument("--roi_y", type=int, default=96)
    parser.add_argument("--roi_z", type=int, default=96)
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[0.8, 0.8, 0.8])
    parser.add_argument("--apply_spacing", action="store_true")
    parser.add_argument("--apply_orientation", action="store_true")
    parser.add_argument("--foreground_only", action="store_true", help="Remap labels to foreground indices")
    parser.add_argument("--use_label_crop", action="store_true")
    parser.add_argument("--label_crop_samples", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=87)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--no_swin_checkpoint", action="store_true")
    parser.add_argument("--cache_rate", type=float, default=0.0)
    parser.add_argument("--cache_num_workers", type=int, default=4)
    parser.add_argument("--volume_stats", type=str, default=None, help="Optional volume stats for weighted sampling")
    parser.add_argument("--laterality_pairs_json", type=str, default=None, help="Optional JSON of LR label pairs")
    parser.add_argument("--enable_weighted_sampling", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--trm_momentum", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    distributed = init_distributed(args)
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = is_main_process(args)

    results_dir = Path(args.results_dir)
    if is_main:
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    train_loader, val_loader = get_target_dataloaders(
        args,
        is_distributed=distributed,
        world_size=args.world_size,
        rank=args.rank,
    )

    target_model = SwinUNETRWrapper(args).to(device)
    source_model = SwinUNETRWrapper(args).to(device)
    load_pretrained_weights(target_model, args.pretrained_checkpoint)
    load_pretrained_weights(source_model, args.pretrained_checkpoint)

    for param in source_model.parameters():
        param.requires_grad = False
    source_model.eval()

    if distributed:
        target_model = DDP(target_model, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = torch.optim.AdamW(target_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    trm_manager = TransferRiskManager(args.out_channels, device=device, momentum=args.trm_momentum)

    best_dice = -math.inf

    try:
        for epoch in range(1, args.epochs + 1):
            if distributed:
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)
            train_loss = train_epoch(
                target_model,
                source_model,
                train_loader,
                optimizer,
                trm_manager,
                device,
                scaler,
                epoch,
                warmup_epochs=args.warmup_epochs,
                accumulation_steps=args.accumulation_steps,
            )

            val_loss, val_metrics = validate_epoch(
                target_model if not isinstance(target_model, DDP) else target_model.module,
                val_loader,
                device,
                num_classes=args.out_channels,
            )
            dice = torch.tensor(val_metrics.get("dice", 0.0), device=device)
            dice = reduce_tensor(dice).item()

            if is_main:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice: {dice:.4f}")

            scheduler.step()

            if dice > best_dice:
                best_dice = dice
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": (target_model.module if isinstance(target_model, DDP) else target_model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_dice": best_dice,
                        "args": vars(args),
                    },
                    results_dir / "best.ckpt",
                    is_main=is_main,
                )
            if epoch % args.save_every == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": (target_model.module if isinstance(target_model, DDP) else target_model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_dice": best_dice,
                        "args": vars(args),
                    },
                    results_dir / f"epoch-{epoch:03d}.ckpt",
                    is_main=is_main,
                )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
