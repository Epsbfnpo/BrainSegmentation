#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta

from monai.networks.nets import SwinUNETR

from age_aware_modules import SimplifiedDAUnetModule
from data_loader_age_aware import get_target_dataloaders
from graph_prior_loss import AgeConditionedGraphPriorLoss
from trainer_age_aware import CombinedSegmentationLoss, train_epoch, validate_epoch


def init_distributed(args) -> bool:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", init_method="env://", timeout=timedelta(minutes=args.dist_timeout))
        torch.cuda.set_device(args.local_rank)
        return True
    else:
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


def build_model(args, device: torch.device) -> SimplifiedDAUnetModule:
    backbone = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_swin_checkpoint,
    ).to(device)

    if args.pretrained_checkpoint:
        if is_main_process(args):
            print(f"Loading pretrained weights from {args.pretrained_checkpoint}")
        state = torch.load(args.pretrained_checkpoint, map_location=device)
        if "state_dict" in state:
            state = state["state_dict"]
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        backbone.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)

    wrapper = SimplifiedDAUnetModule(
        backbone,
        num_classes=args.out_channels,
        class_prior_path=args.class_prior_json,
        foreground_only=args.foreground_only,
        enhanced_class_weights=args.enhanced_class_weights,
    )
    return wrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Target-only age-aware segmentation training")
    parser.add_argument("--split_json", required=True, type=str, help="Target dataset split JSON")
    parser.add_argument("--results_dir", default="./results", type=str)
    parser.add_argument("--epochs", default=400, type=int)
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
    parser.add_argument("--class_prior_json", type=str, default=None)
    parser.add_argument("--enhanced_class_weights", action="store_true", default=True)
    parser.add_argument("--pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--loss_config", type=str, default="dice_focal", choices=["dice_ce", "dice_focal", "dice_ce_focal"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--volume_stats", type=str, default=None)
    parser.add_argument("--sdf_templates", type=str, default=None)
    parser.add_argument("--adjacency_prior", type=str, default=None)
    parser.add_argument("--restricted_mask", type=str, default=None)
    parser.add_argument("--lambda_volume", type=float, default=0.2)
    parser.add_argument("--lambda_shape", type=float, default=0.2)
    parser.add_argument("--lambda_edge", type=float, default=0.1)
    parser.add_argument("--lambda_spec", type=float, default=0.05)
    parser.add_argument("--sdf_temperature", type=float, default=4.0)
    parser.add_argument("--dist_timeout", type=int, default=180)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--use_swin_checkpoint", dest="use_swin_checkpoint", action="store_true", default=True)
    parser.add_argument("--no_swin_checkpoint", dest="use_swin_checkpoint", action="store_false")
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    results_dir = Path(args.results_dir)
    if is_main_process(args):
        results_dir.mkdir(parents=True, exist_ok=True)

    distributed = init_distributed(args)
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        args.use_amp = False

    train_loader, val_loader = get_target_dataloaders(
        args,
        is_distributed=distributed,
        world_size=args.world_size,
        rank=args.rank,
    )

    model = build_model(args, device)
    class_weights = model.get_class_weights()

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    loss_fn = CombinedSegmentationLoss(
        num_classes=args.out_channels,
        class_weights=class_weights,
        foreground_only=args.foreground_only,
        loss_config=args.loss_config,
        focal_gamma=args.focal_gamma,
    )

    prior_loss = AgeConditionedGraphPriorLoss(
        num_classes=args.out_channels,
        volume_stats_path=args.volume_stats,
        sdf_templates_path=args.sdf_templates,
        adjacency_prior_path=args.adjacency_prior,
        r_mask_path=args.restricted_mask,
        lambda_volume=args.lambda_volume,
        lambda_shape=args.lambda_shape,
        lambda_edge=args.lambda_edge,
        lambda_spec=args.lambda_spec,
        sdf_temperature=args.sdf_temperature,
    ).to(device)

    best_dice = 0.0
    best_path = results_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        start = time.time()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            prior_loss,
            device=device,
            epoch=epoch,
            use_amp=args.use_amp,
        )
        scheduler.step()
        duration = time.time() - start

        if is_main_process(args):
            print(f"Epoch {epoch:03d}: train loss={train_metrics['loss']:.4f} seg={train_metrics['seg']:.4f} prior={train_metrics['prior']:.4f} time={duration:.1f}s")

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_metrics = validate_epoch(
                model,
                val_loader,
                device=device,
                num_classes=args.out_channels,
                foreground_only=args.foreground_only,
            )
            if is_main_process(args):
                print(f"  Validation dice={val_metrics['dice']:.4f}")
                if val_metrics['dice'] > best_dice:
                    best_dice = val_metrics['dice']
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'dice': best_dice}, best_path)
                    print(f"  ✓ New best model saved to {best_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
