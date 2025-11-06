#!/usr/bin/env python3
"""Training entry-point for the texture-focused adaptation pipeline."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import torch
from monai.utils import set_determinism

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


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Texture-centric domain adaptation for brain segmentation")

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
    parser.add_argument("--out_channels", type=int, default=88)
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--texture_embed_dim", type=int, default=128)
    parser.add_argument("--texture_stats_proj_dim", type=int, default=128)
    parser.add_argument("--texture_domain_hidden", type=int, default=128)
    parser.add_argument("--grl_lambda", type=float, default=1.0)

    # Loss weights
    parser.add_argument("--domain_loss_weight", type=float, default=0.5)
    parser.add_argument("--embed_align_weight", type=float, default=0.1)
    parser.add_argument("--stats_align_weight", type=float, default=0.1)

    # Checkpointing
    parser.add_argument("--pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=10)

    return parser


def _log_message(log_path: str, message: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message, flush=True)


def _infer_texture_dim(loader) -> int:
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


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    set_determinism(seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, "training.log")
    history_path = os.path.join(args.results_dir, "metrics_history.json")

    _log_message(log_path, "Starting texture adaptation run")
    _log_message(log_path, json.dumps(vars(args), indent=2))

    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    train_loader, val_loader = create_texture_dataloaders(
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
    )

    texture_stats_dim = _infer_texture_dim(train_loader)
    _log_message(log_path, f"Inferred texture feature dimension: {texture_stats_dim}")

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    loss_fn = build_loss(num_classes=args.out_channels, include_background=True)
    dice_metric = build_dice_metric(num_classes=args.out_channels, include_background=False)

    start_epoch = 1
    best_metric: float | None = None

    if args.pretrained_checkpoint:
        _log_message(log_path, f"Loading pretrained weights from {args.pretrained_checkpoint}")
        if os.path.isfile(args.pretrained_checkpoint):
            checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.segmenter.load_state_dict(state_dict, strict=False)
        else:
            _log_message(log_path, "⚠️ Pretrained checkpoint not found, continuing without it")

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        _log_message(log_path, f"Resuming from {args.resume_checkpoint}")
        checkpoint = load_checkpoint(model, optimizer, args.resume_checkpoint, map_location=device)
        start_epoch = checkpoint.get("epoch", start_epoch)
        best_metric = checkpoint.get("best_metric", best_metric)

    metrics_history = []
    if os.path.isfile(history_path):
        try:
            with open(history_path, "r") as f:
                metrics_history = json.load(f)
        except Exception:
            metrics_history = []

    for epoch in range(start_epoch, args.epochs + 1):
        _log_message(log_path, f"Epoch {epoch}/{args.epochs}")

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
        )

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
        )

        scheduler.step()

        _log_message(
            log_path,
            (
                f"Epoch {epoch}: total_loss={train_stats['total_loss']:.4f}, seg_loss={train_stats['seg_loss']:.4f}, "
                f"domain_loss={train_stats['domain_loss']:.4f}, align_loss={train_stats['align_loss']:.4f}, "
                f"stats_align_loss={train_stats['stats_align_loss']:.4f}, domain_acc={train_stats['domain_acc']:.3f}, "
                f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f}"
            ),
        )

        metrics_history.append(
            {
                "epoch": epoch,
                **train_stats,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        with open(history_path, "w") as f:
            json.dump(metrics_history, f, indent=2)

        if best_metric is None or val_dice > best_metric:
            best_metric = val_dice
            best_path = os.path.join(args.results_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, best_path, best_metric=best_metric)
            _log_message(log_path, f"  ✅ New best model saved (dice={best_metric:.4f})")

        if epoch % max(1, args.save_every) == 0:
            ckpt_path = os.path.join(args.results_dir, f"checkpoint_epoch_{epoch:03d}.pth")
            save_checkpoint(model, optimizer, epoch, ckpt_path, best_metric=best_metric)

    final_path = os.path.join(args.results_dir, "last_model.pth")
    save_checkpoint(model, optimizer, args.epochs, final_path, best_metric=best_metric)
    _log_message(log_path, f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
