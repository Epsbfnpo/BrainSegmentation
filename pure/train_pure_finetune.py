#!/usr/bin/env python3
"""Pure fine-tuning script without causal or graph-based regularisation."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import torch
from monai.networks.nets import SwinUNETR
from monai.utils import set_determinism

from data_loader_pure import create_dataloaders
from trainer_pure import (
    build_dice_metric,
    build_loss,
    load_checkpoint,
    save_checkpoint,
    train_epoch,
    validate,
)


_DEF_RESULTS = "./results/pure_finetune"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pure fine-tuning on target dataset")

    # Data
    parser.add_argument("--split_json", type=str, required=True, help="Path to target dataset split JSON")
    parser.add_argument("--results_dir", type=str, default=_DEF_RESULTS, help="Directory to store checkpoints and logs")
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

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision")
    parser.set_defaults(amp=True)

    # Model
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=88)
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Checkpointing
    parser.add_argument("--pretrained_checkpoint", type=str, default=None, help="Checkpoint with source-domain weights")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")

    return parser


def _log_message(log_path: str, message: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message, flush=True)


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    set_determinism(seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, "training.log")
    history_path = os.path.join(args.results_dir, "metrics_history.json")

    _log_message(log_path, "Starting pure fine-tuning run")
    _log_message(log_path, json.dumps(vars(args), indent=2))

    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    train_loader, val_loader = create_dataloaders(
        args.split_json,
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

    model = SwinUNETR(
        img_size=roi_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=True,
        dropout_rate=args.dropout,
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    loss_fn = build_loss(num_classes=args.out_channels, include_background=True)
    dice_metric = build_dice_metric(num_classes=args.out_channels, include_background=False)

    start_epoch = 1
    best_metric = None

    if args.pretrained_checkpoint:
        _log_message(log_path, f"Loading pretrained weights from {args.pretrained_checkpoint}")
        if os.path.isfile(args.pretrained_checkpoint):
            checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
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

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            amp=args.amp and device.type == "cuda",
            scaler=scaler,
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
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_dice={val_dice:.4f}",
        )

        metrics_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
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
            ckpt_path = os.path.join(args.results_dir, f"epoch_{epoch:04d}.pth")
            save_checkpoint(model, optimizer, epoch, ckpt_path, best_metric=best_metric)

    final_path = os.path.join(args.results_dir, "final_model.pth")
    save_checkpoint(model, optimizer, args.epochs, final_path, best_metric=best_metric)
    _log_message(log_path, "Training completed")


if __name__ == "__main__":
    main()
