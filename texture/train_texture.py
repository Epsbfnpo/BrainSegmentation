#!/usr/bin/env python3
"""Train SwinUNETR with texture-centric auxiliary objectives."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
from monai.networks.nets import SwinUNETR
from monai.utils import set_determinism

from data_loader_texture import create_texture_dataloaders
from texture_modules import DomainDiscriminator, StyleEncoder
from trainer_texture import TextureTrainer, TextureTrainingState


_DEF_SPLIT_SOURCE = os.path.join(Path(__file__).resolve().parents[1], "dHCP_split.json")
_DEF_SPLIT_TARGET = os.path.join(Path(__file__).resolve().parents[1], "PPREMOPREBO_split.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Texture-focused domain adaptation training")

    parser.add_argument("--source-split", type=str, default=_DEF_SPLIT_SOURCE, help="Source domain split JSON")
    parser.add_argument("--target-split", type=str, default=_DEF_SPLIT_TARGET, help="Target domain split JSON")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store logs and checkpoints")

    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-rate", type=float, default=0.25)
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--roi", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--apply-spacing", action="store_true", dest="apply_spacing")
    parser.add_argument("--no-apply-spacing", action="store_false", dest="apply_spacing")
    parser.add_argument("--apply-orientation", action="store_true", dest="apply_orientation")
    parser.add_argument("--no-apply-orientation", action="store_false", dest="apply_orientation")
    parser.set_defaults(apply_spacing=True, apply_orientation=True)
    parser.add_argument("--spacing", type=float, nargs=3, default=(0.8, 0.8, 0.8))
    parser.add_argument("--source-repeat", type=int, default=1, help="Repeat factor for source samples in training")
    parser.add_argument("--target-repeat", type=int, default=1, help="Repeat factor for target samples in training")

    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=88)
    parser.add_argument("--feature-size", type=int, default=48)
    parser.add_argument("--swin-drop-rate", type=float, default=0.0)

    parser.add_argument("--lambda-domain", type=float, default=0.5)
    parser.add_argument("--lambda-mmd", type=float, default=0.0)
    parser.add_argument("--style-dim", type=int, default=128)
    parser.add_argument("--style-base-ch", type=int, default=16)
    parser.add_argument("--domain-hidden", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine", "plateau"], default="cosine")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision")
    parser.add_argument("--device", type=str, default="cuda")

    return parser


def prepare_output(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)
    return path


def create_scheduler(name: str, optimizer: torch.optim.Optimizer, epochs: int):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10)
    return None


def save_checkpoint(state: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_determinism(seed=args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = prepare_output(args.output_dir)
    metrics_file = output_dir / "metrics.jsonl"

    loaders = create_texture_dataloaders(
        source_split_json=args.source_split,
        target_split_json=args.target_split,
        roi_size=args.roi,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        cache_num_workers=args.cache_workers,
        apply_spacing=args.apply_spacing,
        target_spacing=args.spacing,
        apply_orientation=args.apply_orientation,
        source_repeat=args.source_repeat,
        target_repeat=args.target_repeat,
    )

    model = SwinUNETR(
        img_size=tuple(args.roi),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=args.swin_drop_rate,
    )

    style_encoder = StyleEncoder(
        in_channels=args.in_channels,
        base_channels=args.style_base_ch,
        embedding_dim=args.style_dim,
    )
    domain_head = DomainDiscriminator(
        embedding_dim=args.style_dim,
        hidden_dim=args.domain_hidden,
        num_domains=2,
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(style_encoder.parameters()) + list(domain_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = create_scheduler(args.scheduler, optimizer, args.epochs)

    trainer = TextureTrainer(
        model=model,
        style_encoder=style_encoder,
        domain_discriminator=domain_head,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=args.out_channels,
        lambda_domain=args.lambda_domain,
        lambda_mmd=args.lambda_mmd,
        use_amp=args.amp,
        max_grad_norm=args.max_grad_norm,
    )

    training_state = TextureTrainingState(epoch=0)

    for epoch in range(1, args.epochs + 1):
        training_state.epoch = epoch
        train_metrics = trainer.train_epoch(loaders["train"], epoch)
        val_source = trainer.evaluate(loaders["val_source"])
        val_target = trainer.evaluate(loaders["val_target"])

        if args.scheduler == "plateau":
            trainer.step_scheduler(metric=val_target["dice"])
        else:
            trainer.step_scheduler()

        current_time = datetime.utcnow().isoformat()
        summary = {
            "epoch": epoch,
            "timestamp": current_time,
            "train": {k: v for k, v in train_metrics.items() if k != "texture_stats"},
            "train_texture": train_metrics.get("texture_stats", {}),
            "val_source": val_source,
            "val_target": val_target,
            "lr": optimizer.param_groups[0]["lr"],
        }

        with metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        print("=" * 80)
        print(f"Epoch {epoch}/{args.epochs} :: LR {summary['lr']:.6f}")
        print(f"  Train total loss: {summary['train']['loss_total']:.4f} | Dice: {summary['train']['dice_train']:.4f}")
        print(f"  Domain acc: {summary['train']['domain_acc']:.3f} | MMD: {summary['train']['loss_mmd']:.4f}")
        print(f"  Val Source Dice: {val_source['dice']:.4f} | Val Target Dice: {val_target['dice']:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state": trainer.model.state_dict(),
            "style_state": trainer.style_encoder.state_dict(),
            "domain_state": trainer.domain_discriminator.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_target_dice": training_state.best_target_dice,
            "args": vars(args),
        }
        save_checkpoint(ckpt, output_dir / "checkpoints" / f"epoch-{epoch:04d}.pt")

        if val_target["dice"] > training_state.best_target_dice:
            training_state.best_target_dice = val_target["dice"]
            save_checkpoint(ckpt, output_dir / "checkpoints" / "best.pt")

    print("Training complete. Best target Dice: {:.4f}".format(training_state.best_target_dice))


if __name__ == "__main__":
    main()
