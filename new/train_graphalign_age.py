#!/usr/bin/env python3
"""Target-only training entrypoint with production diagnostics."""

import argparse
import os
import random
import signal
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from monai.networks.nets import SwinUNETR
from torch.utils.tensorboard import SummaryWriter

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
            print(f"📦 Loading pretrained weights from {args.pretrained_checkpoint}")
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
        use_age_conditioning=False,
        debug_mode=args.debug_mode,
    )
    return wrapper


def get_model_state(model: torch.nn.Module) -> dict:
    return model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()


def save_checkpoint(path: Path,
                    *,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler,
                    epoch: int,
                    global_step: int,
                    best_dice: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_dice": best_dice,
            "state_dict": get_model_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path,
                    *,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler):
    payload = torch.load(path, map_location="cpu")
    model_state = payload.get("state_dict", payload)
    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    target.load_state_dict(model_state, strict=True)

    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])

    return payload.get("epoch", 0), payload.get("global_step", 0), payload.get("best_dice", 0.0)


def register_signal_handlers(flag_container: dict, *, is_main: bool) -> None:
    def _handler(signum, frame):
        if is_main:
            print(f"⚠️  Received signal {signum}; checkpoint will be saved at epoch boundary", flush=True)
        flag_container["triggered"] = True

    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, _handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Target-only age-aware segmentation training")
    parser.add_argument("--split_json", required=True, type=str, help="Target dataset split JSON")
    parser.add_argument("--results_dir", default="./results", type=str)
    parser.add_argument("--log_dir", default=None, type=str, help="TensorBoard log directory")
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
    parser.add_argument("--laterality_pairs_json", type=str, default=None)
    parser.add_argument("--use_label_crop", dest="use_label_crop", action="store_true", default=True)
    parser.add_argument("--no_label_crop", dest="use_label_crop", action="store_false")
    parser.add_argument("--label_crop_samples", type=int, default=1)
    parser.add_argument("--enable_weighted_sampling", action="store_true", default=False)
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
    parser.add_argument("--prior_warmup_epochs", type=int, default=10)
    parser.add_argument("--structural_rules", type=str, default=None)
    parser.add_argument("--lambda_required", type=float, default=0.05)
    parser.add_argument("--lambda_forbidden", type=float, default=0.05)
    parser.add_argument("--lambda_symmetry", type=float, default=0.02)
    parser.add_argument("--lambda_dyn", type=float, default=0.2)
    parser.add_argument("--dyn_start_epoch", type=int, default=60)
    parser.add_argument("--dyn_ramp_epochs", type=int, default=40)
    parser.add_argument("--dyn_mismatch_ref", type=float, default=0.08)
    parser.add_argument("--dyn_max_scale", type=float, default=3.0)
    parser.add_argument("--age_reliability_min", type=float, default=0.3)
    parser.add_argument("--age_reliability_pow", type=float, default=0.5)
    parser.add_argument("--dist_timeout", type=int, default=180)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--use_swin_checkpoint", dest="use_swin_checkpoint", action="store_true", default=True)
    parser.add_argument("--no_swin_checkpoint", dest="use_swin_checkpoint", action="store_false")
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_sliding_window", dest="use_sliding_window", action="store_true", default=True)
    parser.add_argument("--no_eval_sliding_window", dest="use_sliding_window", action="store_false")
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--sw_overlap", type=float, default=0.25)
    parser.add_argument("--multi_scale_eval", action="store_true", default=False)
    parser.add_argument("--eval_scales", type=float, nargs="*", default=[1.0])
    parser.add_argument("--grad_clip", type=float, default=12.0)
    parser.add_argument("--save_interval", type=int, default=20, help="Save checkpoint every N epochs")
    parser.add_argument("--max_keep_ckpt", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--debug_mode", action="store_true", default=False)
    parser.add_argument("--debug_step_limit", type=int, default=2)
    parser.add_argument("--debug_val_limit", type=int, default=1)
    parser.add_argument("--prior_debug_batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    distributed = init_distributed(args)
    is_main = is_main_process(args)

    log_dir = Path(args.log_dir) if args.log_dir else results_dir / "tensorboard"
    writer: Optional[SummaryWriter] = SummaryWriter(log_dir=str(log_dir)) if is_main else None

    signal_state = {"triggered": False}
    register_signal_handlers(signal_state, is_main=is_main)

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
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

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
        structural_rules_path=args.structural_rules,
        lr_pairs_path=args.laterality_pairs_json,
        lambda_volume=args.lambda_volume,
        lambda_shape=args.lambda_shape,
        lambda_edge=args.lambda_edge,
        lambda_spec=args.lambda_spec,
        lambda_required=args.lambda_required,
        lambda_forbidden=args.lambda_forbidden,
        lambda_symmetry=args.lambda_symmetry,
        sdf_temperature=args.sdf_temperature,
        warmup_epochs=args.prior_warmup_epochs,
        lambda_dyn=args.lambda_dyn,
        dyn_start_epoch=args.dyn_start_epoch,
        dyn_ramp_epochs=args.dyn_ramp_epochs,
        dyn_mismatch_ref=args.dyn_mismatch_ref,
        dyn_max_scale=args.dyn_max_scale,
        age_reliability_min=args.age_reliability_min,
        age_reliability_pow=args.age_reliability_pow,
        debug=args.debug_mode,
        debug_max_batches=args.prior_debug_batches,
    ).to(device)
    prior_loss.configure_schedule(args.epochs)
    prior_loss.set_debug(args.debug_mode, args.prior_debug_batches)

    best_dice = 0.0
    global_step = 0
    start_epoch = 1
    checkpoint_history: deque[Path] = deque()

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            resume_epoch, resume_step, resume_best = load_checkpoint(
                resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            start_epoch = resume_epoch + 1
            global_step = resume_step
            best_dice = resume_best
            if is_main:
                print(f"🔁 Resumed from {resume_path} at epoch {resume_epoch}, global_step {global_step}, best dice {best_dice:.4f}")
        elif is_main:
            print(f"⚠️  Resume checkpoint {resume_path} not found; starting from scratch")

    if is_main:
        print(f"📂 Results directory: {results_dir}")
        print(f"📝 Logging to: {log_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        prior_loss.set_epoch(epoch)

        start_time = time.time()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            prior_loss,
            device=device,
            epoch=epoch,
            use_amp=args.use_amp,
            grad_clip=args.grad_clip,
            writer=writer,
            global_step=global_step,
            is_main=is_main,
            log_interval=args.log_interval,
            debug_mode=args.debug_mode,
            debug_step_limit=args.debug_step_limit,
        )
        scheduler.step()
        global_step = int(train_metrics.get("global_step", global_step))
        duration = time.time() - start_time

        if is_main:
            print(
                f"Epoch {epoch:03d}: loss={train_metrics['loss']:.4f} seg={train_metrics['seg']:.4f} "
                f"prior={train_metrics['prior']:.4f} warmup={train_metrics.get('warmup', 1.0):.3f} "
                f"edge={train_metrics.get('edge', 0.0):.4f} spec={train_metrics.get('spectral', 0.0):.4f} "
                f"req={train_metrics.get('required', 0.0):.4f} forb={train_metrics.get('forbidden', 0.0):.4f} "
                f"sym={train_metrics.get('symmetry', 0.0):.4f} dyn={train_metrics.get('dyn_lambda', 1.0):.3f} "
                f"grad={train_metrics.get('grad_norm', 0.0):.3f} time={duration:.1f}s",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar("train/lr", train_metrics["lr"], epoch)

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_metrics = validate_epoch(
                model,
                val_loader,
                device=device,
                num_classes=args.out_channels,
                foreground_only=args.foreground_only,
                use_sliding_window=args.use_sliding_window,
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
                sw_batch_size=args.sw_batch_size,
                sw_overlap=args.sw_overlap,
                multi_scale=args.multi_scale_eval,
                eval_scales=args.eval_scales,
                debug_mode=args.debug_mode,
                debug_step_limit=args.debug_val_limit,
                is_main=is_main,
                prior_loss=prior_loss,
            )
            if is_main:
                extra_msgs = []
                adj_errors = val_metrics.get("adjacency_errors")
                if adj_errors:
                    extra_msgs.append(f"adj_mae={adj_errors.get('mean_adj_error', 0.0):.4f}")
                    extra_msgs.append(f"spec={adj_errors.get('spectral_distance', 0.0):.4f}")
                struct = val_metrics.get("structural_violations")
                if struct:
                    extra_msgs.append(f"req_miss={struct.get('required_missing', 0.0):.2f}")
                    extra_msgs.append(f"forb={struct.get('forbidden_present', 0.0):.2f}")
                sym_scores = val_metrics.get("symmetry_scores")
                if sym_scores:
                    extra_msgs.append(f"sym={sym_scores[0]:.4f}")
                msg_extra = " ".join(extra_msgs)
                print(f"  Validation dice={val_metrics['dice']:.4f} {msg_extra}".strip())
                if writer is not None:
                    writer.add_scalar("val/dice", val_metrics["dice"], epoch)
                    if adj_errors:
                        writer.add_scalar("val/adjacency_mae", adj_errors.get("mean_adj_error", 0.0), epoch)
                        writer.add_scalar("val/spectral_gap", adj_errors.get("spectral_distance", 0.0), epoch)
                    if struct:
                        writer.add_scalar("val/required_missing", struct.get("required_missing", 0.0), epoch)
                        writer.add_scalar("val/forbidden_present", struct.get("forbidden_present", 0.0), epoch)
                    if sym_scores:
                        writer.add_scalar("val/symmetry_score", sym_scores[0], epoch)
                if val_metrics["dice"] > best_dice:
                    best_dice = val_metrics["dice"]
                    best_path = results_dir / "best_model.pt"
                    save_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        best_dice=best_dice,
                    )
                    print(f"  ✅ New best checkpoint saved to {best_path}")

        if is_main and args.save_interval > 0 and epoch % args.save_interval == 0:
            ckpt_path = results_dir / f"checkpoint_epoch{epoch:03d}.pt"
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_dice=best_dice,
            )
            checkpoint_history.append(ckpt_path)
            while len(checkpoint_history) > args.max_keep_ckpt:
                old = checkpoint_history.popleft()
                if old.exists():
                    old.unlink()
                    print(f"  🧹 Removed old checkpoint {old}")

        if signal_state.get("triggered") and is_main:
            emergency_path = results_dir / f"checkpoint_signal_epoch{epoch:03d}.pt"
            save_checkpoint(
                emergency_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_dice=best_dice,
            )
            print(f"  💾 Signal-triggered checkpoint saved to {emergency_path}")
            checkpoint_history.append(emergency_path)
            signal_state["triggered"] = False

    if writer is not None:
        writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()
