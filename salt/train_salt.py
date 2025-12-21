#!/usr/bin/env python3

"""SALT training entrypoint (Adapted from L2-SP)."""

import argparse
import os
import time
import random
import signal
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets import SwinUNETR

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.append(str(SCRIPT_DIR))

# Local independent imports
from modules import SimplifiedDAUnetModule, apply_salt_to_model
from data_loader import get_target_dataloaders
from trainer_salt import CombinedSegmentationLoss, ExponentialMovingAverage, train_epoch, validate_epoch


# --- Utility functions copied from L2-SP for consistency ---

def parse_slurm_timelimit(raw):
    if not raw:
        return None
    if "-" in raw:
        days, rest = raw.split("-")
        total = int(days) * 24 * 3600
    else:
        total = 0
        rest = raw
    parts = [int(x) for x in rest.split(":")]
    if len(parts) == 3:
        total += parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        total += parts[0] * 60 + parts[1]
    else:
        total += parts[0] * 60
    return float(total)


def compute_job_deadline(buffer_seconds):
    end = os.environ.get("SLURM_JOB_END_TIME")
    if end:
        return float(end) - buffer_seconds
    start = os.environ.get("SLURM_JOB_START_TIME")
    limit = os.environ.get("SLURM_JOB_TIME_LIMIT")
    if start and limit:
        return float(start) + parse_slurm_timelimit(limit) - buffer_seconds
    return None


def record_resume_checkpoint(results_dir, path):
    (results_dir / "resume_from.txt").write_text(str(path.resolve()), encoding="utf-8")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process(args):
    return args.rank == 0


def init_distributed(args):
    if "RANK" in os.environ:
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


def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, best_dice, ema_state=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_dice": best_dice,
            "state_dict": state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema_state_dict": ema_state,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler):
    payload = torch.load(path, map_location="cpu")
    target = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    target.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return (
        payload.get("epoch", 0),
        payload.get("global_step", 0),
        payload.get("best_dice", 0.0),
        payload.get("ema_state_dict"),
    )


# --- Model Building (The Critical Part) ---

def build_model(args, device):
    # 1. Instantiate the Source Architecture (Exactly as L2-SP)
    backbone = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_swin_checkpoint,
    ).to(device)

    # 2. Load Pretrained Weights (Source-Free requirement)
    if args.pretrained_checkpoint:
        if is_main_process(args):
            print(f"📦 Loading source weights from {args.pretrained_checkpoint}")
        state = torch.load(args.pretrained_checkpoint, map_location=device)
        state_dict = state.get("state_dict", state)
        backbone.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
    else:
        raise ValueError("SALT requires a pretrained source model to adapt!")

    # 3. Apply SALT Transformation
    # This replaces Conv3d/Linear with SALT layers and freezes the original weights
    if is_main_process(args):
        print(f"🧂 Applying SALT adaptation (Rank={args.salt_rank}, LoRA={args.salt_lora_rank})")
    backbone = apply_salt_to_model(backbone, rank=args.salt_rank, r_lora=args.salt_lora_rank)

    # 4. Wrap in Module (handles class weights etc.)
    wrapper = SimplifiedDAUnetModule(
        backbone,
        num_classes=args.out_channels,
        volume_stats_path=args.volume_stats,
        foreground_only=args.foreground_only,
        enhanced_class_weights=args.enhanced_class_weights,
    )
    return wrapper.to(device)


def parse_args():
    parser = argparse.ArgumentParser()

    # Standard L2-SP args
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--volume_stats", default=None)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--roi_x", default=96, type=int)
    parser.add_argument("--roi_y", default=96, type=int)
    parser.add_argument("--roi_z", default=96, type=int)
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=87, type=int)
    parser.add_argument("--feature_size", default=48, type=int)
    parser.add_argument("--foreground_only", action="store_true", default=True)
    parser.add_argument("--enhanced_class_weights", action="store_true", default=True)
    parser.add_argument("--use_swin_checkpoint", action="store_true", default=True)
    parser.add_argument("--slurm_time_buffer", default=300, type=float)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--grad_accum_steps", default=1, type=int)

    # SALT specific args
    parser.add_argument("--salt_rank", default=32, type=int, help="Rank for SVD Scale/Shift")
    parser.add_argument("--salt_lora_rank", default=8, type=int, help="Rank for LoRA residual")
    parser.add_argument("--salt_reg_weight", default=0.01, type=float, help="Regularization weight")

    # Extra data loader args required by `get_target_dataloaders`
    parser.add_argument("--cache_rate", default=0.0, type=float)
    parser.add_argument("--cache_num_workers", default=4, type=int)
    parser.add_argument("--apply_spacing", action="store_true", default=True)
    parser.add_argument("--apply_orientation", action="store_true", default=True)
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[0.8, 0.8, 0.8])
    parser.add_argument("--use_label_crop", action="store_true", default=True)
    parser.add_argument("--label_crop_samples", default=1, type=int)
    parser.add_argument("--enable_weighted_sampling", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    distributed = init_distributed(args)
    is_main = is_main_process(args)
    device = torch.device(f"cuda:{args.local_rank}")

    # Signal handling for Graceful Exit
    stop_event = {"triggered": False}

    def sig_handler(signum, frame):
        if is_main:
            print(f"⚠️ Received signal {signum}, scheduling stop.")
        stop_event["triggered"] = True

    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGUSR1, sig_handler)

    job_deadline = compute_job_deadline(args.slurm_time_buffer)

    # Data & Model
    train_loader, val_loader = get_target_dataloaders(
        args, is_distributed=distributed, world_size=args.world_size, rank=args.rank
    )
    model = build_model(args, device)

    # Optimizer (only train SALT parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        print(f"🔧 Trainable parameters: {len(trainable_params)} tensors")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = CombinedSegmentationLoss(args.out_channels, model.class_weights, args.foreground_only)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False
        )

    # State tracking
    start_epoch = 1
    best_dice = 0.0
    global_step = 0
    latest_ckpt = results_dir / "latest_model.pt"

    # Resume
    if args.resume:
        start_epoch, global_step, best_dice, _ = load_checkpoint(Path(args.resume), model, optimizer, scheduler)
        if is_main:
            print(f"🔁 Resumed from {args.resume} at epoch {start_epoch}")
    elif (results_dir / "resume_from.txt").exists():
        resume_path = (results_dir / "resume_from.txt").read_text().strip()
        if resume_path and Path(resume_path).exists():
            start_epoch, global_step, best_dice, _ = load_checkpoint(Path(resume_path), model, optimizer, scheduler)
            if is_main:
                print(f"🔁 Auto-resumed from {resume_path}")

    writer = SummaryWriter(log_dir=str(results_dir / "logs")) if is_main else None

    # Loop
    for epoch in range(start_epoch, args.epochs + 1):
        # Time check
        if stop_event["triggered"] or (job_deadline and time.time() > job_deadline):
            if is_main:
                print("🛑 Time limit or Signal. Saving latest and exiting.")
            save_checkpoint(latest_ckpt, model, optimizer, scheduler, epoch - 1, global_step, best_dice)
            record_resume_checkpoint(results_dir, latest_ckpt)
            return 0  # Exit code 0 for requeue logic in sbatch

        if distributed:
            train_loader.sampler.set_epoch(epoch)

        metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=device,
            epoch=epoch,
            salt_reg_weight=args.salt_reg_weight,
            grad_accum_steps=args.grad_accum_steps,
            writer=writer,
            global_step=global_step,
            is_main=is_main,
        )
        scheduler.step()
        global_step = metrics["global_step"]

        if is_main:
            print(f"Epoch {epoch:03d}: loss={metrics['loss']:.4f} seg={metrics['seg']:.4f} reg={metrics['salt_reg']:.4f}")

        # Validation
        if epoch % 5 == 0:
            val_metrics = validate_epoch(
                model,
                val_loader,
                device=device,
                num_classes=args.out_channels,
                foreground_only=args.foreground_only,
                is_main=is_main,
            )
            if is_main:
                print(f"  Validation dice={val_metrics['dice']:.4f}")
                if val_metrics['dice'] > best_dice:
                    best_dice = val_metrics['dice']
                    best_ckpt = results_dir / "best_model.pt"
                    save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch, global_step, best_dice)
                    record_resume_checkpoint(results_dir, best_ckpt)
                    print("  ✅ New best saved.")

    # Final save
    if is_main:
        final_ckpt = results_dir / "final_model.pt"
        save_checkpoint(final_ckpt, model, optimizer, scheduler, args.epochs, global_step, best_dice)
        print("🏁 Training complete.")


if __name__ == "__main__":
    main()
