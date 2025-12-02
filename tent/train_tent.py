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
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric

# Local imports
from age_aware_modules import SimplifiedDAUnetModule
from data_loader_age_aware import get_target_dataloaders
# TENT core logic
from tent_core import configure_model_for_tent, softmax_entropy
# Extra metrics
from extra_metrics import compute_cldice, compute_cbdice, compute_clce, compute_rve


def parse_args():
    parser = argparse.ArgumentParser(description="TENT Adaptation")
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--results_dir", default="./results_tent")
    parser.add_argument("--pretrained_checkpoint", required=True)
    # TENT specific
    parser.add_argument("--lr", default=1e-4, type=float, help="Lower learning rate for TENT stability")
    parser.add_argument("--epochs", default=50, type=int, help="TENT adapts fast")
    parser.add_argument("--save_interval", default=10, type=int)
    parser.add_argument("--eval_interval", default=5, type=int)
    # Model/Data args
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--roi_x", default=96, type=int)
    parser.add_argument("--roi_y", default=96, type=int)
    parser.add_argument("--roi_z", default=96, type=int)
    parser.add_argument("--out_channels", default=87, type=int)
    parser.add_argument("--feature_size", default=48, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--resume", type=str, default=None)
    # Buffers
    parser.add_argument("--slurm_time_buffer", default=300, type=float)
    # Dummy args for compatibility
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

    # FIX: Add explicit flag to disable checkpointing to fix DDP error
    parser.add_argument("--no_swin_checkpoint", action="store_true", help="Disable Swin gradient checkpointing")
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
        "state_dict": model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)


def load_weights_precise(model, checkpoint_path, rank):
    if rank == 0:
        print(f"📦 Loading weights from {checkpoint_path} ...")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        new_state = {}
        for k, v in state_dict.items():
            # Classifier Head Mapping
            if "out.conv.weight" in k:
                new_state["backbone.out.conv.conv.weight"] = v
                continue
            if "out.conv.bias" in k:
                new_state["backbone.out.conv.conv.bias"] = v
                continue

            # Standard Mapping
            if k.startswith("backbone."):
                new_state[k] = v
            else:
                new_state[f"backbone.{k}"] = v

        msg = model.load_state_dict(new_state, strict=False)
        if rank == 0:
            chk_key = "backbone.out.conv.conv.weight"
            if chk_key in new_state:
                print("✅ Classifier Head Verification PASSED.")
            else:
                print(f"❌ Classifier Head Verification FAILED!")
    except Exception as e:
        if rank == 0:
            print(f"❌ Error loading checkpoint: {e}")
        raise e


def validate(model, loader, device, args):
    """
    Validation with Oracle Mask (derived from Label).
    """
    model.eval()

    metrics_sum = {"dice": 0.0}
    total_steps = 0

    with torch.no_grad():
        for batch in loader:
            val_images = batch["image"].to(device)
            val_labels = batch["label"].to(device)  # Labels: -1 (bg), 0..86 (fg)

            # --- ORACLE MASK GENERATION ---
            # Use Ground Truth Label to define the ROI
            if args.foreground_only:
                # Background is -1, Foreground is >= 0
                brain_mask = (val_labels >= 0).float()
            else:
                # Background is 0, Foreground is > 0
                brain_mask = (val_labels > 0).float()

            # Inference
            val_logits = sliding_window_inference(
                val_images, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.25
            )

            # Predictions: 0..86 (All foreground assumption by model)
            val_preds = torch.argmax(val_logits, dim=1, keepdim=True)

            # Shift indices for metric calculation (0=Bg, 1..87=Fg)
            target_shifted = val_labels + 1
            pred_shifted = val_preds + 1

            # --- APPLY ORACLE MASK ---
            # Force any prediction outside the GT brain mask to be 0 (Background)
            # This ensures Dice reflects performance strictly within the brain ROI
            pred_shifted = pred_shifted * brain_mask.long()

            n_classes_expanded = args.out_channels + 1  # 88 classes

            pred_oh = F.one_hot(pred_shifted.squeeze(1).long(), num_classes=n_classes_expanded).permute(0, 4, 1, 2, 3)
            target_oh = F.one_hot(target_shifted.squeeze(1).long(), num_classes=n_classes_expanded).permute(0, 4, 1, 2, 3)

            # Compute Dice (Ignore index 0/Background)
            pred_fg = pred_oh[:, 1:, ...]
            target_fg = target_oh[:, 1:, ...]

            dims = (2, 3, 4)
            intersection = (pred_fg * target_fg).sum(dim=dims)
            cardinality = pred_fg.sum(dim=dims) + target_fg.sum(dim=dims)
            dice_batch = (2.0 * intersection + 1e-5) / (cardinality + 1e-5)
            metrics_sum["dice"] += dice_batch.mean().item()

            total_steps += 1

    final_metrics = {}
    if dist.is_initialized():
        total_steps_tensor = torch.tensor(total_steps, device=device)
        dist.all_reduce(total_steps_tensor, op=dist.ReduceOp.SUM)
        total_steps_global = total_steps_tensor.item()

        for k, v in metrics_sum.items():
            v_tensor = torch.tensor(v, device=device)
            dist.all_reduce(v_tensor, op=dist.ReduceOp.SUM)
            final_metrics[k] = (v_tensor / max(total_steps_global, 1)).item()
    else:
        total_steps_global = max(total_steps, 1)
        for k, v in metrics_sum.items():
            final_metrics[k] = v / total_steps_global

    return final_metrics


def main():
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    is_distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_target_dataloaders(args, is_distributed=is_distributed, world_size=world_size, rank=rank)

    use_checkpoint = not args.no_swin_checkpoint
    backbone = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=1,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=use_checkpoint,
    ).to(device)

    model = SimplifiedDAUnetModule(backbone, num_classes=args.out_channels).to(device)

    if args.pretrained_checkpoint:
        load_weights_precise(model, args.pretrained_checkpoint, rank)

    tent_params, param_names = configure_model_for_tent(model)
    if rank == 0:
        print(f"TENT Active: Updating {len(tent_params)} affine parameters.")

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, static_graph=True
        )

    optimizer = torch.optim.AdamW(tent_params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 1
    best_dice = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt["best_dice"]
        model.module.load_state_dict(ckpt["state_dict"]) if is_distributed else model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if rank == 0:
            print(f"Resumed from epoch {start_epoch-1}")

    job_deadline = None
    if "SLURM_JOB_END_TIME" in os.environ:
        job_deadline = float(os.environ["SLURM_JOB_END_TIME"]) - args.slurm_time_buffer

    writer = SummaryWriter(log_dir=os.path.join(args.results_dir, "logs")) if rank == 0 else None

    for epoch in range(start_epoch, args.epochs + 1):
        if job_deadline and time.time() > job_deadline:
            if rank == 0:
                print("Time limit reached. Saving and exiting for requeue.")
            break

        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        steps = 0

        for batch in train_loader:
            img = batch["image"].to(device)
            labels = batch["label"].to(device)  # Read labels for mask

            # --- ORACLE MASK GENERATION (Training) ---
            if args.foreground_only:
                mask = (labels >= 0).float()  # -1 is BG, >=0 is FG
            else:
                mask = (labels > 0).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(img)
                ent = softmax_entropy(logits)  # (B, H, W, D)

                # Apply Mask: Only minimize entropy where mask == 1
                if mask.shape[1] == 1:
                    mask_s = mask.squeeze(1)
                else:
                    mask_s = mask

                # Avoid div by zero if mask is empty
                loss = (ent * mask_s).sum() / (mask_s.sum() + 1e-6)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / steps if steps > 0 else 0
        if rank == 0:
            print(f"Epoch {epoch}: Masked Entropy Loss = {avg_loss:.6f}")
            if writer:
                writer.add_scalar("train/entropy", avg_loss, epoch)

            if epoch % args.eval_interval == 0:
                val_results = validate(model, val_loader, device, args)
                dice = val_results["dice"]
                print(f"Epoch {epoch} Val Dice: {dice:.4f}")
                if writer:
                    writer.add_scalar("val/dice", dice, epoch)

                if dice > best_dice:
                    best_dice = dice
                    save_checkpoint(os.path.join(args.results_dir, "best_model.pt"), model, optimizer, epoch, best_dice)

            save_checkpoint(os.path.join(args.results_dir, "latest_model.pt"), model, optimizer, epoch, best_dice)

    if is_main_process(args) and (not job_deadline or time.time() < job_deadline):
        save_checkpoint(os.path.join(args.results_dir, "final_model.pt"), model, optimizer, args.epochs, best_dice)

    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args):
    return (not dist.is_initialized()) or dist.get_rank() == 0


if __name__ == "__main__":
    main()
