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

# Local imports
from age_aware_modules import SimplifiedDAUnetModule
from data_loader_age_aware import get_target_dataloaders
from tent_core import configure_model_for_tent, softmax_entropy


def parse_args():
    parser = argparse.ArgumentParser(description="TENT Adaptation")
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--results_dir", default="./results_tent")
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--save_interval", default=10, type=int)
    parser.add_argument("--eval_interval", default=5, type=int)
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
    # Disable Swin gradient checkpointing (fixes DDP + checkpointing conflict)
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


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


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


def load_weights_precise(model, checkpoint_path, rank):
    """Load source weights with classifier head remapping.

    The pretrained checkpoint stores the output head as ``out.conv.weight`` / ``out.conv.bias``
    but the current model expects ``backbone.out.conv.conv.*`` because of the extra ``Conv``
    wrapper. Without this remap the classifier stays randomly initialised and TENT cannot
    adapt. All other keys are prefixed with ``backbone.`` when missing as before.
    """
    if rank == 0:
        print(f"📦 Loading weights from {checkpoint_path} ...")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_state = {}
    for key, val in state_dict.items():
        # Remap classifier head names
        if key.endswith("out.conv.weight"):
            new_state["backbone.out.conv.conv.weight"] = val
            continue
        if key.endswith("out.conv.bias"):
            new_state["backbone.out.conv.conv.bias"] = val
            continue

        # Standard backbone prefixing
        if key.startswith("backbone."):
            new_state[key] = val
        else:
            new_state[f"backbone.{key}"] = val

    msg = model.load_state_dict(new_state, strict=False)
    if rank == 0:
        if msg.missing_keys:
            print(f"⚠️  Missing keys ({len(msg.missing_keys)}): {msg.missing_keys[:5]} ...")
        if msg.unexpected_keys:
            print(f"⚠️  Unexpected keys ({len(msg.unexpected_keys)}): {msg.unexpected_keys[:5]} ...")

        # Verify backbone and classifier head mapping
        probe_key = "backbone.swinViT.patch_embed.proj.weight"
        head_key = "backbone.out.conv.conv.weight"
        if probe_key in new_state:
            model_val = model.state_dict()[probe_key].flatten()[0].item()
            ckpt_val = new_state[probe_key].flatten()[0].item()
            if abs(model_val - ckpt_val) < 1e-5:
                print("✅ Weight Verification PASSED: Backbone weights loaded correctly.")
            else:
                print(f"❌ Weight Verification FAILED: Model: {model_val}, Ckpt: {ckpt_val}")
        else:
            print("⚠️  Probe key not found for verification.")

        if head_key in new_state:
            head_model_val = model.state_dict()[head_key].flatten()[0].item()
            head_ckpt_val = new_state[head_key].flatten()[0].item()
            if abs(head_model_val - head_ckpt_val) < 1e-5:
                print("✅ Classifier head loaded correctly.")
            else:
                print("❌ Classifier head mismatch after loading.")
        else:
            print("⚠️  Classifier head key missing from remapped checkpoint; head may be random.")


def validate(model, loader, device, args):
    """Run validation with foreground-only labels.

    The dataloader can emit labels in [-1, 86] when ``foreground_only`` is set,
    which breaks ``F.one_hot`` on CUDA. Shift both labels and predictions by +1
    so background maps to 0 and classes map to 1..87, then ignore background via
    ``include_background=False``.
    """

    model.eval()
    num_classes_expanded = args.out_channels + 1  # +1 for shifted background
    # Local accumulators
    local_dice_sum = torch.tensor(0.0, device=device)
    local_steps = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch in loader:
            val_images = batch["image"].to(device)
            val_labels = batch["label"].to(device)  # labels may contain -1 for background

            val_outputs = sliding_window_inference(
                val_images, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.25
            )

            # Predictions are 0..86; shift to 1..87
            val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True) + 1
            # Labels are -1..86; shift to 0..87
            val_labels = val_labels + 1

            val_outputs_onehot = F.one_hot(
                val_outputs.squeeze(1).long(), num_classes=num_classes_expanded
            ).permute(0, 4, 1, 2, 3)

            val_labels_onehot = F.one_hot(
                val_labels.squeeze(1).long(), num_classes=num_classes_expanded
            ).permute(0, 4, 1, 2, 3)

            # Remove background channel (index 0)
            pred_fg = val_outputs_onehot[:, 1:, ...]
            label_fg = val_labels_onehot[:, 1:, ...]

            dims = (2, 3, 4)
            intersection = (pred_fg * label_fg).sum(dim=dims)
            cardinality = pred_fg.sum(dim=dims) + label_fg.sum(dim=dims)
            dice_batch = (2.0 * intersection + 1e-5) / (cardinality + 1e-5)
            dice_score = dice_batch.mean()

            local_dice_sum += dice_score
            local_steps += 1

    if dist.is_initialized():
        dist.all_reduce(local_dice_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_steps, op=dist.ReduceOp.SUM)

    if local_steps.item() == 0:
        return 0.0

    return (local_dice_sum / local_steps).item()


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

    use_checkpoint = args.use_swin_checkpoint and not args.no_swin_checkpoint
    if rank == 0:
        print(f"Building SwinUNETR (use_checkpoint={use_checkpoint})...")

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
        print(f"TENT Active: Updating {len(tent_params)} affine parameters in Norm layers.")
        if param_names:
            print(" - " + "\n - ".join(param_names))

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            static_graph=True,
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
            if rank == 0:
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

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            dice = validate(model, val_loader, device, args)
            if rank == 0:
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

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
