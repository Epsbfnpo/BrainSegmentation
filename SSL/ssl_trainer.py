"""
Training functions for SSL pretraining with AMP support and non-blocking transfers
DISTRIBUTED VERSION with multi-GPU synchronization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
import time
from typing import Dict, Optional
import gc
from monai.inferers import sliding_window_inference
from torch.cuda.amp import autocast, GradScaler


def is_dist():
    """Check if distributed training is initialized"""
    return dist.is_initialized()


def dist_mean_scalar(x: torch.Tensor):
    """Average a scalar across all processes"""
    if is_dist():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x


def train_epoch_ssl(
        model: nn.Module,
        ssl_loss: nn.Module,
        train_loader,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        epoch: int,
        device: torch.device,
        writer,
        args
) -> Dict[str, float]:
    """Train one epoch of SSL with AMP support and optimized transfers (single GPU version)"""

    model.train()
    ssl_loss.train()

    # Metrics tracking
    metrics = {
        'total_loss': 0.0,
        'inpainting_loss': 0.0,
        'rotation_loss': 0.0,
        'contrastive_loss': 0.0,
        'grad_norm': 0.0,
        'lr': optimizer.param_groups[0]['lr']
    }

    num_steps = len(train_loader)

    print(f"\n🚀 Training - Epoch {epoch}")
    print(f"  Learning rate: {metrics['lr']:.6f}")
    print(f"  AMP: Enabled (BF16)")

    start_time = time.time()
    batch_start_time = time.time()

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for i, batch in enumerate(pbar):
        # Non-blocking transfer to GPU
        images = batch['image'].to(device, non_blocking=True)

        if torch.isnan(images).any():
            print(f"\n⚠️  WARNING: NaN in input at step {i}")
            continue

        optimizer.zero_grad()

        # Forward pass with autocast for BF16
        with autocast(dtype=torch.bfloat16):
            loss, loss_dict = ssl_loss(model, images)

        if torch.isnan(loss):
            print(f"\n⚠️  WARNING: NaN loss at step {i}")
            print(f"  Loss components: {loss_dict}")
            continue

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Unscale gradients for clipping
        scaler.unscale_(optimizer)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(ssl_loss.parameters()),
            args.clip
        )

        # Update weights with scaler
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        metrics['total_loss'] += loss.item()
        metrics['inpainting_loss'] += loss_dict['inpainting'].item()
        metrics['rotation_loss'] += loss_dict['rotation'].item()
        metrics['contrastive_loss'] += loss_dict['contrastive'].item()
        metrics['grad_norm'] += grad_norm.item()

        # Calculate batch time
        batch_time = time.time() - batch_start_time
        batch_start_time = time.time()

        # Update progress bar with timing info
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'inp': f"{loss_dict['inpainting'].item():.4f}",
            'rot': f"{loss_dict['rotation'].item():.4f}",
            'con': f"{loss_dict['contrastive'].item():.4f}",
            'scale': f"{scaler.get_scale():.0f}",
            'time': f"{batch_time:.2f}s"
        })

        # Log to tensorboard
        if i % 10 == 0:
            global_step = (epoch - 1) * num_steps + i
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/inpainting_loss', loss_dict['inpainting'].item(), global_step)
            writer.add_scalar('train/rotation_loss', loss_dict['rotation'].item(), global_step)
            writer.add_scalar('train/contrastive_loss', loss_dict['contrastive'].item(), global_step)
            writer.add_scalar('train/grad_norm', grad_norm.item(), global_step)
            writer.add_scalar('train/lr', metrics['lr'], global_step)
            writer.add_scalar('train/amp_scale', scaler.get_scale(), global_step)
            writer.add_scalar('train/batch_time', batch_time, global_step)

    # Average metrics
    for key in metrics:
        if key != 'lr':
            metrics[key] /= num_steps

    elapsed = time.time() - start_time
    avg_batch_time = elapsed / num_steps

    print(f"\n✓ Epoch {epoch} completed in {elapsed:.1f}s")
    print(f"  Average batch time: {avg_batch_time:.2f}s")
    print(f"  Throughput: {len(train_loader.dataset) / elapsed:.1f} samples/sec")
    print(f"  Average loss: {metrics['total_loss']:.4f}")
    print(f"  Inpainting: {metrics['inpainting_loss']:.4f}")
    print(f"  Rotation: {metrics['rotation_loss']:.4f}")
    print(f"  Contrastive: {metrics['contrastive_loss']:.4f}")
    print(f"  Gradient norm: {metrics['grad_norm']:.4f}")
    print(f"  AMP scale: {scaler.get_scale():.0f}")

    return metrics


def train_epoch_ssl_distributed(
        model: nn.Module,
        ssl_loss: nn.Module,
        train_loader,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        epoch: int,
        device: torch.device,
        writer,
        args,
        is_distributed: bool = False,
        world_size: int = 1,
        rank: int = 0
) -> Dict[str, float]:
    """Train one epoch of SSL with distributed training support"""

    is_main = (not is_distributed) or rank == 0

    model.train()
    ssl_loss.train()

    # Metrics tracking
    metrics = {
        'total_loss': 0.0,
        'inpainting_loss': 0.0,
        'rotation_loss': 0.0,
        'contrastive_loss': 0.0,
        'grad_norm': 0.0,
        'lr': optimizer.param_groups[0]['lr']
    }

    num_steps = len(train_loader)

    if is_main:
        print(f"\n🚀 Training - Epoch {epoch}")
        print(f"  Learning rate: {metrics['lr']:.6f}")
        print(f"  AMP: Enabled ({args.amp_dtype})")
        if world_size > 1:
            print(f"  World size: {world_size} GPUs")

    start_time = time.time()
    batch_start_time = time.time()

    # Progress bar (only on main process)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}") if is_main else train_loader

    # Set AMP dtype
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16

    for i, batch in enumerate(pbar):
        # Non-blocking transfer to GPU
        images = batch['image'].to(device, non_blocking=True)

        if torch.isnan(images).any():
            if is_main:
                print(f"\n⚠️  WARNING: NaN in input at step {i}")
            continue

        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast(dtype=amp_dtype):
            # Handle DDP wrapped models
            if is_distributed:
                loss, loss_dict = ssl_loss.module(model.module, images)
            else:
                loss, loss_dict = ssl_loss(model, images)

        if torch.isnan(loss):
            if is_main:
                print(f"\n⚠️  WARNING: NaN loss at step {i}")
                print(f"  Loss components: {loss_dict}")
            continue

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Unscale gradients for clipping
        scaler.unscale_(optimizer)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(ssl_loss.parameters()),
            args.clip
        )

        # Update weights with scaler
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        metrics['total_loss'] += loss.item()
        metrics['inpainting_loss'] += loss_dict['inpainting'].item()
        metrics['rotation_loss'] += loss_dict['rotation'].item()
        metrics['contrastive_loss'] += loss_dict['contrastive'].item()
        metrics['grad_norm'] += grad_norm.item()

        # Calculate batch time
        batch_time = time.time() - batch_start_time
        batch_start_time = time.time()

        # Update progress bar with timing info (only on main process)
        if is_main and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'inp': f"{loss_dict['inpainting'].item():.4f}",
                'rot': f"{loss_dict['rotation'].item():.4f}",
                'con': f"{loss_dict['contrastive'].item():.4f}",
                'scale': f"{scaler.get_scale():.0f}",
                'time': f"{batch_time:.2f}s"
            })

        # Log to tensorboard (only on main process)
        if writer and i % 10 == 0:
            global_step = (epoch - 1) * num_steps + i
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/inpainting_loss', loss_dict['inpainting'].item(), global_step)
            writer.add_scalar('train/rotation_loss', loss_dict['rotation'].item(), global_step)
            writer.add_scalar('train/contrastive_loss', loss_dict['contrastive'].item(), global_step)
            writer.add_scalar('train/grad_norm', grad_norm.item(), global_step)
            writer.add_scalar('train/lr', metrics['lr'], global_step)
            writer.add_scalar('train/amp_scale', scaler.get_scale(), global_step)
            writer.add_scalar('train/batch_time', batch_time, global_step)

    # Average metrics
    for key in metrics:
        if key != 'lr':
            metrics[key] /= num_steps

    # Synchronize metrics across processes if distributed
    if is_distributed:
        metrics_to_sync = torch.tensor([
            metrics['total_loss'],
            metrics['inpainting_loss'],
            metrics['rotation_loss'],
            metrics['contrastive_loss'],
            metrics['grad_norm']
        ], device=device)

        dist.all_reduce(metrics_to_sync, op=dist.ReduceOp.SUM)
        metrics_to_sync /= world_size

        (metrics['total_loss'], metrics['inpainting_loss'],
         metrics['rotation_loss'], metrics['contrastive_loss'],
         metrics['grad_norm']) = metrics_to_sync.tolist()

    elapsed = time.time() - start_time
    avg_batch_time = elapsed / num_steps

    if is_main:
        print(f"\n✓ Epoch {epoch} completed in {elapsed:.1f}s")
        print(f"  Average batch time: {avg_batch_time:.2f}s")
        print(f"  Throughput: {len(train_loader.dataset) * world_size / elapsed:.1f} samples/sec")
        print(f"  Average loss: {metrics['total_loss']:.4f}")
        print(f"  Inpainting: {metrics['inpainting_loss']:.4f}")
        print(f"  Rotation: {metrics['rotation_loss']:.4f}")
        print(f"  Contrastive: {metrics['contrastive_loss']:.4f}")
        print(f"  Gradient norm: {metrics['grad_norm']:.4f}")
        print(f"  AMP scale: {scaler.get_scale():.0f}")

    return metrics


def val_epoch_ssl(
        model: nn.Module,
        ssl_loss: nn.Module,
        val_loader,
        epoch: int,
        device: torch.device,
        writer,
        args
) -> Dict[str, float]:
    """Validate one epoch of SSL with AMP support and optimized transfers (single GPU version)"""

    model.eval()
    ssl_loss.eval()

    metrics = {
        'total_loss': 0.0,
        'inpainting_loss': 0.0,
        'rotation_loss': 0.0,
        'contrastive_loss': 0.0,
    }

    num_steps = len(val_loader)

    print(f"\n📊 Validation - Epoch {epoch}")

    start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
            # Non-blocking transfer to GPU
            images = batch['image'].to(device, non_blocking=True)

            if torch.isnan(images).any():
                print(f"\n⚠️  WARNING: NaN in validation input at step {i}")
                continue

            # Use autocast for validation too
            with autocast(dtype=torch.bfloat16):
                if images.shape[2:] != (args.roi_x, args.roi_y, args.roi_z):
                    def ssl_loss_wrapper(x):
                        _, loss_dict = ssl_loss(model, x)
                        return loss_dict['total']

                    loss, loss_dict = ssl_loss(model, images)
                else:
                    loss, loss_dict = ssl_loss(model, images)

            metrics['total_loss'] += loss.item()
            metrics['inpainting_loss'] += loss_dict['inpainting'].item()
            metrics['rotation_loss'] += loss_dict['rotation'].item()
            metrics['contrastive_loss'] += loss_dict['contrastive'].item()

    for key in metrics:
        metrics[key] /= num_steps

    elapsed = time.time() - start_time
    avg_batch_time = elapsed / num_steps

    writer.add_scalar('val/loss', metrics['total_loss'], epoch)
    writer.add_scalar('val/inpainting_loss', metrics['inpainting_loss'], epoch)
    writer.add_scalar('val/rotation_loss', metrics['rotation_loss'], epoch)
    writer.add_scalar('val/contrastive_loss', metrics['contrastive_loss'], epoch)

    print(f"\n✓ Validation completed in {elapsed:.1f}s")
    print(f"  Average batch time: {avg_batch_time:.2f}s")
    print(f"  Throughput: {len(val_loader.dataset) / elapsed:.1f} samples/sec")
    print(f"  Loss: {metrics['total_loss']:.4f}")
    print(f"  Inpainting: {metrics['inpainting_loss']:.4f}")
    print(f"  Rotation: {metrics['rotation_loss']:.4f}")
    print(f"  Contrastive: {metrics['contrastive_loss']:.4f}")

    return metrics


def val_epoch_ssl_distributed(
        model: nn.Module,
        ssl_loss: nn.Module,
        val_loader,
        epoch: int,
        device: torch.device,
        writer,
        args,
        is_distributed: bool = False,
        world_size: int = 1,
        rank: int = 0
) -> Dict[str, float]:
    """Validate one epoch of SSL with distributed training support"""

    is_main = (not is_distributed) or rank == 0

    model.eval()
    ssl_loss.eval()

    metrics = {
        'total_loss': 0.0,
        'inpainting_loss': 0.0,
        'rotation_loss': 0.0,
        'contrastive_loss': 0.0,
    }

    num_steps = len(val_loader)

    if is_main:
        print(f"\n📊 Validation - Epoch {epoch}")
        if world_size > 1:
            print(f"  Distributed validation across {world_size} GPUs")

    start_time = time.time()

    # Set AMP dtype
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16

    # Progress bar (only on main process)
    pbar = tqdm(val_loader, desc="Validation") if is_main else val_loader

    with torch.no_grad():
        for i, batch in enumerate(pbar):
            # Non-blocking transfer to GPU
            images = batch['image'].to(device, non_blocking=True)

            if torch.isnan(images).any():
                if is_main:
                    print(f"\n⚠️  WARNING: NaN in validation input at step {i}")
                continue

            # Use autocast for validation too
            with autocast(dtype=amp_dtype):
                # Handle DDP wrapped models
                if is_distributed:
                    loss, loss_dict = ssl_loss.module(model.module, images)
                else:
                    loss, loss_dict = ssl_loss(model, images)

            metrics['total_loss'] += loss.item()
            metrics['inpainting_loss'] += loss_dict['inpainting'].item()
            metrics['rotation_loss'] += loss_dict['rotation'].item()
            metrics['contrastive_loss'] += loss_dict['contrastive'].item()

    # Average metrics
    for key in metrics:
        metrics[key] /= num_steps

    # Synchronize metrics across processes if distributed
    if is_distributed:
        metrics_to_sync = torch.tensor([
            metrics['total_loss'],
            metrics['inpainting_loss'],
            metrics['rotation_loss'],
            metrics['contrastive_loss']
        ], device=device)

        dist.all_reduce(metrics_to_sync, op=dist.ReduceOp.SUM)
        metrics_to_sync /= world_size

        (metrics['total_loss'], metrics['inpainting_loss'],
         metrics['rotation_loss'], metrics['contrastive_loss']) = metrics_to_sync.tolist()

    elapsed = time.time() - start_time
    avg_batch_time = elapsed / num_steps

    # Log to tensorboard (only on main process)
    if writer:
        writer.add_scalar('val/loss', metrics['total_loss'], epoch)
        writer.add_scalar('val/inpainting_loss', metrics['inpainting_loss'], epoch)
        writer.add_scalar('val/rotation_loss', metrics['rotation_loss'], epoch)
        writer.add_scalar('val/contrastive_loss', metrics['contrastive_loss'], epoch)

    if is_main:
        print(f"\n✓ Validation completed in {elapsed:.1f}s")
        print(f"  Average batch time: {avg_batch_time:.2f}s")
        print(f"  Throughput: {len(val_loader.dataset) * world_size / elapsed:.1f} samples/sec")
        print(f"  Loss: {metrics['total_loss']:.4f}")
        print(f"  Inpainting: {metrics['inpainting_loss']:.4f}")
        print(f"  Rotation: {metrics['rotation_loss']:.4f}")
        print(f"  Contrastive: {metrics['contrastive_loss']:.4f}")

    return metrics


def save_checkpoint_ssl(
        model: nn.Module,
        ssl_loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler],
        epoch: int,
        best_val_loss: float,
        args,
        filepath: str,
        monitor_history: Optional[Dict] = None
):
    """Save SSL training checkpoint with AMP state"""

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ssl_loss_state_dict': ssl_loss.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'args': args,
        'monitor_history': monitor_history
    }

    # Save scaler state if using AMP
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")