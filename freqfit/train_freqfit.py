#!/usr/bin/env python3
import argparse
import os
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
import torch.nn.functional as F

from data_loader import get_target_dataloaders
from modules import SwinUNETRWrapper, load_pretrained_weights
from freqfit_core import inject_freqfit_and_lora


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--roi_x", type=int, default=128)
    parser.add_argument("--roi_y", type=int, default=128)
    parser.add_argument("--roi_z", type=int, default=128)
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[0.8, 0.8, 0.8])
    parser.add_argument("--out_channels", type=int, default=87)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--feature_size", type=int, default=48)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_rate", type=float, default=0.0)
    parser.add_argument("--cache_num_workers", type=int, default=4)
    parser.add_argument("--no_swin_checkpoint", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--apply_spacing", action="store_true")
    parser.add_argument("--apply_orientation", action="store_true")
    parser.add_argument("--foreground_only", action="store_true")
    return parser.parse_args()


def train(args, model, loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)

    for batch in loader:
        img = batch["image"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(img)
            loss_c = ce_loss(logits, label.squeeze(1).long())

            l_dice = label.clone()
            l_dice[l_dice < 0] = 0
            if l_dice.ndim == 4: l_dice = l_dice.unsqueeze(1)
            loss_d = dice_loss(logits, l_dice)

            loss = loss_c + loss_d

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device, args):
    model.eval()
    # 在 CPU 上计算 Metric，避免 OOM
    metric = DiceMetric(include_background=False, reduction="mean")

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            label = batch["label"]  # 标签先留在 CPU，或者显式移回 CPU

            # 1. 使用滑动窗口推理
            # 注意：sw_batch_size=4 可以提高速度，显存不够就降为 1
            logits = sliding_window_inference(
                img,
                (args.roi_x, args.roi_y, args.roi_z),
                sw_batch_size=1,
                predictor=model,
                overlap=0.25
            )

            # 2. 关键步骤：在 GPU 上直接 Argmax，大幅降低显存
            # (B, 87, H, W, D) -> (B, 1, H, W, D)
            val_pred_idx = torch.argmax(logits, dim=1, keepdim=True)

            # 3. 立即释放 Logits 大张量
            del logits
            torch.cuda.empty_cache()

            # 4. 移至 CPU
            val_pred_idx = val_pred_idx.cpu()
            val_label = label.long().cpu()

            # 5. 处理 Label 背景 (-1 -> 0)
            val_label[val_label < 0] = 0

            # 6. 在 CPU 上转 One-Hot
            # F.one_hot 输出 (B, H, W, D, C)，需要 permute 回 (B, C, H, W, D)
            val_pred_onehot = F.one_hot(val_pred_idx.squeeze(1), num_classes=args.out_channels).permute(0, 4, 1, 2, 3)
            val_label_onehot = F.one_hot(val_label.squeeze(1), num_classes=args.out_channels).permute(0, 4, 1, 2, 3)

            # 7. 计算 Metric
            metric(y_pred=val_pred_onehot, y=val_label_onehot)

    return metric.aggregate().item()


def main():
    args = parse_args()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
        is_main = args.rank == 0
    else:
        print("⚠️  Running in non-distributed mode (Single GPU)")
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    if is_main:
        print(f"🚀 Initialized. World Size: {args.world_size}, Rank: {args.rank}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, val_loader = get_target_dataloaders(
        args,
        is_distributed=dist.is_initialized(),
        rank=args.rank,
        world_size=args.world_size
    )

    # 1. Build Model
    model = SwinUNETRWrapper(args).to(device)
    load_pretrained_weights(model, args.pretrained_checkpoint)

    # 2. Inject FreqFiT + LoRA
    target_modules = ["attn.qkv", "attn.proj", "mlp.linear1", "mlp.linear2"]
    model = inject_freqfit_and_lora(model, target_modules, rank=args.lora_rank).to(device)

    params_to_train = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        print(f"🔥 Trainable parameters: {sum(p.numel() for p in params_to_train)}")

    if dist.is_initialized():
        # 🔥 关键修复：static_graph=True 解决 Checkpointing 导致的 "ready twice" 报错
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
            static_graph=True
        )

    optimizer = torch.optim.AdamW(params_to_train, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_dice = 0.0

    # Auto-Resume
    if args.resume and os.path.isfile(args.resume):
        if is_main: print(f"🔄 Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        target_model = model.module if dist.is_initialized() else model
        target_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_val_dice', 0.0)
        if is_main: print(f"   -> Resumed at Epoch {start_epoch}, Best Dice: {best_dice:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        if dist.is_initialized():
            train_loader.sampler.set_epoch(epoch)

        loss = train(args, model, train_loader, optimizer, scaler, device, epoch)

        if epoch % 5 == 0:
            dice = validate(model, val_loader, device, args)
            if is_main:
                print(f"Epoch {epoch}: Train Loss {loss:.4f}, Val Dice {dice:.4f}")
                if dice > best_dice:
                    best_dice = dice
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if dist.is_initialized() else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_dice': best_dice
                    }, os.path.join(args.results_dir, "best_model.pt"))

        scheduler.step()

        if is_main:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if dist.is_initialized() else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_dice': best_dice
            }, os.path.join(args.results_dir, "latest_model.pt"))

    if is_main:
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.module.state_dict() if dist.is_initialized() else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_dice': best_dice
        }, os.path.join(args.results_dir, "final_model.pt"))


if __name__ == "__main__":
    main()