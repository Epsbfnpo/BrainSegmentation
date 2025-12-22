import argparse
import sys
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter

# Import local components
from components import get_dataloaders, MedSeqFTWrapper, MedSeqFTLoss
from utils_medseqft import SignalHandler, check_slurm_deadline, robust_one_hot

# --- MONAI Imports ---
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--buffer_json", default=None)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--volume_stats", type=str, default=None)

    # Model params
    parser.add_argument("--roi_x", default=128, type=int)
    parser.add_argument("--roi_y", default=128, type=int)
    parser.add_argument("--roi_z", default=128, type=int)
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=87, type=int)
    parser.add_argument("--feature_size", default=48, type=int)

    # Training params
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--cache_rate", default=0.0, type=float)
    parser.add_argument("--lambda_kd", default=1.0, type=float)
    parser.add_argument("--grad_accum_steps", default=2, type=int)

    # Flags
    parser.add_argument("--apply_spacing", action="store_true", default=True)
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[0.8, 0.8, 0.8])
    parser.add_argument("--apply_orientation", action="store_true", default=True)
    parser.add_argument("--foreground_only", action="store_true", default=True)

    args = parser.parse_args()

    # --- 1. DDP 初始化 (关键修改) ---
    # torchrun 会自动设置 LOCAL_RANK 等环境变量
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_main_process = (global_rank == 0)
    else:
        # 兼容非 DDP 运行
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True
        local_rank = 0

    res_dir = Path(args.results_dir)
    if is_main_process:
        res_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. 仅在 Rank 0 初始化 Writer ---
    if is_main_process:
        writer = SummaryWriter(log_dir=str(res_dir / "logs"))
    else:
        writer = None

    sig_handler = SignalHandler()

    # Data
    # 注意：在理想 DDP 中这里最好用 DistributedSampler，但目前随机 shuffle 也能跑，
    # 相当于变相增大了 Batch Size (4卡 x BS2 = 有效BS 8)
    train_loader, val_loader = get_dataloaders(args)

    # Models
    if is_main_process:
        print("🏗️ Building Student Model...")
    student = MedSeqFTWrapper(args, device).to(device)
    student.load_pretrained(args.pretrained_checkpoint)

    # --- 3. DDP 包装模型 ---
    if dist.is_initialized():
        student = DDP(student, device_ids=[local_rank], output_device=local_rank)

    if is_main_process:
        print("🏗️ Building Teacher Model (Frozen Source)...")
    teacher = MedSeqFTWrapper(args, device).to(device)
    teacher.load_pretrained(args.pretrained_checkpoint)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    # Optimization
    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    loss_fn = MedSeqFTLoss(num_classes=args.out_channels, lambda_kd=args.lambda_kd)

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # Resume Logic
    start_epoch = 0
    best_dice = 0.0
    ckpt_path = res_dir / "latest_checkpoint.pt"

    # 仅 Rank 0 读取和广播 Checkpoint (或者让大家都去读)
    # 简单起见，这里大家都尝试读，只要文件存在不冲突即可 (读操作是安全的)
    if ckpt_path.exists():
        if is_main_process:
            print(f"🔄 Resuming from {ckpt_path}")
        # map_location 必须指定到当前 GPU
        ckpt = torch.load(ckpt_path, map_location=device)

        # 处理 DDP 带来的 module. 前缀问题 (如果之前保存的是 DDP state dict)
        # 你的 MedSeqFTWrapper.load_pretrained 已经处理了 module. 前缀，但这里是直接 load_state_dict
        # 我们需要简单清洗一下
        model_state = ckpt['model']
        # 如果当前是 DDP，但 checkpoint 不是 (或者反之)，key 会对不上
        # 最稳妥的方式：直接加载。因为保存时通常建议 student.module.state_dict()
        # 这里假设保存的是 student.state_dict() (即包含了 module. 前缀)
        try:
            student.load_state_dict(model_state)
        except RuntimeError:
            # 尝试去掉 module. 前缀 (如果 ckpt 有而 model 没有)
            new_state = {k.replace('module.', ''): v for k, v in model_state.items()}
            # 如果 model 有 module. (DDP) 而 ckpt 没有，加上
            if isinstance(student, DDP):
                # 这种情况下通常直接 load 就行，因为 DDP 包装后 key 变了
                # 这里做个简单的 fallback
                student.module.load_state_dict(new_state)
            else:
                student.load_state_dict(new_state)

        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)
        if is_main_process:
            print(f"   Last Best Dice: {best_dice:.4f}")

    if is_main_process:
        print(f"🚀 Starting MedSeqFT Training (KD-based FFT) for {args.epochs} epochs")

    optimizer.zero_grad()

    for epoch in range(start_epoch, args.epochs):
        # 如果使用了 DistributedSampler，这里需要 train_loader.sampler.set_epoch(epoch)

        student.train()
        epoch_loss = 0
        epoch_seg = 0
        epoch_kd = 0
        steps = 0

        for batch in train_loader:
            if sig_handler.stop_requested or check_slurm_deadline(buffer_seconds=600):
                if is_main_process:
                    print(f"🛑 检测到退出信号或超时 (Epoch {epoch})，保存断点并退出...")
                    state = {
                        'model': student.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch - 1,
                        'best_dice': best_dice
                    }
                    torch.save(state, ckpt_path)
                    print("👋 优雅退出 (Exit 0)")
                # 确保所有进程同步退出
                if dist.is_initialized():
                    dist.barrier()
                sys.exit(0)

            if not batch:
                if is_main_process:
                    print(f"⚠️ Warning: Skipped empty batch at epoch {epoch}")
                continue
                # ==============================

            img = batch["image"].to(device)
            label = batch["label"].to(device)

            with autocast():
                pred = student(img)
                with torch.no_grad():
                    teacher_pred = teacher(img)

                if teacher_pred.shape[1] != pred.shape[1]:
                    min_ch = min(teacher_pred.shape[1], pred.shape[1])
                    teacher_pred_safe = teacher_pred[:, :min_ch, ...]
                    pred_safe_for_kd = pred[:, :min_ch, ...]
                    total_loss, l_seg, l_kd = loss_fn(pred, label, teacher_pred_safe, pred_kd=pred_safe_for_kd)
                else:
                    total_loss, l_seg, l_kd = loss_fn(pred, label, teacher_pred)

                loss = total_loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (steps + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 12.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 简单的 Loss 聚合用于打印 (只在 Rank 0 打印)
            if dist.is_initialized():
                dist.all_reduce(total_loss)
                total_loss /= dist.get_world_size()  # 平均 Loss

            epoch_loss += total_loss.item()
            epoch_seg += l_seg.item()
            epoch_kd += l_kd.item()
            steps += 1

        if steps % args.grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()

        # --- VALIDATION LOOP ---
        if (epoch + 1) % 5 == 0:
            student.eval()
            if is_main_process:
                print(f"🔍 Validating at epoch {epoch}...")

            # 创建一个用于 Metric 聚合的 List
            val_dice_list = []

            with torch.no_grad():
                for val_batch in val_loader:
                    # 验证阶段的信号检查
                    if sig_handler.stop_requested or check_slurm_deadline(buffer_seconds=600):
                        if is_main_process:
                            print(f"🛑 验证阶段退出 (Epoch {epoch})...")
                            state = {
                                'model': student.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'epoch': epoch,
                                'best_dice': best_dice
                            }
                            torch.save(state, ckpt_path)
                        if dist.is_initialized():
                            dist.barrier()
                        sys.exit(0)

                    v_img = val_batch["image"].to(device)
                    v_label = val_batch["label"].to(device)

                    with autocast():
                        val_out = sliding_window_inference(
                            v_img, (args.roi_x, args.roi_y, args.roi_z), 4, student, overlap=0.5
                        )

                    val_pred = torch.argmax(val_out, dim=1, keepdim=True)
                    v_label_expanded, brain_mask = robust_one_hot(
                        v_label, num_classes=args.out_channels, ignore_index=-1
                    )
                    val_pred_expanded, _ = robust_one_hot(
                        val_pred, num_classes=args.out_channels, ignore_index=-1
                    )
                    val_pred_expanded = val_pred_expanded * brain_mask

                    # 计算当前 Batch 的 Dice
                    dice_metric(y_pred=val_pred_expanded, y=v_label_expanded)

                # 聚合本地 Dice
                local_dice = dice_metric.aggregate().item()
                dice_metric.reset()

                # DDP: 聚合所有卡的 Dice
                if dist.is_initialized():
                    metric_tensor = torch.tensor(local_dice).to(device)
                    dist.all_reduce(metric_tensor)
                    mean_dice = metric_tensor.item() / dist.get_world_size()
                else:
                    mean_dice = local_dice

                if is_main_process:
                    print(f"Epoch {epoch}: Train Loss={epoch_loss:.4f} | Val Dice={mean_dice:.4f}")
                    if writer:
                        writer.add_scalar("val/dice", mean_dice, epoch)

                    if mean_dice > best_dice:
                        best_dice = mean_dice
                        print(f"🌟 New Best Dice: {best_dice:.4f}")
                        # 保存最佳模型 (注意: DDP时 student.state_dict() 包含 module.)
                        torch.save(student.state_dict(), res_dir / "best_model.pt")
                        state = {
                            'model': student.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch,
                            'best_dice': best_dice
                        }
                        torch.save(state, ckpt_path)

        # 仅在 Rank 0 记录训练日志和保存 Checkpoint
        if is_main_process:
            if writer:
                writer.add_scalar("train/loss", epoch_loss, epoch)
                writer.add_scalar("train/seg_loss", epoch_seg, epoch)
                writer.add_scalar("train/kd_loss", epoch_kd, epoch)
                writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)

            state = {
                'model': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_dice': best_dice
            }
            torch.save(state, ckpt_path)

            if epoch == args.epochs - 1:
                torch.save(student.state_dict(), res_dir / "final_model.pt")

    if is_main_process:
        print("🏎️ Training Finished.")

    # 销毁进程组
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()