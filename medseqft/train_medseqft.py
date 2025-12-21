import argparse
import sys
import torch
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
    # 必须开启，以匹配预训练模型的 remapping 逻辑
    parser.add_argument("--foreground_only", action="store_true", default=True)

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(res_dir / "logs"))
    sig_handler = SignalHandler()

    # Data
    train_loader, val_loader = get_dataloaders(args)

    # Models
    print("🏗️ Building Student Model...")
    student = MedSeqFTWrapper(args, device).to(device)
    student.load_pretrained(args.pretrained_checkpoint)

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

    # Metrics
    # 🚨【关键修改】include_background=True
    # 原因：你的 Channel 0 是真实的组织（Remapped Class 1），而不是背景。
    # 背景已经在空间上通过 Mask 被剔除了，不在 Channel 维度上。
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # Resume Logic
    start_epoch = 0
    best_dice = 0.0
    ckpt_path = res_dir / "latest_checkpoint.pt"
    if ckpt_path.exists():
        print(f"🔄 Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)
        print(f"   Last Best Dice: {best_dice:.4f}")

    # Training Loop
    print(f"🚀 Starting MedSeqFT Training (KD-based FFT) for {args.epochs} epochs")

    optimizer.zero_grad()

    for epoch in range(start_epoch, args.epochs):
        student.train()
        epoch_loss = 0
        epoch_seg = 0
        epoch_kd = 0
        steps = 0

        for batch in train_loader:
            if sig_handler.stop_requested or check_slurm_deadline(buffer_seconds=600):
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
                sys.exit(0)

            img = batch["image"].to(device)
            label = batch["label"].to(device)

            with autocast():
                pred = student(img)
                with torch.no_grad():
                    teacher_pred = teacher(img)

                # Loss 计算会自动处理 Mask
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

        # --- VALIDATION LOOP (工程化修正) ---
        if (epoch + 1) % 5 == 0:
            student.eval()
            print(f"🔍 Validating at epoch {epoch}...")
            with torch.no_grad():
                for val_batch in val_loader:
                    v_img = val_batch["image"].to(device)
                    v_label = val_batch["label"].to(device)  # 这里有 -1

                    with autocast():
                        val_out = sliding_window_inference(
                            v_img, (args.roi_x, args.roi_y, args.roi_z), 4, student, overlap=0.5
                        )

                    val_pred = torch.argmax(val_out, dim=1, keepdim=True)

                    # 1. 提取 Brain Mask (从 Ground Truth 中获取，模拟实际工程中的 Mask 输入)
                    v_label_expanded, brain_mask = robust_one_hot(
                        v_label, num_classes=args.out_channels, ignore_index=-1
                    )

                    # 2. 展开预测结果
                    # val_pred 本身在背景处会有随机预测 (因为 Softmax)，但我们不关心
                    val_pred_expanded, _ = robust_one_hot(
                        val_pred, num_classes=args.out_channels, ignore_index=-1
                    )

                    # 3. 【关键步骤】应用 Mask 到预测结果
                    # 强制将背景区域的预测清零。
                    # 如果不做这一步，背景的随机噪声会降低 Dice，导致分数虚低。
                    val_pred_expanded = val_pred_expanded * brain_mask

                    # 4. 计算指标
                    # 此时 prediction 和 label 在背景处都是全 0
                    dice_metric(y_pred=val_pred_expanded, y=v_label_expanded)

                mean_dice = dice_metric.aggregate().item()
                dice_metric.reset()

                print(f"Epoch {epoch}: Train Loss={epoch_loss:.4f} | Val Dice={mean_dice:.4f}")
                writer.add_scalar("val/dice", mean_dice, epoch)

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    print(f"🌟 New Best Dice: {best_dice:.4f}")
                    torch.save(student.state_dict(), res_dir / "best_model.pt")
                    state = {
                        'model': student.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'best_dice': best_dice
                    }
                    torch.save(state, ckpt_path)

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

    print("🏎️ Training Finished.")


if __name__ == "__main__":
    main()
