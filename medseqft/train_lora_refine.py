import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model

from components import get_dataloaders, MedSeqFTWrapper
from utils_medseqft import SignalHandler, check_slurm_deadline, save_checkpoint, robust_one_hot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--lora_rank", default=4, type=int)

    parser.add_argument("--roi_x", default=128, type=int)
    parser.add_argument("--roi_y", default=128, type=int)
    parser.add_argument("--roi_z", default=128, type=int)
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=87, type=int)
    parser.add_argument("--feature_size", default=48, type=int)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--cache_rate", default=0.0, type=float)

    parser.add_argument("--apply_spacing", action="store_true", default=True)
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[0.8, 0.8, 0.8])
    parser.add_argument("--apply_orientation", action="store_true", default=True)
    parser.add_argument("--foreground_only", action="store_true", default=True)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    sig_handler = SignalHandler()

    print("🏗️ Loading Teacher (Stage 1 Result)...")
    teacher = MedSeqFTWrapper(args, device).to(device)
    teacher.backbone.load_state_dict(torch.load(args.stage1_checkpoint, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("🏗️ Building Student with LoRA...")
    student_base = MedSeqFTWrapper(args, device).to(device)
    student_base.load_pretrained(args.pretrained_checkpoint)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["qkv", "proj"],
        lora_dropout=0.0,
        bias="none",
    )
    student = get_peft_model(student_base.backbone, lora_config)
    student.to(device)
    student.print_trainable_parameters()

    optimizer = AdamW(student.parameters(), lr=args.lr)
    scaler = GradScaler()
    mse_loss = nn.MSELoss(reduction="none")

    train_loader, _ = get_dataloaders(args)

    start_epoch = 0
    ckpt_path = res_dir / "lora_ckpt_latest.pt"
    if ckpt_path.exists():
        print(f"🔄 Resuming LoRA training from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    print("🚀 Starting Stage 2: LoRA Refinement...")
    for epoch in range(start_epoch, args.epochs):
        student.train()
        epoch_loss = 0

        for batch in train_loader:
            if sig_handler.stop_requested or check_slurm_deadline():
                print(f"🛑 SLURM Interrupt at Epoch {epoch}. Saving & Exiting...")
                save_checkpoint(
                    {
                        "model": student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch - 1,
                    },
                    ckpt_path,
                )
                sys.exit(0)

            img = batch["image"].to(device)
            label = batch["label"].to(device)

            _, valid_mask = robust_one_hot(label, args.out_channels, ignore_index=-1)

            with torch.no_grad():
                target_feat = teacher(img)

            with autocast():
                student_feat = student(img)

                loss_map = mse_loss(student_feat, target_feat)
                loss = (loss_map * valid_mask.float()).sum() / (
                    valid_mask.float().sum() * args.out_channels + 1e-5
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        print(f"Epoch {epoch}: Refine Loss = {epoch_loss:.4f}")

        save_checkpoint(
            {
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            ckpt_path,
        )

    student.save_pretrained(str(res_dir / "lora_final"))
    print("🏁 LoRA Refinement Finished.")


if __name__ == "__main__":
    main()
