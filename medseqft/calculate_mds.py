import argparse
import json
import os
import sys
from pathlib import Path

import torch

from components import get_dataloaders, MedSeqFTWrapper
from utils_medseqft import SignalHandler, check_slurm_deadline


def compute_proxy_score(model, img):
    """
    Compute MDS score using a proxy SSL loss.
    Default implementation uses prediction entropy as familiarity proxy.
    """
    logits = model(img)
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)
    return entropy.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--top_k", type=int, default=50)

    parser.add_argument("--roi_x", default=128, type=int)
    parser.add_argument("--roi_y", default=128, type=int)
    parser.add_argument("--roi_z", default=128, type=int)
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=87, type=int)
    parser.add_argument("--feature_size", default=48, type=int)

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--cache_rate", default=0.0, type=float)

    parser.add_argument("--apply_spacing", action="store_true", default=True)
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[0.8, 0.8, 0.8])
    parser.add_argument("--apply_orientation", action="store_true", default=True)
    parser.add_argument("--foreground_only", action="store_true", default=True)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_handler = SignalHandler()

    train_loader, _ = get_dataloaders(args, shuffle_train=False)

    model = MedSeqFTWrapper(args, device).to(device)
    model.load_pretrained(args.pretrained_checkpoint)
    model.eval()

    scores = []
    processed_files = set()
    temp_file = Path(args.output_json).parent / "mds_progress.json"

    if temp_file.exists():
        with open(temp_file, "r") as f:
            scores = json.load(f)
            processed_files = {item["image"] for item in scores}
        print(f"🔄 Resuming MDS: {len(scores)} samples already processed.")

    print("🚀 Starting MDS Calculation...")

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            img_path = batch["image_meta_dict"]["filename_or_obj"][0]

            if img_path in processed_files:
                continue

            if sig_handler.stop_requested or check_slurm_deadline():
                print("🛑 SLURM Interrupt. Saving progress...")
                with open(temp_file, "w") as f:
                    json.dump(scores, f)
                sys.exit(0)

            img = batch["image"].to(device)
            score = compute_proxy_score(model, img)

            label_path = batch["label_meta_dict"]["filename_or_obj"][0]
            scores.append(
                {
                    "image": img_path,
                    "label": label_path,
                    "score": score,
                }
            )

            if i % 10 == 0:
                print(f"Processed {len(scores)} samples. Last score: {score:.4f}")

    scores.sort(key=lambda x: x["score"])
    buffer_data = scores[: args.top_k]

    final_output = [{"image": x["image"], "label": x["label"]} for x in buffer_data]

    with open(args.output_json, "w") as f:
        json.dump(final_output, f, indent=2)

    if temp_file.exists():
        os.remove(temp_file)

    print(f"✅ MDS Selection Complete. Saved {len(buffer_data)} samples to {args.output_json}")


if __name__ == "__main__":
    main()
