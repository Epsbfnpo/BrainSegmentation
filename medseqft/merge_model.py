import argparse

import torch
from peft import PeftModel

from components import MedSeqFTWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--output_path", required=True)

    parser.add_argument("--roi_x", default=128, type=int)
    parser.add_argument("--roi_y", default=128, type=int)
    parser.add_argument("--roi_z", default=128, type=int)
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=15, type=int)
    parser.add_argument("--feature_size", default=48, type=int)

    args = parser.parse_args()
    device = torch.device("cpu")

    print("1. Loading Base Model...")
    wrapper = MedSeqFTWrapper(args, device)
    wrapper.load_pretrained(args.base_model_path)

    print("2. Loading LoRA & Merging...")
    peft_model = PeftModel.from_pretrained(wrapper.backbone, args.lora_path)
    merged_backbone = peft_model.merge_and_unload()

    print(f"3. Saving to {args.output_path}...")
    torch.save(merged_backbone.state_dict(), args.output_path)
    print("✅ Done.")


if __name__ == "__main__":
    main()
