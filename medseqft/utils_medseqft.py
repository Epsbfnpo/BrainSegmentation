import os
import signal
import time
from typing import Tuple

import torch


def robust_one_hot(
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Foreground-only one-hot conversion with safe masking.

    Args:
        target: [B, 1, D, H, W] tensor with values in {-1, 0, 1, ...}
        num_classes: number of classes
        ignore_index: background index to ignore (default: -1)

    Returns:
        one_hot: [B, num_classes, D, H, W]
        valid_mask: [B, 1, D, H, W]
    """
    shape = list(target.shape)
    shape[1] = num_classes
    one_hot = torch.zeros(shape, device=target.device, dtype=torch.float32)

    valid_mask = target != ignore_index

    safe_indices = target.clone().long()
    safe_indices[~valid_mask] = 0
    safe_indices = torch.clamp(safe_indices, min=0, max=num_classes - 1)

    one_hot.scatter_(1, safe_indices, 1.0)
    one_hot = one_hot * valid_mask.float()

    return one_hot, valid_mask


class SignalHandler:
    def __init__(self) -> None:
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self.handler)
        signal.signal(signal.SIGUSR1, self.handler)

    def handler(self, signum, frame) -> None:
        print(f"🚩 Signal {signum} received. Requesting graceful stop.")
        self.stop_requested = True


def check_slurm_deadline(buffer_seconds: int = 600) -> bool:
    end_time_str = os.environ.get("SLURM_JOB_END_TIME")
    if end_time_str:
        try:
            remaining = float(end_time_str) - time.time()
            if remaining < buffer_seconds:
                print(
                    f"⏳ Time limit approaching ({remaining:.1f}s left). Stopping gracefully...",
                    flush=True,
                )
                return True
        except ValueError:
            pass
    return False


def save_checkpoint(state, path) -> None:
    torch.save(state, path)
    print(f"💾 Checkpoint saved to {path}")
