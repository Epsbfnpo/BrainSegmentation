import math
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from trm_core import TRMWeightedLoss


def is_finite(tensor: torch.Tensor) -> bool:
    return torch.isfinite(tensor).all().item()  # type: ignore[return-value]


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def train_epoch(
    target_model: torch.nn.Module,
    source_model: torch.nn.Module,
    loader,
    optimizer,
    trm_manager,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    *,
    warmup_epochs: int,
    accumulation_steps: int = 1,
) -> float:
    target_model.train()
    source_model.eval()

    loss_fn = TRMWeightedLoss(num_classes=trm_manager.num_classes).to(device)
    total_loss = 0.0
    steps = 0

    # Warmup 阶段结束后冻结统计量
    if epoch > warmup_epochs and not trm_manager.frozen:
        trm_manager.freeze_statistics()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            source_logits = source_model(images)
            # Warmup 阶段更新统计量
            if not trm_manager.frozen:
                trm_manager.update_statistics(source_logits, labels)

            # 计算 Risk Map
            risk_map = trm_manager.compute_risk_map(source_logits, labels)

        with torch.cuda.amp.autocast():
            logits = target_model(images)
            loss = loss_fn(logits, labels, risk_map)
            loss = loss / accumulation_steps

        if not is_finite(loss):
            print(f"Warning: Non-finite loss {loss.item()} at epoch {epoch}, step {steps}")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if (steps + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 还原损失方便记录
        total_loss += loss.item() * accumulation_steps
        steps += 1

    return float(total_loss / max(1, steps))


def validate_epoch(model: torch.nn.Module, loader, device: torch.device, *, num_classes: int) -> Tuple[float, Dict[str, float]]:
    model.eval()

    # 将预测转换为 one-hot 以匹配标签形状
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    total_loss = 0.0
    loss_fn = TRMWeightedLoss(num_classes=num_classes).to(device)
    steps = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images)

            # 验证时使用全 1 权重计算 Loss
            dummy_weight = torch.ones_like(labels, dtype=torch.float32)
            loss = loss_fn(logits, labels, dummy_weight)
            total_loss += loss.item()
            steps += 1

            # 预测转 one-hot
            preds_onehot = post_pred(logits)

            # 标签：-1 置零再转 one-hot
            target = labels.long()
            target[target < 0] = 0
            target_1h = post_label(target)

            dice_metric(y_pred=preds_onehot, y=target_1h)

    mean_dice = dice_metric.aggregate().item() if steps > 0 else 0.0
    dice_metric.reset()

    return float(total_loss / max(1, steps)), {"dice": mean_dice}
