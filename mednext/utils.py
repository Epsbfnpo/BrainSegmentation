import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
import contextlib

# --- 防御性导入 Hausdorff 函数式 API ---
try:
    from monai.metrics import compute_hausdorff_distance
except ImportError:  # pragma: no cover - compatibility path
    try:
        from monai.metrics.hausdorff_distance import compute_hausdorff_distance
    except ImportError:  # pragma: no cover - last resort fallback
        print("⚠️ Warning: compute_hausdorff_distance not found; returning zeros.")

        def compute_hausdorff_distance(*args, **kwargs):
            return torch.tensor([0.0])


# --- DIY Dice 计算 ---
def compute_dice_score(y_pred, y, smooth=1e-5):
    """
    y_pred, y: One-Hot Tensors (B, C, D, H, W)
    Returns: (B, C) dice scores
    """

    dims = tuple(range(2, y_pred.ndim))
    intersect = (y_pred * y).sum(dim=dims)
    union = y_pred.sum(dim=dims) + y.sum(dim=dims)
    dice = (2.0 * intersect + smooth) / (union + smooth)
    return dice


# --- NaN 检测 ---
def check_finite(name, tensor):
    if tensor is None:
        return True
    if torch.isfinite(tensor).all():
        return True
    print(f"[NaN DETECTED] {name} has NaN/Inf", flush=True)
    return False


# --- EMA ---
class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# --- 组合 Loss，自动适配 Deep Supervision 输出 ---
class CombinedLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True, include_background=False, squared_pred=True)
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, logits, labels):
        if labels.ndim == 4:
            labels = labels.unsqueeze(1)

        # Deep supervision: 对 label 做最近邻下采样以匹配 logits 尺寸
        if logits.shape[2:] != labels.shape[2:]:
            labels = F.interpolate(labels.float(), size=logits.shape[2:], mode="nearest")

        if labels.ndim == 5:
            labels = labels.squeeze(1)

        labels_rect = labels.clone()
        labels_rect[labels_rect < 0] = 0

        loss_ce = self.ce(logits, labels.long())
        loss_dice = self.dice(logits, labels_rect.unsqueeze(1))
        return 0.5 * loss_ce + 0.5 * loss_dice


# --- 训练 Loop ---
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, ema=None):
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        img, lbl = batch["image"].to(device), batch["label"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(img)
            if isinstance(logits, (list, tuple)):
                loss = sum(loss_fn(l, lbl) for l in logits) / len(logits)
            else:
                loss = loss_fn(logits, lbl)

        if not check_finite("loss", loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        scaler.step(optimizer)
        scaler.update()

        if ema:
            ema.update()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


# --- 验证 Loop：纯函数式计算，避免 Metric 类同步问题 ---
def validate(model, loader, device, roi_size):
    model.eval()

    total_dice = 0.0
    total_hd95 = 0.0
    count_dice = 0
    count_hd95 = 0

    def _predictor(x):
        out = model(x)
        if isinstance(out, (list, tuple)):
            return out[-1]
        return out

    with torch.no_grad():
        for batch in loader:
            img, lbl = batch["image"].to(device), batch["label"].to(device)
            lbl = lbl.long()
            lbl[lbl < 0] = 0

            logits = sliding_window_inference(img, roi_size, 1, _predictor, overlap=0.25, mode="gaussian")
            if isinstance(logits, (list, tuple)):
                logits = logits[-1]

            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)
            n_class = logits.shape[1]

            pred_oh = F.one_hot(pred.squeeze(1), n_class).permute(0, 4, 1, 2, 3)
            lbl_oh = F.one_hot(lbl.squeeze(1), n_class).permute(0, 4, 1, 2, 3)

            # Dice（排除背景通道）
            batch_dice = compute_dice_score(pred_oh[:, 1:], lbl_oh[:, 1:])
            total_dice += batch_dice.mean().item()
            count_dice += 1

            # HD95：函数式 API，逐 batch 计算
            hd95_tensor = compute_hausdorff_distance(
                y_pred=pred_oh[:, 1:],
                y=lbl_oh[:, 1:],
                include_background=False,
                percentile=95,
            )

            valid_mask = torch.isfinite(hd95_tensor)
            if valid_mask.any():
                batch_hd95 = hd95_tensor[valid_mask].mean().item()
                total_hd95 += batch_hd95
                count_hd95 += 1

    avg_dice = total_dice / max(count_dice, 1)
    avg_hd95 = total_hd95 / max(count_hd95, 1)
    return avg_dice, avg_hd95
