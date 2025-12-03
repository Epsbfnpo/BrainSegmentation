import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.losses import DiceLoss
from monai.metrics import HausdorffDistanceMetric
from monai.inferers import sliding_window_inference

# --- 巧思: 本地 HD95 计算器 (最终修复版) ---
class LocalHausdorffDistanceMetric(HausdorffDistanceMetric):
    """
    继承 MONAI HD95，屏蔽 DDP 同步。
    使用递归展平 + torch.cat 确保正确聚合 buffer 中的 batch 结果。
    """

    def _sync(self):
        # 1. 递归收集所有 tensor
        # MONAI 的 buffer 可能存为 [Tensor, Tensor] 或 [[Tensor], [Tensor]]
        all_tensors = []

        def recursive_extract(item):
            if isinstance(item, torch.Tensor):
                all_tensors.append(item)
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    recursive_extract(sub_item)

        for b in self._buffers:
            recursive_extract(b)

        # 2. 使用 cat 而不是 stack
        # HD95 每个 batch 返回 (B, C)。多个 batch 应该拼接成 (Total_B, C)。
        # stack 会变成 (N, B, C)，导致维度错误。
        if len(all_tensors) > 0:
            try:
                self._synced_tensors = torch.cat(all_tensors, dim=0)
            except Exception as e:
                print(f"[LocalMetric] Cat failed: {e}. Trying stack...")
                try:
                    self._synced_tensors = torch.stack(all_tensors)
                except Exception as e2:
                    print(f"[LocalMetric] Stack failed: {e2}. Returning empty.")
                    self._synced_tensors = torch.empty(0, device="cpu")
        else:
            self._synced_tensors = torch.empty(0, device="cpu")


# --- 巧思: 手写 Dice 计算 ---
def compute_dice_score(y_pred, y, smooth=1e-5):
    """
    y_pred, y: One-Hot Tensors (B, C, D, H, W)
    Returns: (B, C) dice scores
    """
    # 排除 Batch(0) 和 Channel(1) 维度，对 Spatial 维度求和
    dims = tuple(range(2, y_pred.ndim))
    intersect = (y_pred * y).sum(dim=dims)
    union = y_pred.sum(dim=dims) + y.sum(dim=dims)
    dice = (2.0 * intersect + smooth) / (union + smooth)
    return dice


# --- 巧思: NaN 检测 ---
def check_finite(name, tensor):
    if tensor is None:
        return True
    if torch.isfinite(tensor).all():
        return True
    print(f"[NaN DETECTED] {name} has NaN/Inf", flush=True)
    return False


# --- 巧思: EMA ---
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


# --- 巧思: 组合Loss (带自动 Deep Supervision 适配) ---
class CombinedLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True, include_background=False, squared_pred=True)
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, logits, labels):
        if labels.ndim == 4:
            labels = labels.unsqueeze(1)

        # Deep Supervision 尺寸适配
        if logits.shape[2:] != labels.shape[2:]:
            labels = F.interpolate(labels.float(), size=logits.shape[2:], mode='nearest')

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
    total_loss = 0
    steps = 0

    for batch in loader:
        img, lbl = batch['image'].to(device), batch['label'].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(img)
            if isinstance(logits, (list, tuple)):
                loss = sum([loss_fn(l, lbl) for l in logits]) / len(logits)
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


# --- 验证 Loop ---
def validate(model, loader, device, roi_size):
    model.eval()

    total_dice = 0.0
    total_hd95 = 0.0
    count_dice = 0
    count_hd95 = 0

    # 使用 Local Metric，避免 DDP Sync
    hd95_metric = LocalHausdorffDistanceMetric(
        include_background=False,
        percentile=95,
        reduction="mean"
    )

    def _predictor(x):
        out = model(x)
        if isinstance(out, (list, tuple)):
            return out[-1]
        return out

    with torch.no_grad():
        for batch in loader:
            img, lbl = batch['image'].to(device), batch['label'].to(device)
            lbl = lbl.long()
            lbl[lbl < 0] = 0

            logits = sliding_window_inference(img, roi_size, 1, _predictor, overlap=0.25, mode='gaussian')
            # 再次检查，确保 logits 是 Tensor
            if isinstance(logits, (list, tuple)):
                logits = logits[-1]

            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)
            n_class = logits.shape[1]

            pred_oh = F.one_hot(pred.squeeze(1), n_class).permute(0, 4, 1, 2, 3)
            lbl_oh = F.one_hot(lbl.squeeze(1), n_class).permute(0, 4, 1, 2, 3)

            # 1. DIY Dice
            batch_dice = compute_dice_score(pred_oh[:, 1:], lbl_oh[:, 1:])
            total_dice += batch_dice.mean().item()
            count_dice += 1

            # 2. Local HD95 (Compute-Reset)
            hd95_metric.reset()
            hd95_metric(y_pred=pred_oh, y=lbl_oh)

            # _sync 使用 cat，返回 (1, C)，aggregate 计算 mean 得到 scalar
            res = hd95_metric.aggregate()

            if isinstance(res, torch.Tensor):
                val = res.item()
                if np.isfinite(val):
                    total_hd95 += val
                    count_hd95 += 1

    avg_dice = total_dice / max(count_dice, 1)
    avg_hd95 = total_hd95 / max(count_hd95, 1)

    return avg_dice, avg_hd95
