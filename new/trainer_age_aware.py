from __future__ import annotations

import math
import itertools
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value


def _summarize_tensor(name: str, tensor: torch.Tensor) -> str:
    tensor = tensor.detach()
    finite_mask = torch.isfinite(tensor)
    min_val = tensor[finite_mask].min().item() if finite_mask.any() else float("nan")
    max_val = tensor[finite_mask].max().item() if finite_mask.any() else float("nan")
    mean_val = tensor[finite_mask].mean().item() if finite_mask.any() else float("nan")
    return (
        f"{name}: shape={tuple(tensor.shape)} min={min_val:.4e} max={max_val:.4e} "
        f"mean={mean_val:.4e} any_nan={torch.isnan(tensor).any().item()} "
        f"any_inf={torch.isinf(tensor).any().item()}"
    )


def _check_finite(name: str, tensor: torch.Tensor, *, prefix: str = "") -> bool:
    if tensor is None:
        return True
    if torch.isfinite(tensor).all():
        return True
    print(f"[NaN DETECTED]{prefix} {_summarize_tensor(name, tensor)}", flush=True)
    return False


class ExponentialMovingAverage:
    """Track an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.0):
        if decay <= 0.0 or decay >= 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.model = model
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._register_initial()

    def _named_parameters(self):
        module = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        for name, param in module.named_parameters():
            if param.requires_grad:
                yield name, param

    def _register_initial(self) -> None:
        for name, param in self._named_parameters():
            self.shadow[name] = param.detach().clone()

    def update(self) -> None:
        for name, param in self._named_parameters():
            assert name in self.shadow, "EMA shadow not initialized"
            shadow_param = self.shadow[name]
            shadow_param.mul_(self.decay)
            shadow_param.add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self) -> None:
        self.backup = {}
        for name, param in self._named_parameters():
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        if not self.backup:
            return
        for name, param in self._named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: tensor.cpu() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if not state_dict:
            return
        self.shadow = {name: tensor.to(next(self.model.parameters()).device) for name, tensor in state_dict.items()}


class CombinedSegmentationLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 *,
                 class_weights: Optional[torch.Tensor] = None,
                 foreground_only: bool = True,
                 loss_config: str = "dice_focal",
                 focal_gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.foreground_only = foreground_only
        self.loss_config = loss_config
        self.class_weights = class_weights
        self.focal_gamma = float(focal_gamma)

        if loss_config == "dice_ce":
            self.dice_weight = 0.6
            self.ce_weight = 0.4
            self.focal_weight = 0.0
        elif loss_config == "dice_focal":
            self.dice_weight = 0.5
            self.ce_weight = 0.1
            self.focal_weight = 0.4
        elif loss_config == "dice_ce_focal":
            self.dice_weight = 0.4
            self.ce_weight = 0.3
            self.focal_weight = 0.3
        else:
            raise ValueError(f"Unsupported loss_config: {loss_config}")

        if self.dice_weight > 0:
            self.dice_loss = DiceLoss(
                to_onehot_y=True,
                softmax=True,
                include_background=not foreground_only,
                squared_pred=True,
            )
        if self.ce_weight > 0:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        if self.focal_weight > 0:
            if class_weights is not None and class_weights.ndim != 1:
                raise ValueError("class_weights must be a 1D tensor when using focal loss")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        if labels.ndim == 5 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        result: Dict[str, torch.Tensor] = {}

        labels_no_ignore = labels.clone()
        labels_no_ignore[labels_no_ignore < 0] = 0
        valid_mask: Optional[torch.Tensor]
        if labels.ndim >= 4:
            valid_mask = (labels >= 0).float()
        else:
            valid_mask = None

        if labels_no_ignore.ndim == 4:
            labels_for_dice = labels_no_ignore.unsqueeze(1)
        elif labels_no_ignore.ndim == 5 and labels_no_ignore.shape[1] == 1:
            labels_for_dice = labels_no_ignore
        else:
            raise RuntimeError(f"Unexpected label shape for Dice: {labels_no_ignore.shape}")

        if self.ce_weight > 0:
            ce = self.ce_loss(logits, labels.long())
        else:
            ce = torch.zeros(1, device=logits.device)
        result["ce"] = ce

        if self.dice_weight > 0:
            dice = self.dice_loss(logits, labels_for_dice.long())
        else:
            dice = torch.zeros(1, device=logits.device)
        result["dice"] = dice

        if self.focal_weight > 0:
            focal = self._compute_focal_loss(logits, labels_no_ignore.long(), valid_mask)
        else:
            focal = torch.zeros(1, device=logits.device)
        result["focal"] = focal

        total = self.dice_weight * dice + self.ce_weight * ce + self.focal_weight * focal
        result["total"] = total
        return result

    def _compute_focal_loss(self,
                             logits: torch.Tensor,
                             target: torch.Tensor,
                             valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        num_classes = logits.shape[1]
        if num_classes != self.num_classes:
            raise ValueError(
                f"Logits channel dimension ({num_classes}) does not match configured num_classes ({self.num_classes})"
            )

        # Convert targets to one-hot representation to align with the class dimension of the logits.
        target = target.clamp(min=0).long()
        one_hot = F.one_hot(target, num_classes=num_classes)
        # Move the class axis to position 1 to match the logits layout (N, C, ...).
        if one_hot.ndim > 2:
            permute_order = (0, one_hot.ndim - 1, *range(1, one_hot.ndim - 1))
            one_hot = one_hot.permute(permute_order)
        else:
            one_hot = one_hot.unsqueeze(1)
        one_hot = one_hot.to(dtype=logits.dtype, device=logits.device)

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        pt = (probs * one_hot).sum(dim=1)
        focal_factor = (1.0 - pt).clamp_min(0.0) ** self.focal_gamma
        loss = -focal_factor * (log_probs * one_hot).sum(dim=1)

        if self.class_weights is not None:
            view_shape = (1, -1) + (1,) * (loss.ndim - 1)
            class_w = self.class_weights.view(view_shape).to(logits.device)
            class_w = (class_w * one_hot).sum(dim=1)
            loss = loss * class_w

        if valid_mask is not None:
            valid_mask = valid_mask.to(device=loss.device, dtype=loss.dtype)
            loss = loss * valid_mask
            denom = valid_mask.sum().clamp_min(1.0)
        else:
            denom = loss.new_tensor(loss.numel(), dtype=loss.dtype)

        return loss.sum() / denom


def train_epoch(model: nn.Module,
                loader,
                optimizer: torch.optim.Optimizer,
                loss_fn: CombinedSegmentationLoss,
                prior_loss,
                *,
                device: torch.device,
                epoch: int,
                use_amp: bool = True,
                grad_clip: float = 12.0,
                grad_accum_steps: int = 1,
                writer=None,
                global_step: int = 0,
                is_main: bool = True,
                log_interval: int = 20,
                debug_mode: bool = False,
                debug_step_limit: int = 2,
                ema_helper: Optional[ExponentialMovingAverage] = None) -> Dict[str, float]:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    grad_accum_steps = max(1, int(grad_accum_steps))
    sum_metrics = {
        "loss": 0.0,
        "seg": 0.0,
        "prior": 0.0,
        "dice": 0.0,
        "ce": 0.0,
        "focal": 0.0,
        "volume": 0.0,
        "shape": 0.0,
        "edge": 0.0,
        "spectral": 0.0,
        "required": 0.0,
        "forbidden": 0.0,
        "symmetry": 0.0,
        "warmup": 0.0,
        "dyn_lambda": 0.0,
        "qap_mismatch": 0.0,
        "age_weight": 0.0,
        "adj_mae": 0.0,
        "spec_gap": 0.0,
        "symmetry_gap": 0.0,
        "required_missing": 0.0,
        "forbidden_present": 0.0,
    }
    batch_count = 0
    update_count = 0
    grad_norm_sum = 0.0
    accum_metrics = {key: 0.0 for key in sum_metrics}
    accum_batches = 0
    debug_step_limit = max(1, int(debug_step_limit))
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        ages = batch.get("age")
        if ages is None:
            raise RuntimeError("Data loader must provide 'age' key")
        ages = ages.to(device).view(-1)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            seg_losses = loss_fn(logits, labels)
            probs = torch.softmax(logits, dim=1)
            if prior_loss is not None:
                prior_dict = prior_loss(probs, ages)
            else:
                zeros = torch.zeros_like(seg_losses["total"])
                prior_dict = {
                    "total": zeros,
                    "volume": zeros,
                    "shape": zeros,
                    "edge": zeros,
                    "spectral": zeros,
                    "warmup": torch.tensor(1.0, device=probs.device, dtype=zeros.dtype),
                }
            loss = seg_losses["total"] + prior_dict["total"]

        check_targets = {
            "seg_total": seg_losses["total"],
            "seg_dice": seg_losses["dice"],
            "seg_ce": seg_losses["ce"],
            "seg_focal": seg_losses["focal"],
            "prior_total": prior_dict["total"],
            "prior_volume": prior_dict.get("volume"),
            "prior_shape": prior_dict.get("shape"),
            "prior_edge": prior_dict.get("edge"),
            "prior_spectral": prior_dict.get("spectral"),
            "prior_required": prior_dict.get("required"),
            "prior_forbidden": prior_dict.get("forbidden"),
            "prior_symmetry": prior_dict.get("symmetry"),
            "loss_total": loss,
        }
        finite_ok = True
        for name, tensor in check_targets.items():
            if tensor is None:
                continue
            if not _check_finite(name, tensor, prefix=f"[epoch={epoch} step={step}]"):
                finite_ok = False
        if not finite_ok:
            if is_main:
                print(
                    f"[Train][Epoch {epoch:03d}][Step {step:03d}] encountered non-finite loss components, skipping batch",
                    flush=True,
                )
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_for_backward = loss / grad_accum_steps
        scaler.scale(loss_for_backward).backward()

        batch_count += 1
        accum_batches += 1

        per_batch = {
            "loss": loss.detach().item(),
            "seg": seg_losses["total"].detach().item(),
            "prior": prior_dict["total"].detach().item(),
            "dice": seg_losses["dice"].detach().item(),
            "ce": seg_losses["ce"].detach().item(),
            "focal": seg_losses["focal"].detach().item(),
            "volume": prior_dict.get("volume", torch.zeros(1, device=device)).detach().item(),
            "shape": prior_dict.get("shape", torch.zeros(1, device=device)).detach().item(),
            "edge": prior_dict.get("edge", torch.zeros(1, device=device)).detach().item(),
            "spectral": prior_dict.get("spectral", torch.zeros(1, device=device)).detach().item(),
            "required": prior_dict.get("required", torch.zeros(1, device=device)).detach().item(),
            "forbidden": prior_dict.get("forbidden", torch.zeros(1, device=device)).detach().item(),
            "symmetry": prior_dict.get("symmetry", torch.zeros(1, device=device)).detach().item(),
            "warmup": float(prior_dict.get("warmup", torch.tensor(1.0, device=device)).detach().item()),
            "dyn_lambda": float(prior_dict.get("dyn_lambda", torch.tensor(1.0, device=device)).detach().item()),
            "qap_mismatch": float(prior_dict.get("qap_mismatch", torch.tensor(0.0, device=device)).detach().item()),
            "age_weight": float(prior_dict.get("age_weight", torch.tensor(1.0, device=device)).detach().item()),
            "adj_mae": float(prior_dict.get("adj_mae", torch.tensor(0.0, device=device)).detach().item()),
            "spec_gap": float(prior_dict.get("spec_gap", torch.tensor(0.0, device=device)).detach().item()),
            "symmetry_gap": float(prior_dict.get("symmetry_gap", torch.tensor(0.0, device=device)).detach().item()),
            "required_missing": float(prior_dict.get("required_missing", torch.tensor(0.0, device=device)).detach().item()),
            "forbidden_present": float(prior_dict.get("forbidden_present", torch.tensor(0.0, device=device)).detach().item()),
        }
        for key, value in per_batch.items():
            sum_metrics[key] += value
            accum_metrics[key] += value

        should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader))

        if should_step:
            scaler.unscale_(optimizer)

            bad_grad = None
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                if not torch.isfinite(param.grad).all():
                    bad_grad = (name, param.grad.detach())
                    break

            if bad_grad is not None:
                name, grad_tensor = bad_grad
                print(
                    f"[NaN GRAD][epoch={epoch} step={step}] {_summarize_tensor(name + '.grad', grad_tensor)}",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_count += 1
            grad_norm_sum += float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            if ema_helper is not None:
                ema_helper.update()

            if writer is not None and is_main and (global_step % log_interval == 0):
                denom = max(1, accum_batches)
                writer.add_scalar("train/loss", accum_metrics["loss"] / denom, global_step)
                writer.add_scalar("train/seg_loss", accum_metrics["seg"] / denom, global_step)
                writer.add_scalar("train/prior_loss", accum_metrics["prior"] / denom, global_step)
                writer.add_scalar("train/warmup", accum_metrics["warmup"] / denom, global_step)
                writer.add_scalar("train/prior_volume", accum_metrics["volume"] / denom, global_step)
                writer.add_scalar("train/prior_edge", accum_metrics["edge"] / denom, global_step)
                writer.add_scalar("train/prior_spectral", accum_metrics["spectral"] / denom, global_step)
                writer.add_scalar("train/prior_required", accum_metrics["required"] / denom, global_step)
                writer.add_scalar("train/prior_forbidden", accum_metrics["forbidden"] / denom, global_step)
                writer.add_scalar("train/prior_symmetry", accum_metrics["symmetry"] / denom, global_step)
                writer.add_scalar("train/prior_dyn_lambda", accum_metrics["dyn_lambda"] / denom, global_step)
                writer.add_scalar("train/prior_qap", accum_metrics["qap_mismatch"] / denom, global_step)
                if update_count > 0:
                    writer.add_scalar("train/grad_norm", grad_norm_sum / update_count, global_step)

            for key in accum_metrics:
                accum_metrics[key] = 0.0
            accum_batches = 0
            global_step += 1

        if debug_mode and is_main and step < debug_step_limit:
            print(
                f"[Train][Epoch {epoch:03d}][Step {step:03d}] "
                f"loss={loss.item():.4f} seg={seg_losses['total'].item():.4f} "
                f"prior={prior_dict['total'].item():.4f} warmup={prior_dict.get('warmup', torch.tensor(1.0)).item():.3f} "
                f"grad_norm={(grad_norm_sum / max(update_count, 1)):.3f}",
                flush=True,
            )
    metrics = {}
    denom = max(batch_count, 1)
    for key, total in sum_metrics.items():
        value = torch.tensor(total / denom, device=device)
        metrics[key] = float(reduce_mean(value))

    grad_norm_avg = torch.tensor(grad_norm_sum / max(update_count, 1), device=device)
    metrics["grad_norm"] = float(reduce_mean(grad_norm_avg))
    metrics["steps"] = update_count
    metrics["batches"] = batch_count
    metrics["epoch"] = epoch
    metrics["lr"] = optimizer.param_groups[0]["lr"]
    metrics["global_step"] = global_step
    metrics["updates"] = update_count
    metrics["warmup"] = metrics.get("warmup", 0.0)
    metrics["prior"] = metrics.get("prior", 0.0)
    return metrics


def validate_epoch(model: nn.Module,
                   loader,
                   *,
                   device: torch.device,
                   num_classes: int,
                   foreground_only: bool = True,
                   use_sliding_window: bool = False,
                   roi_size: Optional[Sequence[int]] = None,
                   sw_batch_size: int = 1,
                   sw_overlap: float = 0.25,
                   multi_scale: bool = False,
                   eval_scales: Optional[Sequence[float]] = None,
                   eval_tta: bool = False,
                   tta_axes: Optional[Sequence[int]] = None,
                   debug_mode: bool = False,
                   debug_step_limit: int = 1,
                   is_main: bool = True,
                   prior_loss=None,
                   return_per_class: bool = False) -> Dict[str, float]:
    model.eval()
    dice_metric = DiceMetric(include_background=not foreground_only, reduction="mean_batch")
    per_class_metric = None
    if return_per_class:
        per_class_metric = DiceMetric(
            include_background=not foreground_only,
            reduction="none",
            get_not_nans=True,
        )

    steps = 0
    debug_step_limit = max(1, int(debug_step_limit))
    struct_totals = {
        "adj_mae": 0.0,
        "spec_gap": 0.0,
        "symmetry_gap": 0.0,
        "required_missing": 0.0,
        "forbidden_present": 0.0,
        "dyn_lambda": 0.0,
        "qap_mismatch": 0.0,
        "age_weight": 0.0,
    }
    struct_steps = 0

    tta_axes = tuple(sorted({int(ax) for ax in (tta_axes or []) if ax in (0, 1, 2)}))

    def _flip_tensor(tensor: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
        if not axes:
            return tensor
        spatial_dims = list(range(tensor.ndim - 3, tensor.ndim))
        dims = [spatial_dims[ax] for ax in axes if ax < len(spatial_dims)]
        return torch.flip(tensor, dims=dims)

    def _run_inference(vol: torch.Tensor) -> torch.Tensor:
        if use_sliding_window:
            scales = list(eval_scales or ([1.0] if not multi_scale else eval_scales))
            if not scales:
                scales = [1.0]
            preds_multi = []
            for scale in scales:
                if scale != 1.0:
                    scaled = F.interpolate(
                        vol,
                        scale_factor=scale,
                        mode="trilinear",
                        align_corners=False,
                        recompute_scale_factor=True,
                    )
                else:
                    scaled = vol
                logits_scaled = sliding_window_inference(
                    scaled,
                    roi_size=tuple(roi_size) if roi_size is not None else scaled.shape[2:],
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=sw_overlap,
                )
                if scale != 1.0:
                    logits_scaled = F.interpolate(
                        logits_scaled,
                        size=vol.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                preds_multi.append(logits_scaled)
            return torch.stack(preds_multi, dim=0).mean(dim=0)
        return model(vol)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            labels_eval = labels.clone()
            labels_eval[labels_eval < 0] = 0
            labels_eval = labels_eval.long()

            brain_mask = labels_eval > 0
            if brain_mask.ndim == 5 and brain_mask.shape[1] == 1:
                brain_mask = brain_mask.squeeze(1)

            if eval_tta and tta_axes:
                combos = [()]  # include identity
                for r in range(1, len(tta_axes) + 1):
                    combos.extend(itertools.combinations(tta_axes, r))
                logits_list = []
                for combo in combos:
                    flipped = _flip_tensor(images, combo)
                    logits_combo = _run_inference(flipped)
                    logits_combo = _flip_tensor(logits_combo, combo)
                    logits_list.append(logits_combo)
                logits = torch.stack(logits_list, dim=0).mean(dim=0)
            else:
                logits = _run_inference(images)

            probs = torch.softmax(logits, dim=1)

            pred_labels = torch.argmax(probs, dim=1)
            pred_labels = torch.where(brain_mask, pred_labels, torch.zeros_like(pred_labels))
            preds = F.one_hot(pred_labels, num_classes=num_classes)
            preds = preds.permute(0, 4, 1, 2, 3).to(dtype=probs.dtype)

            preds = preds * brain_mask.unsqueeze(1)

            labels_eval_wo_channel = labels_eval
            if labels_eval_wo_channel.ndim == 5 and labels_eval_wo_channel.shape[1] == 1:
                labels_eval_wo_channel = labels_eval_wo_channel.squeeze(1)
            target = F.one_hot(labels_eval_wo_channel, num_classes=num_classes)
            target = target.permute(0, 4, 1, 2, 3).to(dtype=probs.dtype)

            target = target * brain_mask.unsqueeze(1)

            dice_metric(y_pred=preds, y=target)
            if per_class_metric is not None:
                per_class_metric(y_pred=preds, y=target)
            steps += 1

            if prior_loss is not None and "age" in batch:
                ages = batch["age"].to(device).view(-1)
                diag = prior_loss.diagnostics(probs, ages, apply_warmup=False)
                struct_steps += 1
                for key in struct_totals:
                    struct_totals[key] += float(diag.get(key, 0.0))

            if debug_mode and is_main and step < debug_step_limit:
                partial = dice_metric.aggregate().detach().cpu().numpy()
                print(f"[Val][Step {step:03d}] running dice mean={partial.mean():.4f}", flush=True)

    dice = dice_metric.aggregate()
    dice_metric.reset()

    per_class_scores: Optional[torch.Tensor] = None
    per_class_valid: Optional[torch.Tensor] = None
    if per_class_metric is not None:
        aggregate_out = per_class_metric.aggregate()
        per_class_metric.reset()

        raw_scores: Optional[torch.Tensor]
        raw_counts: Optional[torch.Tensor]

        if isinstance(aggregate_out, (tuple, list)) and len(aggregate_out) == 2:
            raw_scores, raw_counts = aggregate_out
        else:
            raw_scores = aggregate_out
            raw_counts = None

        def _to_tensor(data) -> torch.Tensor:
            if isinstance(data, torch.Tensor):
                tensor = data
            elif isinstance(data, (list, tuple)):
                tensor = torch.as_tensor(data, device=device, dtype=torch.float32)
            else:
                tensor = torch.as_tensor(float(data), device=device, dtype=torch.float32)
            return tensor.to(device=device, dtype=torch.float32)

        if raw_scores is not None:
            per_class_scores = _to_tensor(raw_scores)
            if per_class_scores.ndim >= 2:
                per_class_scores = per_class_scores.mean(dim=0)
            per_class_scores = torch.nan_to_num(per_class_scores, nan=0.0, posinf=0.0, neginf=0.0)

        if raw_counts is not None:
            per_class_valid = _to_tensor(raw_counts)
            if per_class_valid.ndim >= 2:
                per_class_valid = per_class_valid.sum(dim=0)
            per_class_valid = torch.nan_to_num(per_class_valid, nan=0.0, posinf=0.0, neginf=0.0)
            per_class_valid = per_class_valid.to(dtype=torch.float32)

        if per_class_scores is not None and per_class_valid is not None:
            # Guard against classes that never appear in the validation set.
            valid_mask = per_class_valid > 0.5
            if valid_mask.numel() == per_class_scores.numel():
                per_class_scores = torch.where(valid_mask, per_class_scores, torch.zeros_like(per_class_scores))
            else:
                per_class_valid = None

    if isinstance(dice, torch.Tensor):
        if dice.numel() == 0:
            dice = torch.tensor(0.0, device=device, dtype=torch.float32)
        elif dice.numel() > 1:
            dice = dice.mean()
        dice = dice.to(device)
    elif isinstance(dice, (list, tuple)):
        if len(dice) == 0:
            dice = torch.tensor(0.0, device=device, dtype=torch.float32)
        else:
            dice = torch.as_tensor(dice, device=device, dtype=torch.float32).mean()
    else:
        dice = torch.tensor(float(dice), device=device, dtype=torch.float32)

    dice = float(reduce_mean(dice))

    result = {"dice": dice, "steps": steps}
    if per_class_scores is not None:
        per_class_scores = reduce_mean(per_class_scores)
        result["per_class_dice"] = per_class_scores.detach().cpu().tolist()
        if per_class_valid is not None:
            per_class_valid = reduce_mean(per_class_valid)
            valid_counts = per_class_valid.detach().cpu().tolist()
            result["per_class_valid_counts"] = valid_counts
            missing = [idx for idx, count in enumerate(valid_counts) if count <= 0.5]
            if missing:
                result["per_class_missing_labels"] = missing
    if struct_steps > 0:
        avg = {k: v / struct_steps for k, v in struct_totals.items()}
        result["structural_violations"] = {
            "required_missing": avg["required_missing"],
            "forbidden_present": avg["forbidden_present"],
        }
        result["adjacency_errors"] = {
            "mean_adj_error": avg["adj_mae"],
            "spectral_distance": avg["spec_gap"],
        }
        result["symmetry_scores"] = [max(0.0, 1.0 - avg["symmetry_gap"])]
        result["prior_dyn_lambda"] = avg["dyn_lambda"]
        result["prior_qap_mismatch"] = avg["qap_mismatch"]
        result["prior_age_weight"] = avg["age_weight"]
    return result
