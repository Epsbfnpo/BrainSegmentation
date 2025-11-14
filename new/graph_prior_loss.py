from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    abs_x = x.abs()
    quadratic = torch.minimum(abs_x, torch.tensor(delta, device=x.device, dtype=x.dtype))
    linear = abs_x - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def _row_normalise(mat: torch.Tensor) -> torch.Tensor:
    mat = mat.clone()
    mat = mat - torch.diag_embed(torch.diagonal(mat, dim1=-2, dim2=-1))
    rowsum = mat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return mat / rowsum


def _laplacian(adj: torch.Tensor) -> torch.Tensor:
    adj = 0.5 * (adj + adj.transpose(-1, -2))
    deg = torch.diag_embed(adj.sum(dim=-1))
    return deg - adj


class AgeConditionedGraphPriorLoss(nn.Module):
    def __init__(self,
                 *,
                 num_classes: int,
                 volume_stats_path: Optional[str] = None,
                 sdf_templates_path: Optional[str] = None,
                 adjacency_prior_path: Optional[str] = None,
                 r_mask_path: Optional[str] = None,
                 lambda_volume: float = 0.2,
                 lambda_shape: float = 0.2,
                 lambda_edge: float = 0.1,
                 lambda_spec: float = 0.05,
                 sdf_temperature: float = 4.0,
                 huber_delta: float = 1.0,
                 spectral_top_k: int = 20,
                 warmup_epochs: int = 10,
                 debug: bool = False,
                 debug_max_batches: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self._base_lambda_volume = float(lambda_volume)
        self._base_lambda_shape = float(lambda_shape)
        self._base_lambda_edge = float(lambda_edge)
        self._base_lambda_spec = float(lambda_spec)
        self.sdf_temperature = sdf_temperature
        self.huber_delta = huber_delta
        self.spectral_top_k = spectral_top_k
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.current_epoch = 0
        self.total_epochs: Optional[int] = None
        self.debug_mode = debug
        self.debug_max_batches = max(0, int(debug_max_batches))
        self._debug_counter = 0
        self._last_metrics: Dict[str, float] = {}

        self.volume_bins: Optional[torch.Tensor] = None
        self.volume_means: Optional[torch.Tensor] = None
        self.volume_stds: Optional[torch.Tensor] = None
        self.volume_bin_width: Optional[float] = None

        self.sdf_age_values: Optional[torch.Tensor] = None
        self.sdf_templates: Optional[Dict[float, torch.Tensor]] = None
        self.sdf_band: Optional[float] = None

        self.adj_age_values: Optional[torch.Tensor] = None
        self.adj_templates: Optional[torch.Tensor] = None
        self.adj_bin_width: Optional[float] = None

        self.r_mask: Optional[torch.Tensor] = None

        if volume_stats_path:
            self._load_volume_stats(volume_stats_path)
        if sdf_templates_path:
            self._load_sdf_templates(sdf_templates_path)
        if adjacency_prior_path:
            self._load_adjacency_prior(adjacency_prior_path)
        if r_mask_path and os.path.exists(r_mask_path):
            mask = torch.from_numpy(np.load(r_mask_path)).float()
            self.register_buffer("r_mask", mask)
        else:
            self.register_buffer("r_mask", torch.empty(0))

    # ------------------------------------------------------------------
    # schedule helpers
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Update the internal epoch counter for warmup scheduling."""

        self.current_epoch = int(epoch)

    def configure_schedule(self, total_epochs: int) -> None:
        self.total_epochs = int(total_epochs)

    def get_warmup_factor(self) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        return float(min(1.0, max(0.0, self.current_epoch / max(1, self.warmup_epochs))))

    def set_debug(self, enabled: bool, max_batches: Optional[int] = None) -> None:
        self.debug_mode = bool(enabled)
        if max_batches is not None:
            self.debug_max_batches = max(0, int(max_batches))
        if not self.debug_mode:
            self._debug_counter = 0

    def get_last_metrics(self) -> Dict[str, float]:
        return dict(self._last_metrics)

    # ------------------------------------------------------------------
    # loading helpers
    # ------------------------------------------------------------------
    def _load_volume_stats(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        bins = []
        means = []
        stds = []
        bin_width = None
        for key, entry in data.items():
            try:
                age_bin = int(key)
            except ValueError:
                continue
            if bin_width is None:
                bin_width = float(entry.get("age_bin_width", 1.0))
            bins.append(age_bin)
            vec_mean = np.asarray(entry.get("means", []), dtype=np.float32)
            vec_std = np.asarray(entry.get("stds", []), dtype=np.float32)
            vec_mean = self._align_classes(vec_mean)
            vec_std = self._align_classes(vec_std)
            std_floor = 0.02
            vec_std = np.clip(vec_std, std_floor, None)
            means.append(vec_mean)
            stds.append(vec_std)
        if not bins:
            return
        order = np.argsort(bins)
        self.volume_bins = torch.tensor(np.array(bins)[order], dtype=torch.float32)
        self.volume_means = torch.tensor(np.stack(means, axis=0)[order], dtype=torch.float32)
        self.volume_stds = torch.tensor(np.stack(stds, axis=0)[order], dtype=torch.float32)
        self.volume_bin_width = bin_width or 1.0

    def _load_sdf_templates(self, path: str) -> None:
        payload = np.load(path, allow_pickle=True)
        ages = payload.get("ages")
        if ages is None or ages.size == 0:
            return
        templates_np = payload["T_mean"].astype(np.float32)
        meta = payload.get("meta", {})
        self.sdf_band = float(meta.get("band", 8.0)) if isinstance(meta, dict) else 8.0
        bin_width = float(meta.get("bin_width", 1.0)) if isinstance(meta, dict) else 1.0
        age_values = ages.astype(np.float32) * bin_width
        order = np.argsort(age_values)
        age_values = age_values[order]
        templates_list = []
        for idx in order:
            templates_list.append(torch.from_numpy(self._align_classes_5d(templates_np[idx])))
        self.sdf_age_values = torch.tensor(age_values, dtype=torch.float32)
        self.sdf_templates = torch.stack(templates_list, dim=0)

    def _load_adjacency_prior(self, path: str) -> None:
        payload = np.load(path, allow_pickle=True)
        ages = payload.get("ages")
        if ages is None or ages.size == 0:
            return
        matrices = payload["A_prior"].astype(np.float32)
        meta = payload.get("meta", {})
        bin_width = float(meta.get("bin_width", 1.0)) if isinstance(meta, dict) else 1.0
        self.adj_bin_width = bin_width
        age_values = ages.astype(np.float32) * bin_width
        matrices = self._align_classes_3d(matrices)
        self.adj_age_values = torch.tensor(age_values, dtype=torch.float32)
        self.adj_templates = torch.tensor(matrices, dtype=torch.float32)

    def _align_classes(self, vector: np.ndarray) -> np.ndarray:
        if vector.size == self.num_classes:
            return vector
        if vector.size > self.num_classes:
            return vector[:self.num_classes]
        padded = np.zeros((self.num_classes,), dtype=vector.dtype)
        padded[:vector.size] = vector
        return padded

    def _align_classes_3d(self, array: np.ndarray) -> np.ndarray:
        if array.shape[-1] == self.num_classes:
            return array
        trimmed = array[..., :self.num_classes]
        trimmed = trimmed[:, :self.num_classes, :]
        return trimmed

    def _align_classes_5d(self, array: np.ndarray) -> np.ndarray:
        if array.shape[0] == self.num_classes:
            return array
        trimmed = array[:self.num_classes]
        return trimmed

    # ------------------------------------------------------------------
    # interpolation helpers
    # ------------------------------------------------------------------
    def _interp_volume(self, age: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.volume_bins is None:
            zero = torch.zeros(self.num_classes, device=device)
            return zero, torch.ones(self.num_classes, device=device)
        ages = self.volume_bins * (self.volume_bin_width or 1.0)
        age_tensor = torch.tensor([age], device=ages.device)
        idx = torch.searchsorted(ages, age_tensor)
        idx = idx.clamp(max=ages.numel() - 1)
        lo = torch.clamp(idx - 1, min=0)
        hi = torch.clamp(idx, min=0)
        if ages.numel() == 1:
            mean = self.volume_means[0]
            std = self.volume_stds[0]
        else:
            age_lo = ages[lo]
            age_hi = ages[hi]
            denom = (age_hi - age_lo).clamp(min=1e-6)
            t = (age_tensor - age_lo) / denom
            mean = (1 - t) * self.volume_means[lo] + t * self.volume_means[hi]
            std = (1 - t) * self.volume_stds[lo] + t * self.volume_stds[hi]
        return mean.squeeze(0).to(device), std.squeeze(0).to(device)

    def _interp_sdf(self, age: float, device: torch.device, target_shape: Tuple[int, int, int]) -> Optional[torch.Tensor]:
        if not self.sdf_templates or self.sdf_age_values is None:
            return None
        ages = self.sdf_age_values
        age_tensor = torch.tensor([age], device=ages.device)
        idx = torch.searchsorted(ages, age_tensor)
        idx = idx.clamp(max=ages.numel() - 1)
        lo_idx = int(torch.clamp(idx - 1, min=0).item())
        hi_idx = int(idx.item())
        lo_age = float(ages[lo_idx])
        hi_age = float(ages[hi_idx])
        if hi_idx == lo_idx:
            t = 0.0
        else:
            t = float((age - lo_age) / (hi_age - lo_age + 1e-6))
        tmpl_lo = self.sdf_templates[lo_idx]
        tmpl_hi = self.sdf_templates[hi_idx]
        tmpl = (1 - t) * tmpl_lo + t * tmpl_hi
        tmpl = tmpl.to(device=device)
        if tmpl.shape[1:] != target_shape:
            tmpl = F.interpolate(tmpl.unsqueeze(0), size=target_shape, mode="trilinear", align_corners=False).squeeze(0)
        return tmpl

    def _interp_adj(self, age: float, device: torch.device) -> Optional[torch.Tensor]:
        if self.adj_templates is None or self.adj_age_values is None:
            return None
        ages = self.adj_age_values
        age_tensor = torch.tensor([age], device=ages.device)
        idx = torch.searchsorted(ages, age_tensor)
        idx = idx.clamp(max=ages.numel() - 1)
        lo_idx = int(torch.clamp(idx - 1, min=0).item())
        hi_idx = int(idx.item())
        if ages.numel() == 1:
            prior = self.adj_templates[0]
        else:
            denom = (ages[hi_idx] - ages[lo_idx]).clamp(min=1e-6)
            t = (age_tensor - ages[lo_idx]) / denom
            prior = (1 - t) * self.adj_templates[lo_idx] + t * self.adj_templates[hi_idx]
        return prior.to(device)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, probs: torch.Tensor, ages: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, X, Y, Z = probs.shape
        device = probs.device
        ages = ages.detach().cpu().tolist()

        volume_losses = []
        shape_losses = []
        edge_losses = []
        spectral_losses = []

        warmup = self.get_warmup_factor()
        lambda_volume = self._base_lambda_volume * warmup
        lambda_shape = self._base_lambda_shape * warmup
        lambda_edge = self._base_lambda_edge * warmup
        lambda_spec = self._base_lambda_spec * warmup

        for b in range(B):
            age = ages[b]
            frac = probs[b].view(C, -1).sum(dim=1)
            frac = frac / frac.sum().clamp(min=1e-6)

            if self.volume_bins is not None:
                mean, std = self._interp_volume(age, device)
                z = (frac - mean) / std.clamp(min=1e-3)
                volume_losses.append(_huber(z, self.huber_delta).mean())

            if self.lambda_shape > 0 and self.sdf_templates is not None:
                tmpl = self._interp_sdf(age, device, (X, Y, Z))
                if tmpl is not None:
                    clamped = probs[b].clamp(min=1e-4, max=1 - 1e-4)
                    sdf_pred = -torch.log(clamped) + torch.log1p(-clamped)
                    sdf_pred = sdf_pred / self.sdf_temperature
                    mask = None
                    if self.sdf_band is not None:
                        mask = (tmpl.abs() <= self.sdf_band).float()
                    diff = sdf_pred - tmpl
                    if mask is not None:
                        diff = diff * mask
                    shape_losses.append(_huber(diff, self.huber_delta).mean())

            if self.lambda_edge > 0 and self.adj_templates is not None:
                prior = self._interp_adj(age, device)
                if prior is not None:
                    flat = probs[b].view(C, -1)
                    adj_pred = torch.matmul(flat, flat.t()) / flat.shape[1]
                    adj_pred = _row_normalise(adj_pred)
                    diff = adj_pred - prior
                    if self.r_mask.numel() > 0:
                        diff = diff * self.r_mask.to(device)
                    edge_losses.append(_huber(diff, self.huber_delta).mean())

                    if self.lambda_spec > 0:
                        L_pred = _laplacian(adj_pred)
                        L_prior = _laplacian(prior)
                        top_k = min(self.spectral_top_k, C - 1)
                        evals_pred, evecs_pred = torch.linalg.eigh(L_pred)
                        evals_prior, evecs_prior = torch.linalg.eigh(L_prior)
                        spec_loss = F.mse_loss(evals_pred[:top_k], evals_prior[:top_k])
                        sub_pred = evecs_pred[:, :top_k]
                        sub_prior = evecs_prior[:, :top_k]
                        proj_pred = sub_pred @ sub_pred.t()
                        proj_prior = sub_prior @ sub_prior.t()
                        spec_loss = spec_loss + F.mse_loss(proj_pred, proj_prior)
                        spectral_losses.append(spec_loss)

        volume_loss = torch.stack(volume_losses).mean() if volume_losses else torch.zeros(1, device=device)
        shape_loss = torch.stack(shape_losses).mean() if shape_losses else torch.zeros(1, device=device)
        edge_loss = torch.stack(edge_losses).mean() if edge_losses else torch.zeros(1, device=device)
        spectral_loss = torch.stack(spectral_losses).mean() if spectral_losses else torch.zeros(1, device=device)

        total = (lambda_volume * volume_loss +
                 lambda_shape * shape_loss +
                 lambda_edge * edge_loss +
                 lambda_spec * spectral_loss)

        metrics = {
            "total": total,
            "volume": volume_loss,
            "shape": shape_loss,
            "edge": edge_loss,
            "spectral": spectral_loss,
            "warmup": torch.tensor(warmup, device=device, dtype=total.dtype),
        }

        self._last_metrics = {k: float(v.detach().cpu().item()) for k, v in metrics.items() if isinstance(v, torch.Tensor)}

        if self.debug_mode and self._debug_counter < self.debug_max_batches:
            self._debug_counter += 1
            debug_msg = (
                f"[AgePrior] warmup={warmup:.3f} "
                f"vol={metrics['volume'].item():.4f} "
                f"shape={metrics['shape'].item():.4f} "
                f"edge={metrics['edge'].item():.4f} "
                f"spec={metrics['spectral'].item():.4f}"
            )
            print(debug_msg, flush=True)

        return metrics
