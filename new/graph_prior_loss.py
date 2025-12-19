from __future__ import annotations

import json
import os
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

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


def _spectral_no_autocast():
    if torch.cuda.is_available() and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


class AgeConditionedGraphPriorLoss(nn.Module):
    def __init__(self,
                 *,
                 num_classes: int,
                 volume_stats_path: Optional[str] = None,
                 sdf_templates_path: Optional[str] = None,
                 adjacency_prior_path: Optional[str] = None,
                 r_mask_path: Optional[str] = None,
                 structural_rules_path: Optional[str] = None,
                 lr_pairs_path: Optional[str] = None,
                 lr_pairs: Optional[Sequence[Tuple[int, int]]] = None,
                 lambda_volume: float = 0.2,
                 lambda_shape: float = 0.2,
                 lambda_edge: float = 0.1,
                 lambda_spec: float = 0.05,
                 lambda_required: float = 0.05,
                 lambda_forbidden: float = 0.05,
                 lambda_symmetry: float = 0.02,
                 sdf_temperature: float = 4.0,
                 huber_delta: float = 1.0,
                 spectral_top_k: int = 20,
                 warmup_epochs: int = 10,
                 required_margin: float = 0.2,
                 forbidden_margin: float = 5e-4,
                 lambda_dyn: float = 0.2,
                 dyn_start_epoch: int = 50,
                 dyn_ramp_epochs: int = 40,
                 dyn_mismatch_ref: float = 0.08,
                 dyn_max_scale: float = 3.0,
                 age_reliability_min: float = 0.3,
                 age_reliability_pow: float = 0.5,
                 debug: bool = False,
                 debug_max_batches: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self._base_lambda_volume = float(lambda_volume)
        self._base_lambda_shape = float(lambda_shape)
        self._base_lambda_edge = float(lambda_edge)
        self._base_lambda_spec = float(lambda_spec)
        self.lambda_required = float(lambda_required)
        self.lambda_forbidden = float(lambda_forbidden)
        self.lambda_symmetry = float(lambda_symmetry)
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
        self._adj_count_max = 0.0
        self.required_margin = float(required_margin)
        self.forbidden_margin = float(forbidden_margin)
        self.lambda_dyn = float(lambda_dyn)
        self.dyn_start_epoch = int(dyn_start_epoch)
        self.dyn_ramp_epochs = max(1, int(dyn_ramp_epochs))
        self.dyn_mismatch_ref = max(1e-6, float(dyn_mismatch_ref))
        self.dyn_max_scale = max(1.0, float(dyn_max_scale))
        self.age_reliability_min = float(age_reliability_min)
        self.age_reliability_pow = float(age_reliability_pow)
        self._last_dyn_scale = 1.0

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

        self.adj_counts: Optional[torch.Tensor] = None
        self.adj_freq: Optional[torch.Tensor] = None
        self.required_edges: List[Tuple[int, int]] = []
        self.forbidden_edges: List[Tuple[int, int]] = []
        self.lr_pairs: List[Tuple[int, int]] = []

        if volume_stats_path:
            self._load_volume_stats(volume_stats_path)
        if sdf_templates_path:
            self._load_sdf_templates(sdf_templates_path)
        if adjacency_prior_path:
            self._load_adjacency_prior(adjacency_prior_path)
        if r_mask_path and os.path.exists(r_mask_path):
            mask = torch.from_numpy(np.load(r_mask_path)).float()
        else:
            mask = torch.empty(0)
        self.register_buffer("r_mask", mask)
        if structural_rules_path and os.path.exists(structural_rules_path):
            self._load_structural_rules(structural_rules_path)
        if lr_pairs is not None:
            self.lr_pairs = [(int(a), int(b)) for a, b in lr_pairs]
        elif lr_pairs_path and os.path.exists(lr_pairs_path):
            self.lr_pairs = self._load_lr_pairs(lr_pairs_path)

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

    def diagnostics(self, probs: torch.Tensor, ages: torch.Tensor, *, apply_warmup: bool = False) -> Dict[str, float]:
        original_warmup = self.warmup_epochs
        if not apply_warmup:
            self.warmup_epochs = 0
        try:
            with torch.no_grad():
                _ = self.forward(probs, ages)
        finally:
            self.warmup_epochs = original_warmup
        return self.get_last_metrics()

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
        order = np.argsort(age_values)
        age_values = age_values[order]
        matrices = matrices[order]
        matrices = self._align_classes_3d(matrices)
        self.adj_age_values = torch.tensor(age_values, dtype=torch.float32)
        self.adj_templates = torch.tensor(matrices, dtype=torch.float32)
        if "freq" in payload:
            freq = payload["freq"].astype(np.float32)[order]
            self.adj_freq = torch.tensor(self._align_classes_3d(freq), dtype=torch.float32)
        if "counts" in payload:
            counts = np.asarray(payload["counts"], dtype=np.float32)[order]
            self.adj_counts = torch.tensor(counts, dtype=torch.float32)
            self._adj_count_max = float(max(float(counts.max()), 0.0))

    def _load_structural_rules(self, path: str) -> None:
        with open(path, "r") as f:
            payload = json.load(f)
        required = payload.get("required", []) or []
        forbidden = payload.get("forbidden", []) or []
        self.required_edges = []
        self.forbidden_edges = []
        for pair in required:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                i, j = int(pair[0]), int(pair[1])
                if 0 <= i < self.num_classes and 0 <= j < self.num_classes and i != j:
                    self.required_edges.append((i, j))
        for pair in forbidden:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                i, j = int(pair[0]), int(pair[1])
                if 0 <= i < self.num_classes and 0 <= j < self.num_classes and i != j:
                    self.forbidden_edges.append((i, j))

    def _load_lr_pairs(self, path: str) -> List[Tuple[int, int]]:
        with open(path, "r") as f:
            payload = json.load(f)
        pairs: List[Tuple[int, int]] = []
        for pair in payload:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                a, b = int(pair[0]), int(pair[1])
                if a <= 0 or b <= 0:
                    continue
                a -= 1
                b -= 1
                if 0 <= a < self.num_classes and 0 <= b < self.num_classes:
                    pairs.append((a, b))
        return pairs

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
        if self.sdf_templates is None or self.sdf_age_values is None:
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

    def _age_reliability(self, age: float) -> float:
        if self.adj_age_values is None or self.adj_counts is None or self._adj_count_max <= 0:
            return 1.0
        ages = self.adj_age_values
        counts = self.adj_counts
        age_tensor = torch.tensor([age], device=ages.device)
        idx = torch.searchsorted(ages, age_tensor)
        idx = idx.clamp(max=ages.numel() - 1)
        lo = torch.clamp(idx - 1, min=0)
        hi = idx
        if ages.numel() == 1:
            weight = counts[0]
        else:
            age_lo = ages[lo]
            age_hi = ages[hi]
            denom = (age_hi - age_lo).clamp(min=1e-6)
            t = (age_tensor - age_lo) / denom
            weight = (1 - t) * counts[lo] + t * counts[hi]
        weight = float(weight.squeeze(0).item())
        rel = max(weight / (self._adj_count_max + 1e-6), self.age_reliability_min)
        return float(rel ** self.age_reliability_pow)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, probs: torch.Tensor, ages: torch.Tensor) -> Dict[str, torch.Tensor]:
        original_dtype = probs.dtype
        if probs.dtype != torch.float32:
            if self.debug_mode:
                print(
                    f"[GraphPriorLoss] casting probs from {original_dtype} to float32 for numerical stability",
                    flush=True,
                )
            probs = probs.float()

        B, C, X, Y, Z = probs.shape
        device = probs.device
        dtype = probs.dtype
        ages_list = ages.detach().cpu().tolist()

        warmup = self.get_warmup_factor()

        def _safe_tensor(tensor: torch.Tensor, name: str, *, age_value: Optional[float] = None) -> torch.Tensor:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                if self.debug_mode:
                    age_msg = f" age={age_value:.2f}" if age_value is not None else ""
                    finite_vals = tensor[torch.isfinite(tensor)]
                    if finite_vals.numel() > 0:
                        stats = finite_vals.detach().cpu()
                        min_val = float(stats.min())
                        max_val = float(stats.max())
                        msg = f"min={min_val:.6f} max={max_val:.6f}"
                    else:
                        msg = "no finite values"
                    print(
                        f"[GraphPriorLoss] detected invalid values in {name}{age_msg}; {msg}",
                        flush=True,
                    )
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
            return tensor

        zero = torch.tensor(0.0, device=device, dtype=dtype)
        weighted_volume = zero.clone()
        weighted_shape = zero.clone()
        weighted_edge = zero.clone()
        weighted_spec = zero.clone()
        weighted_required = zero.clone()
        weighted_forbidden = zero.clone()
        weighted_symmetry = zero.clone()

        volume_losses: List[torch.Tensor] = []
        shape_losses: List[torch.Tensor] = []
        edge_losses: List[torch.Tensor] = []
        spectral_losses: List[torch.Tensor] = []
        required_losses: List[torch.Tensor] = []
        forbidden_losses: List[torch.Tensor] = []
        symmetry_losses: List[torch.Tensor] = []

        acc_rel_volume = zero.clone()
        acc_rel_shape = zero.clone()
        acc_rel_edge = zero.clone()
        acc_rel_spec = zero.clone()
        acc_rel_required = zero.clone()
        acc_rel_forbidden = zero.clone()
        acc_rel_symmetry = zero.clone()

        age_weights: List[float] = []
        dyn_scales: List[float] = []
        qap_mismatches: List[float] = []
        adj_mae_vals: List[float] = []
        spec_gap_vals: List[float] = []
        symmetry_gap_vals: List[float] = []
        required_missing_total = 0
        forbidden_present_total = 0

        mask = self.r_mask.to(device) if self.r_mask.numel() > 0 else None

        for b, age in enumerate(ages_list):
            reliability = self._age_reliability(age)
            age_weights.append(reliability)
            lambda_factor = warmup * reliability
            lambda_factor_tensor = torch.tensor(lambda_factor, device=device, dtype=dtype)
            rel_tensor = torch.tensor(reliability, device=device, dtype=dtype)

            flat = probs[b].view(C, -1)
            flat = _safe_tensor(flat, "flat_probs", age_value=age)
            frac = flat.sum(dim=1)
            frac = frac / frac.sum().clamp(min=1e-6)
            frac = _safe_tensor(frac, "frac", age_value=age)

            if self.volume_bins is not None:
                mean, std = self._interp_volume(age, device)
                z = (frac - mean) / std.clamp(min=1e-3)
                z = _safe_tensor(z, "volume_z", age_value=age)
                vol_loss = _huber(z, self.huber_delta).mean()
                vol_loss = _safe_tensor(vol_loss, "volume_loss", age_value=age)
                weighted_volume = weighted_volume + self._base_lambda_volume * lambda_factor_tensor * vol_loss
                volume_losses.append(vol_loss)
                acc_rel_volume = acc_rel_volume + rel_tensor * vol_loss

            if self._base_lambda_shape > 0 and self.sdf_templates is not None:
                tmpl = self._interp_sdf(age, device, (X, Y, Z))
                if tmpl is not None:
                    clamped = probs[b].clamp(min=1e-4, max=1 - 1e-4)
                    sdf_pred = -torch.log(clamped) + torch.log1p(-clamped)
                    sdf_pred = sdf_pred / self.sdf_temperature
                    diff = sdf_pred - tmpl
                    if self.sdf_band is not None:
                        band_mask = (tmpl.abs() <= self.sdf_band).float()
                        diff = diff * band_mask
                    diff = _safe_tensor(diff, "shape_diff", age_value=age)
                    shape_loss = _huber(diff, self.huber_delta).mean()
                    shape_loss = _safe_tensor(shape_loss, "shape_loss", age_value=age)
                    weighted_shape = weighted_shape + self._base_lambda_shape * lambda_factor_tensor * shape_loss
                    shape_losses.append(shape_loss)
                    acc_rel_shape = acc_rel_shape + rel_tensor * shape_loss

            prior = self._interp_adj(age, device) if self.adj_templates is not None else None
            if prior is not None:
                adj_pred = torch.matmul(flat, flat.t()) / flat.shape[1]
                adj_pred = _row_normalise(adj_pred)
                adj_pred = _safe_tensor(adj_pred, "adj_pred", age_value=age)
                diff = adj_pred - prior
                if mask is not None:
                    diff = diff * mask
                diff = _safe_tensor(diff, "adj_diff", age_value=age)
                edge_loss = _huber(diff, self.huber_delta).mean()
                edge_loss = _safe_tensor(edge_loss, "edge_loss", age_value=age)
                weighted_edge = weighted_edge + self._base_lambda_edge * lambda_factor_tensor * edge_loss
                edge_losses.append(edge_loss)
                acc_rel_edge = acc_rel_edge + rel_tensor * edge_loss

                adj_mae = diff.abs().mean().detach().cpu().item()
                adj_mae_vals.append(adj_mae)

                dyn_scale = 1.0
                if self.lambda_dyn > 0 and self.current_epoch >= self.dyn_start_epoch:
                    progress = min(1.0, (self.current_epoch - self.dyn_start_epoch) / self.dyn_ramp_epochs)
                    dyn_scale = 1.0 + self.lambda_dyn * progress * (adj_mae / self.dyn_mismatch_ref)
                    dyn_scale = float(min(dyn_scale, self.dyn_max_scale))
                dyn_scales.append(dyn_scale)
                self._last_dyn_scale = dyn_scale
                qap_mismatches.append(adj_mae)

                if self._base_lambda_spec > 0:
                    with _spectral_no_autocast():
                        adj_pred_f32 = adj_pred.to(torch.float32)
                        prior_f32 = prior.to(torch.float32)
                        L_pred = _laplacian(adj_pred_f32)
                        L_prior = _laplacian(prior_f32)

                        L_pred = 0.5 * (L_pred + L_pred.transpose(-1, -2))
                        L_prior = 0.5 * (L_prior + L_prior.transpose(-1, -2))
                        eps = 1e-5
                        eye = torch.eye(C, device=L_pred.device, dtype=L_pred.dtype)

                        top_k = max(1, min(self.spectral_top_k, C - 1))
                        try:
                            evals_pred, evecs_pred = torch.linalg.eigh(L_pred + eps * eye)
                            evals_prior, evecs_prior = torch.linalg.eigh(L_prior + eps * eye)
                        except RuntimeError as err:
                            if self.debug_mode:
                                print(f"[GraphPriorLoss] eigh failed, skip spectral loss for this sample: {err}")
                        else:
                            eval_loss = F.mse_loss(evals_pred[:top_k], evals_prior[:top_k])
                            sub_pred = evecs_pred[:, :top_k]
                            sub_prior = evecs_prior[:, :top_k]
                            proj_pred = sub_pred @ sub_pred.t()
                            proj_prior = sub_prior @ sub_prior.t()
                            spec_loss = eval_loss + F.mse_loss(proj_pred, proj_prior)
                            spec_loss = _safe_tensor(spec_loss, "spectral_loss", age_value=age)
                            spec_loss = spec_loss.to(device=device, dtype=dtype)
                            weighted_spec = weighted_spec + (
                                self._base_lambda_spec * lambda_factor_tensor * float(dyn_scale) * spec_loss
                            )
                            spectral_losses.append(spec_loss)
                            acc_rel_spec = acc_rel_spec + rel_tensor * float(dyn_scale) * spec_loss
                            spec_gap_vals.append(float(spec_loss.detach().cpu().item()))

                if self.required_edges:
                    penalties = []
                    for i, j in self.required_edges:
                        val = adj_pred[i, j]
                        penalty = _huber(F.relu(self.required_margin - val), self.huber_delta)
                        penalty = _safe_tensor(penalty, "required_penalty", age_value=age)
                        penalties.append(penalty)
                        if val.detach().item() < self.required_margin:
                            required_missing_total += 1
                    if penalties:
                        req_loss = torch.stack(penalties).mean()
                        req_loss = _safe_tensor(req_loss, "required_loss", age_value=age)
                        weighted_required = weighted_required + self.lambda_required * lambda_factor_tensor * req_loss
                        required_losses.append(req_loss)
                        acc_rel_required = acc_rel_required + rel_tensor * req_loss

                if self.forbidden_edges:
                    penalties = []
                    for i, j in self.forbidden_edges:
                        val = adj_pred[i, j]
                        penalty = _huber(F.relu(val - self.forbidden_margin), self.huber_delta)
                        penalty = _safe_tensor(penalty, "forbidden_penalty", age_value=age)
                        penalties.append(penalty)
                        if val.detach().item() > self.forbidden_margin:
                            forbidden_present_total += 1
                    if penalties:
                        forb_loss = torch.stack(penalties).mean()
                        forb_loss = _safe_tensor(forb_loss, "forbidden_loss", age_value=age)
                        weighted_forbidden = weighted_forbidden + self.lambda_forbidden * lambda_factor_tensor * forb_loss
                        forbidden_losses.append(forb_loss)
                        acc_rel_forbidden = acc_rel_forbidden + rel_tensor * forb_loss

                if self.lr_pairs:
                    pair_losses = []
                    pair_gap = 0.0
                    for left, right in self.lr_pairs:
                        row_gap = torch.mean(torch.abs(adj_pred[left] - adj_pred[right]))
                        col_gap = torch.mean(torch.abs(adj_pred[:, left] - adj_pred[:, right]))
                        row_gap = _safe_tensor(row_gap, "sym_row_gap", age_value=age)
                        col_gap = _safe_tensor(col_gap, "sym_col_gap", age_value=age)
                        pair_losses.extend([row_gap, col_gap])
                        pair_gap += float((row_gap + col_gap).detach().cpu().item() * 0.5)
                    if pair_losses:
                        sym_tensor = torch.stack(pair_losses)
                        sym_tensor = _safe_tensor(sym_tensor, "sym_tensor", age_value=age)
                        sym_loss = _huber(sym_tensor, self.huber_delta).mean()
                        sym_loss = _safe_tensor(sym_loss, "symmetry_loss", age_value=age)
                        weighted_symmetry = weighted_symmetry + self.lambda_symmetry * lambda_factor_tensor * sym_loss
                        symmetry_losses.append(sym_loss)
                        acc_rel_symmetry = acc_rel_symmetry + rel_tensor * sym_loss
                        symmetry_gap_vals.append(pair_gap / max(len(self.lr_pairs), 1))

        denom = max(B, 1)
        weighted_volume = _safe_tensor(weighted_volume, "weighted_volume")
        weighted_shape = _safe_tensor(weighted_shape, "weighted_shape")
        weighted_edge = _safe_tensor(weighted_edge, "weighted_edge")
        weighted_spec = _safe_tensor(weighted_spec, "weighted_spec")
        weighted_required = _safe_tensor(weighted_required, "weighted_required")
        weighted_forbidden = _safe_tensor(weighted_forbidden, "weighted_forbidden")
        weighted_symmetry = _safe_tensor(weighted_symmetry, "weighted_symmetry")

        total = (weighted_volume + weighted_shape + weighted_edge + weighted_spec +
                 weighted_required + weighted_forbidden + weighted_symmetry) / denom
        total = _safe_tensor(total, "total_loss")

        volume_loss = torch.stack(volume_losses).mean() if volume_losses else zero
        shape_loss = torch.stack(shape_losses).mean() if shape_losses else zero
        edge_loss = torch.stack(edge_losses).mean() if edge_losses else zero
        spectral_loss = torch.stack(spectral_losses).mean() if spectral_losses else zero
        required_loss = torch.stack(required_losses).mean() if required_losses else zero
        forbidden_loss = torch.stack(forbidden_losses).mean() if forbidden_losses else zero
        symmetry_loss = torch.stack(symmetry_losses).mean() if symmetry_losses else zero

        volume_loss = _safe_tensor(volume_loss, "volume_loss_mean")
        shape_loss = _safe_tensor(shape_loss, "shape_loss_mean")
        edge_loss = _safe_tensor(edge_loss, "edge_loss_mean")
        spectral_loss = _safe_tensor(spectral_loss, "spectral_loss_mean")
        required_loss = _safe_tensor(required_loss, "required_loss_mean")
        forbidden_loss = _safe_tensor(forbidden_loss, "forbidden_loss_mean")
        symmetry_loss = _safe_tensor(symmetry_loss, "symmetry_loss_mean")

        avg_age_weight = sum(age_weights) / max(len(age_weights), 1)
        avg_dyn = sum(dyn_scales) / max(len(dyn_scales), 1) if dyn_scales else 1.0
        avg_qap = sum(qap_mismatches) / max(len(qap_mismatches), 1) if qap_mismatches else 0.0
        avg_adj_mae = sum(adj_mae_vals) / max(len(adj_mae_vals), 1) if adj_mae_vals else 0.0
        avg_spec_gap = sum(spec_gap_vals) / max(len(spec_gap_vals), 1) if spec_gap_vals else 0.0
        avg_sym_gap = sum(symmetry_gap_vals) / max(len(symmetry_gap_vals), 1) if symmetry_gap_vals else 0.0
        required_missing_avg = required_missing_total / max(len(ages_list), 1)
        forbidden_present_avg = forbidden_present_total / max(len(ages_list), 1)

        metrics = {
            "total": total,
            "volume": volume_loss,
            "shape": shape_loss,
            "edge": edge_loss,
            "spectral": spectral_loss,
            "required": required_loss,
            "forbidden": forbidden_loss,
            "symmetry": symmetry_loss,
            "awl_volume": acc_rel_volume / denom,
            "awl_shape": acc_rel_shape / denom,
            "awl_edge": acc_rel_edge / denom,
            "awl_spectral": acc_rel_spec / denom,
            "awl_required": acc_rel_required / denom,
            "awl_forbidden": acc_rel_forbidden / denom,
            "awl_symmetry": acc_rel_symmetry / denom,
            "warmup": torch.tensor(warmup, device=device, dtype=dtype),
            "dyn_lambda": torch.tensor(avg_dyn, device=device, dtype=dtype),
            "qap_mismatch": torch.tensor(avg_qap, device=device, dtype=dtype),
            "age_weight": torch.tensor(avg_age_weight, device=device, dtype=dtype),
            "adj_mae": torch.tensor(avg_adj_mae, device=device, dtype=dtype),
            "spec_gap": torch.tensor(avg_spec_gap, device=device, dtype=dtype),
            "symmetry_gap": torch.tensor(avg_sym_gap, device=device, dtype=dtype),
            "required_missing": torch.tensor(required_missing_avg, device=device, dtype=dtype),
            "forbidden_present": torch.tensor(forbidden_present_avg, device=device, dtype=dtype),
        }

        self._last_metrics = {k: float(v.detach().cpu().item()) for k, v in metrics.items() if isinstance(v, torch.Tensor)}

        if self.debug_mode and self._debug_counter < self.debug_max_batches:
            self._debug_counter += 1
            debug_msg = (
                f"[AgePrior] warmup={warmup:.3f} age_w={avg_age_weight:.3f} "
                f"vol={metrics['volume'].item():.4f} shape={metrics['shape'].item():.4f} "
                f"edge={metrics['edge'].item():.4f} spec={metrics['spectral'].item():.4f} "
                f"req={metrics['required'].item():.4f} forb={metrics['forbidden'].item():.4f} "
                f"sym={metrics['symmetry'].item():.4f} dyn={avg_dyn:.3f}"
            )
            print(debug_msg, flush=True)

        return metrics
