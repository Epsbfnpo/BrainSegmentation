#!/usr/bin/env python3
"""Utility to validate completeness of anatomical prior bundles.

Given a directory produced by ``build_graph_priors.py`` this script verifies that
all expected artifacts are present and have self-consistent shapes for the
foreground-only (87 class) setting used in the new training pipeline.  It can be
used both as a CLI tool and as a library helper inside the training scripts.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PriorCheckReport:
    """Structured result describing the health of a prior directory."""

    prior_dir: str
    expected_num_classes: Optional[int]
    detected_num_classes: Optional[int]
    missing_required: List[str]
    missing_optional: List[str]
    issues: List[str]

    @property
    def is_ok(self) -> bool:
        """Return True when there are no missing required files nor structural issues."""

        return not self.missing_required and not self.issues

    def summary_lines(self) -> List[str]:
        """Render a human readable report."""

        lines: List[str] = []
        lines.append(f"Prior directory: {self.prior_dir}")
        exp = "unknown" if self.expected_num_classes is None else str(self.expected_num_classes)
        det = "unknown" if self.detected_num_classes is None else str(self.detected_num_classes)
        lines.append(f"  Expected classes: {exp}")
        lines.append(f"  Detected classes: {det}")
        if self.missing_required:
            lines.append("  ❌ Missing required files:")
            for name in self.missing_required:
                lines.append(f"    - {name}")
        if self.missing_optional:
            lines.append("  ⚠️  Missing optional files:")
            for name in self.missing_optional:
                lines.append(f"    - {name}")
        if self.issues:
            lines.append("  ❌ Structural issues:")
            for issue in self.issues:
                lines.append(f"    - {issue}")
        if not self.missing_required and not self.issues:
            lines.append("  ✅ All required artifacts present and consistent.")
        return lines


_REQUIRED_FILES: Sequence[str] = (
    "class_map.json",
    "volume_stats.json",
    "adjacency_prior.npz",
    "sdf_templates.npz",
    "R_mask.npy",
    "structural_rules.json",
)

_OPTIONAL_FILES: Sequence[str] = (
    "R_meta.json",
)


def _require_square_matrix(path: str, num_classes: Optional[int], issues: List[str]) -> Optional[int]:
    """Load an ``.npy`` file and ensure it is a square matrix.

    Args:
        path: Path to the numpy file.
        num_classes: Previously inferred number of classes.
        issues: Mutable list to append textual problems to.

    Returns:
        Updated ``num_classes`` guess.
    """

    try:
        arr = np.load(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        issues.append(f"无法读取 {os.path.basename(path)}: {exc}")
        return num_classes

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        issues.append(
            f"{os.path.basename(path)} 不是方阵，实际形状为 {arr.shape}"
        )
        return num_classes

    if num_classes is None:
        num_classes = int(arr.shape[0])
    elif arr.shape[0] != num_classes:
        issues.append(
            f"{os.path.basename(path)} 尺寸 {arr.shape[0]} 与预期类别数 {num_classes} 不匹配"
        )
    return num_classes


def _check_binary_mask(path: str, num_classes: Optional[int], issues: List[str]) -> None:
    try:
        mask = np.load(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        issues.append(f"无法读取 {os.path.basename(path)}: {exc}")
        return

    if mask.shape[0] != mask.shape[1]:
        issues.append(f"R_mask 不是方阵，形状为 {mask.shape}")
    if num_classes is not None and mask.shape[0] != num_classes:
        issues.append(
            f"R_mask 大小 {mask.shape[0]} 与预期类别数 {num_classes} 不一致"
        )
    unique_vals = np.unique(mask)
    if not np.all(np.isin(unique_vals, [0, 1])):
        issues.append(
            "R_mask 包含非二值元素: " + ", ".join(map(str, unique_vals))
        )


def _check_class_map(path: str, issues: List[str]) -> Optional[int]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        issues.append(f"无法解析 {os.path.basename(path)}: {exc}")
        return None

    index_map = data.get("index_to_raw_label")
    reverse_map = data.get("raw_label_to_index")
    if not isinstance(index_map, dict) or not isinstance(reverse_map, dict):
        issues.append("class_map.json 缺少 index_to_raw_label/raw_label_to_index 字段")
        return None

    inferred = len(index_map)
    for idx_str, raw in index_map.items():
        try:
            idx = int(idx_str)
            raw = int(raw)
        except (TypeError, ValueError):
            issues.append(f"class_map.json 条目 {idx_str}:{raw} 不是整数对")
            continue
        if str(raw) not in reverse_map:
            issues.append(f"class_map.json 缺少反向映射 {raw}")
        elif int(reverse_map[str(raw)]) != idx:
            issues.append(f"class_map.json 的反向映射 {raw}->{reverse_map[str(raw)]} 与索引 {idx} 不一致")
    return inferred


def _check_adjacency_prior(path: str, num_classes: Optional[int], issues: List[str]) -> Optional[int]:
    try:
        payload = np.load(path, allow_pickle=True)
    except Exception as exc:
        issues.append(f"无法读取 {os.path.basename(path)}: {exc}")
        return num_classes

    matrices = payload.get("A_prior")
    ages = payload.get("ages")
    counts = payload.get("counts")
    if matrices is None or ages is None:
        issues.append("adjacency_prior.npz 缺少 A_prior 或 ages 数组")
        return num_classes
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        issues.append(f"A_prior 维度应为 (N,C,C)，当前形状 {matrices.shape}")
        return num_classes
    detected = matrices.shape[1]
    if num_classes is not None and detected != num_classes:
        issues.append(f"A_prior 的类别数 {detected} 与预期 {num_classes} 不一致")
    if counts is not None and len(counts) != matrices.shape[0]:
        issues.append("adjacency_prior.npz 中 counts 长度与 A_prior 第一维不一致")
    return detected if num_classes is None else num_classes


def _check_sdf_templates(path: str, num_classes: Optional[int], issues: List[str]) -> Optional[int]:
    try:
        payload = np.load(path, allow_pickle=True)
    except Exception as exc:
        issues.append(f"无法读取 {os.path.basename(path)}: {exc}")
        return num_classes

    templates = payload.get("T_mean")
    if templates is None:
        issues.append("sdf_templates.npz 缺少 T_mean")
        return num_classes
    if templates.ndim != 5:
        issues.append(f"T_mean 维度应为 (N,C,X,Y,Z)，当前形状 {templates.shape}")
        return num_classes
    detected = templates.shape[1]
    if num_classes is not None and detected != num_classes:
        issues.append(f"SDF 模板类别数 {detected} 与预期 {num_classes} 不一致")
    return detected if num_classes is None else num_classes


def _check_structural_rules(path: str, num_classes: Optional[int], issues: List[str]) -> None:
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        issues.append(f"无法解析 {os.path.basename(path)}: {exc}")
        return

    for key in ("required", "forbidden"):
        edges = payload.get(key, []) or []
        for idx, pair in enumerate(edges):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                issues.append(f"{key} 第 {idx} 项不是长度为2的列表")
                continue
            try:
                i, j = int(pair[0]), int(pair[1])
            except (TypeError, ValueError):
                issues.append(f"{key} 第 {idx} 项包含非整数: {pair}")
                continue
            if num_classes is not None and (i < 0 or j < 0 or i >= num_classes or j >= num_classes):
                issues.append(f"{key} 第 {idx} 对 {pair} 超出类别范围 [0, {num_classes - 1}]")


def _check_required_edges(path: str, num_classes: Optional[int], issues: List[str], key: str) -> None:
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive logging
        issues.append(f"无法解析 {os.path.basename(path)}: {exc}")
        return

    edges = data.get(key, [])
    for idx, pair in enumerate(edges):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            issues.append(f"{os.path.basename(path)} 第 {idx} 项不是长度为2的列表")
            continue
        i, j = pair
        if not all(isinstance(v, int) for v in pair):
            issues.append(f"{os.path.basename(path)} 第 {idx} 项包含非整数: {pair}")
            continue
        if num_classes is not None and (i < 0 or i >= num_classes or j < 0 or j >= num_classes):
            issues.append(
                f"{os.path.basename(path)} 第 {idx} 对 {pair} 超出类别范围 [0,{num_classes - 1}]"
            )


def _check_volume_stats(path: str, num_classes: Optional[int], issues: List[str]) -> None:
    try:
        with open(path, "r") as f:
            stats = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive logging
        issues.append(f"无法解析 {os.path.basename(path)}: {exc}")
        return

    for age, payload in stats.items():
        means = payload.get("means")
        stds = payload.get("stds")
        counts = payload.get("n")
        if means is None or stds is None:
            issues.append(f"volume_stats {age} 缺少 'means' 或 'stds'")
            continue
        if num_classes is not None and len(means) != num_classes:
            issues.append(
                f"volume_stats {age} 的均值长度 {len(means)} 与类别数 {num_classes} 不一致"
            )
        if num_classes is not None and len(stds) != num_classes:
            issues.append(
                f"volume_stats {age} 的标准差长度 {len(stds)} 与类别数 {num_classes} 不一致"
            )
        if counts is not None and num_classes is not None and len(counts) != num_classes:
            issues.append(
                f"volume_stats {age} 的样本计数长度 {len(counts)} 与类别数 {num_classes} 不一致"
            )


def _check_age_weights(path: str, num_classes: Optional[int], issues: List[str]) -> None:
    try:
        with open(path, "r") as f:
            age_weights = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive logging
        issues.append(f"无法解析 {os.path.basename(path)}: {exc}")
        return

    for age, mat in age_weights.items():
        arr = np.asarray(mat)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            issues.append(f"age_weights {age} 不是方阵，形状为 {arr.shape}")
            continue
        if num_classes is not None and arr.shape[0] != num_classes:
            issues.append(
                f"age_weights {age} 大小 {arr.shape[0]} 与类别数 {num_classes} 不匹配"
            )


def _check_symmetry_stats(path: str, num_classes: Optional[int], issues: List[str]) -> None:
    try:
        with open(path, "r") as f:
            stats = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive logging
        issues.append(f"无法解析 {os.path.basename(path)}: {exc}")
        return

    for key, payload in stats.items():
        required_keys = {"mean", "std", "min", "max"}
        if not required_keys.issubset(payload):
            missing = required_keys.difference(payload)
            issues.append(f"symmetry_stats {key} 缺少字段: {', '.join(sorted(missing))}")


def _check_r_meta(path: str, issues: List[str]) -> None:
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        issues.append(f"无法解析 {os.path.basename(path)}: {exc}")
        return
    for key in ("policy", "arg"):
        if key not in payload:
            issues.append(f"R_meta.json 缺少字段 '{key}'")


def check_prior_directory(
    prior_dir: str,
    *,
    expected_num_classes: Optional[int] = None,
    include_optional: bool = True,
) -> PriorCheckReport:
    """Validate a directory containing graph priors."""

    missing_required: List[str] = []
    missing_optional: List[str] = []
    issues: List[str] = []
    num_classes = expected_num_classes

    for fname in _REQUIRED_FILES:
        fpath = os.path.join(prior_dir, fname)
        if not os.path.exists(fpath):
            missing_required.append(fname)
            continue
        if fname == "class_map.json":
            inferred = _check_class_map(fpath, issues)
            if inferred is not None:
                if num_classes is None:
                    num_classes = inferred
                elif inferred != num_classes:
                    issues.append(
                        f"class_map.json 的类别数 {inferred} 与预期 {num_classes} 不一致"
                    )
        elif fname == "volume_stats.json":
            _check_volume_stats(fpath, num_classes, issues)
        elif fname == "adjacency_prior.npz":
            num_classes = _check_adjacency_prior(fpath, num_classes, issues)
        elif fname == "sdf_templates.npz":
            num_classes = _check_sdf_templates(fpath, num_classes, issues)
        elif fname == "R_mask.npy":
            _check_binary_mask(fpath, num_classes, issues)
        elif fname == "structural_rules.json":
            _check_structural_rules(fpath, num_classes, issues)

    if include_optional:
        for fname in _OPTIONAL_FILES:
            fpath = os.path.join(prior_dir, fname)
            if not os.path.exists(fpath):
                missing_optional.append(fname)
                continue
            if fname == "R_meta.json":
                _check_r_meta(fpath, issues)
            else:
                _check_symmetry_stats(fpath, num_classes, issues)

    return PriorCheckReport(
        prior_dir=os.path.abspath(prior_dir),
        expected_num_classes=expected_num_classes,
        detected_num_classes=num_classes,
        missing_required=missing_required,
        missing_optional=missing_optional,
        issues=issues,
    )


def _format_cli_report(report: PriorCheckReport) -> str:
    header = "=" * 80
    lines = [header]
    lines.extend(report.summary_lines())
    lines.append(header)
    return "\n".join(lines)


def run_cli(directories: Sequence[str], expected_num_classes: Optional[int], include_optional: bool) -> int:
    status_ok = True
    for directory in directories:
        report = check_prior_directory(
            directory,
            expected_num_classes=expected_num_classes,
            include_optional=include_optional,
        )
        print(_format_cli_report(report))
        status_ok = status_ok and report.is_ok
    return 0 if status_ok else 1


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate anatomical prior bundles")
    parser.add_argument(
        "--dir",
        dest="directories",
        action="append",
        required=True,
        help="Prior directory to inspect (can be specified multiple times)",
    )
    parser.add_argument(
        "--num-classes",
        dest="num_classes",
        type=int,
        default=None,
        help="Expected number of foreground classes (default: infer automatically)",
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Ignore optional artifacts such as symmetry_stats.json",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    return run_cli(
        directories=args.directories,
        expected_num_classes=args.num_classes,
        include_optional=not args.skip_optional,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
