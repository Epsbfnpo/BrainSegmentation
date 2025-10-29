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
    "prior_adj.npy",
    "weighted_adj.npy",
    "prior_adj_counts.npy",
    "prior_required.json",
    "prior_forbidden.json",
    "age_weights.json",
    "volume_stats.json",
    "R_mask.npy",
    "prior_analysis.txt",
)

_OPTIONAL_FILES: Sequence[str] = (
    "symmetry_stats.json",
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
        if fname in {"prior_adj.npy", "weighted_adj.npy", "prior_adj_counts.npy"}:
            num_classes = _require_square_matrix(fpath, num_classes, issues)
        elif fname == "R_mask.npy":
            _check_binary_mask(fpath, num_classes, issues)
        elif fname == "prior_required.json":
            _check_required_edges(fpath, num_classes, issues, "required")
        elif fname == "prior_forbidden.json":
            _check_required_edges(fpath, num_classes, issues, "forbidden")
        elif fname == "volume_stats.json":
            _check_volume_stats(fpath, num_classes, issues)
        elif fname == "age_weights.json":
            _check_age_weights(fpath, num_classes, issues)
        else:
            # Text report, no structural validation needed
            pass

    if include_optional:
        for fname in _OPTIONAL_FILES:
            fpath = os.path.join(prior_dir, fname)
            if not os.path.exists(fpath):
                missing_optional.append(fname)
                continue
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
