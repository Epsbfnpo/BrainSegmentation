#!/usr/bin/env python3
"""Preprocess large shape template archives into cached ROI-sized tensors."""

import argparse
import os
import sys
from typing import Optional, Tuple

import torch

# Reuse the existing implementation inside graph_prior_loss
from graph_prior_loss import (
    AgeConditionedGraphPriorLoss,
    _derive_shape_template_cache_path,
    _resolve_dtype,
)


def _parse_shape(arg: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not arg:
        return None
    tokens = str(arg).lower().replace("x", " ").replace(",", " ").split()
    if len(tokens) != 3:
        raise argparse.ArgumentTypeError(
            "target shape must be provided as three integers, e.g. 239x290x290"
        )
    try:
        dims = tuple(int(v) for v in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "target shape must contain integers only"
        ) from exc
    if any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError("target shape dimensions must be positive")
    return dims


def _maybe_resolve_output_path(
    input_path: str,
    target_shape: Optional[Tuple[int, int, int]],
    dtype: torch.dtype,
    explicit_output: Optional[str],
) -> str:
    if explicit_output:
        return explicit_output
    return _derive_shape_template_cache_path(input_path, target_shape, dtype)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess age-aware shape templates ahead of training."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw shape_templates.pt archive",
    )
    parser.add_argument(
        "--target-shape",
        type=_parse_shape,
        default=None,
        help="Spatial ROI to resample templates to (e.g. 239x290x290)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Output dtype for cached templates (float32, float16, bfloat16)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of CPU workers (0 = auto ≈ 80% of available cores)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit cache path; defaults to derived .processed.pt next to the input",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable verbose progress logging during preprocessing",
    )

    args = parser.parse_args(argv)

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        parser.error(f"input shape template archive not found: {input_path}")

    dtype = _resolve_dtype(args.dtype)
    cache_path = _maybe_resolve_output_path(
        input_path, args.target_shape, dtype, args.output
    )

    # Instantiate the loss module with shape prior enabled so it performs preprocessing.
    print(f"🚀 Preprocessing shape templates from {input_path}")
    print(f"   Target cache: {cache_path}")

    _ = AgeConditionedGraphPriorLoss(
        shape_templates_path=input_path,
        shape_template_target_shape=args.target_shape,
        shape_template_dtype=args.dtype,
        lambda_volume=0.0,
        lambda_shape=0.0,
        lambda_weighted_adj=0.0,
        lambda_topo=0.0,
        lambda_sym=0.0,
        lambda_spec=0.0,
        lambda_edge=0.0,
        lambda_dyn=0.0,
        shape_template_workers=args.workers,
        shape_template_progress=not args.no_progress,
    )

    if not os.path.exists(cache_path):
        print(
            "⚠️  Completed without locating the expected cache file. Check permissions and input format.",
            file=sys.stderr,
        )
        return 1

    print("✅ Shape templates ready for training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
