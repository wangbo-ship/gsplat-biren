#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the gsplat-biren native SUPA training diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_LOCAL = REPO_ROOT / ".agents" / "temporary" / "python-local"
if str(PYTHON_LOCAL) not in sys.path and PYTHON_LOCAL.exists():
    sys.path.insert(0, str(PYTHON_LOCAL))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from compare_backend_training_common import add_bool_arg, add_common_args, run_training  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        default_data=str(
            REPO_ROOT
            / ".agents"
            / "dataset"
            / "minecraft_dungle_pyramid_undistorted"
        ),
        default_result_dir=str(
            REPO_ROOT
            / ".agents"
            / "output"
            / "compare"
            / "minecraft_dungle_pyramid_undistorted_supa_native"
        ),
        default_device="supa:0",
    )
    parser.add_argument("--native-tiled", choices=("false", "true", "auto"), default="false")
    add_bool_arg(
        parser,
        "--native-tiled-metadata",
        default=False,
        help="Enable native int32-lane tiled metadata.",
    )
    add_bool_arg(
        parser,
        "--native-tiled-train",
        default=False,
        help="Enable native tiled training composite.",
    )
    parser.add_argument(
        "--native-alpha-threshold",
        type=float,
        default=0.0,
        help="Optional alpha cutoff for native tiled metadata; 0 keeps the exact SUPA path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.supa_native_options = {
        "native": True,
        "native_tiled": args.native_tiled,
        "native_tiled_metadata": bool(args.native_tiled_metadata),
        "native_tiled_train": bool(args.native_tiled_train),
        "native_alpha_threshold": float(args.native_alpha_threshold),
    }

    import pycolmap  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
    import torch  # pylint: disable=import-outside-toplevel
    import torch_br  # pylint: disable=import-outside-toplevel,unused-import
    import gsplat  # pylint: disable=import-outside-toplevel
    from gsplat import rasterization  # pylint: disable=import-outside-toplevel

    if not hasattr(torch, "supa") or not torch.supa.is_available():
        raise RuntimeError("torch_br imported successfully, but torch.supa is not available.")
    run_training(
        args,
        backend_name="supa_native",
        gsplat_module=gsplat,
        rasterization_fn=rasterization,
        extra_environment={
            "torch_br_version": getattr(torch_br, "__version__", None),
            "supa_available": bool(torch.supa.is_available()),
            "supa_device_count": int(torch.supa.device_count()),
            "native_options": args.supa_native_options,
        },
    )


if __name__ == "__main__":
    main()
