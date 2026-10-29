#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the gsplat-biren SUPA fallback training diagnostic."""

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

import pycolmap  # noqa: E402  # pylint: disable=wrong-import-position,unused-import
import torch  # noqa: E402  # pylint: disable=wrong-import-position
import torch_br  # noqa: E402  # pylint: disable=wrong-import-position,unused-import

import gsplat  # noqa: E402  # pylint: disable=wrong-import-position
from gsplat import rasterization  # noqa: E402  # pylint: disable=wrong-import-position

from compare_backend_training_common import add_common_args, run_training  # noqa: E402


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
            / "minecraft_dungle_pyramid_undistorted_supa_fallback"
        ),
        default_device="supa:0",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.supa_native_options = {
        "native": False,
        "native_tiled": "false",
        "native_tiled_metadata": False,
        "native_tiled_train": False,
    }
    if not hasattr(torch, "supa") or not torch.supa.is_available():
        raise RuntimeError("torch_br imported successfully, but torch.supa is not available.")
    run_training(
        args,
        backend_name="supa_fallback",
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
