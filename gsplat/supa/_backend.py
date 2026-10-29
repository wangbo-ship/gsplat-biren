# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JIT loader for optional native SUPA kernels."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import torch

from gsplat._device import ensure_supa_runtime


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BUILD_DIR = os.path.join(
    _REPO_ROOT,
    ".agents",
    "temporary",
    "native-build",
    "supa_native_build",
)


@lru_cache(maxsize=1)
def load_supa_backend() -> Optional[object]:
    """Build and load the SUPA extension, returning None when unavailable."""
    if not hasattr(torch, "supa"):
        ensure_supa_runtime()

    try:
        from torch_br.utils.cpp_extension import load
        from torch_br.supa._adapter import get_br_include_paths
        import torch_br
    except ImportError:
        return None

    torch_br_lib = os.path.join(os.path.dirname(os.path.abspath(torch_br.__file__)), "lib")
    sources = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.su"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "binding.cpp"),
    ]
    os.makedirs(_BUILD_DIR, exist_ok=True)
    try:
        return load(
            name="gsplat_supa",
            sources=sources,
            build_directory=_BUILD_DIR,
            extra_include_paths=get_br_include_paths(),
            extra_ldflags=[
                f"-L{torch_br_lib}",
                "-ltorch_br",
                f"-Wl,-rpath,{torch_br_lib}",
            ],
            verbose=os.getenv("VERBOSE", "0") == "1",
            with_supa=True,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent path
        import warnings

        warnings.warn(
            f"gsplat SUPA native extension unavailable, falling back to PyTorch: {exc}",
            stacklevel=2,
        )
        return None
