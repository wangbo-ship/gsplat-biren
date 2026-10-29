# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_supa_rasterization_module():
    path = Path(__file__).resolve().parents[1] / "benchmark" / "supa_rasterization.py"
    spec = importlib.util.spec_from_file_location("benchmark.supa_rasterization_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == "torch_br":
            pytest.skip("torch_br is not installed")
        raise
    return module


def test_supa_rasterization_benchmark_uses_single_tiled_bwd_mode_by_default():
    module = _load_supa_rasterization_module()
    args = argparse.Namespace(tiled_bwd_mode="auto", tiled_bwd_modes=None)

    assert module._tiled_bwd_modes(args) == ["auto"]


def test_supa_rasterization_benchmark_parses_multiple_tiled_bwd_modes():
    module = _load_supa_rasterization_module()
    args = argparse.Namespace(
        tiled_bwd_mode="auto",
        tiled_bwd_modes="pixel,auto,parallel",
    )

    assert module._tiled_bwd_modes(args) == ["pixel", "auto", "parallel"]


def test_supa_rasterization_benchmark_rejects_unknown_tiled_bwd_mode():
    module = _load_supa_rasterization_module()
    args = argparse.Namespace(tiled_bwd_mode="auto", tiled_bwd_modes="auto,bad")

    with pytest.raises(ValueError, match="unknown tiled backward mode"):
        module._tiled_bwd_modes(args)
