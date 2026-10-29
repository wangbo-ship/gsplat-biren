# SPDX-License-Identifier: Apache-2.0

"""Combined correctness and speed guard for native SUPA rasterization operators."""

from __future__ import annotations

import math
import shutil
import statistics
import time
from typing import Any, Callable

import pytest
import torch

from gsplat._device import synchronize
from gsplat.rendering import rasterization


_FALLBACK_OPTIONS: dict[str, object] = {
    "native": False,
    "native_tiled": "false",
    "native_tiled_metadata": False,
    "native_tiled_train": False,
    "native_tiled_backward_mode": "auto",
    "native_alpha_threshold": 0.0,
}

_NATIVE_FULL_SCAN_OPTIONS: dict[str, object] = {
    "native": True,
    "native_tiled": "false",
    "native_tiled_metadata": False,
    "native_tiled_train": False,
    "native_tiled_backward_mode": "auto",
    "native_alpha_threshold": 0.0,
}

_NATIVE_TILED_TRAIN_OPTIONS: dict[str, object] = {
    "native": True,
    "native_tiled": "auto",
    "native_tiled_metadata": True,
    "native_tiled_train": True,
    "native_tiled_backward_mode": "parallel",
    # Keep exact metadata in the guard test. The trainer's native profile uses
    # 1/255 by default, but exact metadata makes native-vs-fallback comparisons
    # a much tighter functional regression check.
    "native_alpha_threshold": 0.0,
}

_GRAD_KEYS = ("means", "quats", "scales", "opacities", "colors")
_RASTER_WIDTH = 64
_RASTER_HEIGHT = 48
_RASTER_TILE_SIZE = 16


def _require_native_supa() -> torch.device:
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pytest.skip("torch_br is not installed")

    if not hasattr(torch, "supa") or not torch.supa.is_available():
        pytest.skip("SUPA device is not available")
    if int(torch.supa.device_count()) < 1:
        pytest.skip("SUPA device is not available")
    if shutil.which("brcc") is None:
        pytest.skip("BRCC must be on PATH for SUPA native operator tests")

    torch.supa.set_device(0)
    return torch.device("supa:0")


def _time_sync(
    fn: Callable[[], dict[str, Any]],
    device: torch.device,
    *,
    warmup: int = 1,
    iters: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    last: dict[str, Any] | None = None
    for _ in range(warmup):
        last = fn()
        synchronize(device)

    samples: list[float] = []
    for _ in range(iters):
        synchronize(device)
        started_at = time.perf_counter()
        last = fn()
        synchronize(device)
        samples.append(time.perf_counter() - started_at)

    assert last is not None
    return {
        "median": float(statistics.median(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
        "samples": samples,
    }, last


def _make_raster_scene() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260724)
    n_gaussians = 128

    means = torch.empty((n_gaussians, 3), dtype=torch.float32)
    means[:, 0] = (torch.rand(n_gaussians, generator=generator) - 0.5) * 1.4
    means[:, 1] = (torch.rand(n_gaussians, generator=generator) - 0.5) * 1.0
    means[:, 2] = 2.0 + torch.rand(n_gaussians, generator=generator) * 2.0
    # Force a stable invisible subset so compact-select kernels are exercised.
    means[::8, 2] = -0.25
    means[1::8, 0] += 6.0
    means[2::8, 1] -= 5.0

    quats = torch.randn((n_gaussians, 4), generator=generator, dtype=torch.float32)
    quats[:, 0] += 2.0
    scales = 0.012 + torch.rand((n_gaussians, 3), generator=generator) * 0.018
    opacities = 0.08 + torch.rand(n_gaussians, generator=generator) * 0.35
    colors = torch.rand((n_gaussians, 4, 3), generator=generator, dtype=torch.float32)
    colors[:, 0] += 0.25

    viewmats = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    Ks = torch.tensor(
        [
            [float(_RASTER_WIDTH), 0.0, _RASTER_WIDTH * 0.5],
            [0.0, float(_RASTER_WIDTH), _RASTER_HEIGHT * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    return {
        "means": means.contiguous(),
        "quats": quats.contiguous(),
        "scales": scales.contiguous(),
        "opacities": opacities.contiguous(),
        "colors": colors.contiguous(),
        "viewmats": viewmats.contiguous(),
        "Ks": Ks.contiguous(),
    }


def _clone_to_device(
    base: dict[str, torch.Tensor],
    device: torch.device,
    grad_keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    scene: dict[str, torch.Tensor] = {}
    for key, tensor in base.items():
        value = tensor.to(device).contiguous().detach().clone()
        if key in grad_keys:
            value.requires_grad_(True)
        scene[key] = value
    return scene


def _run_rasterization_case(
    base: dict[str, torch.Tensor],
    device: torch.device,
    native_options: dict[str, object],
    *,
    sync_after: bool,
) -> dict[str, Any]:
    scene = _clone_to_device(base, device, _GRAD_KEYS)
    with torch.enable_grad():
        render, alpha, meta = rasterization(
            scene["means"],
            scene["quats"],
            scene["scales"],
            scene["opacities"],
            scene["colors"],
            scene["viewmats"],
            scene["Ks"],
            width=_RASTER_WIDTH,
            height=_RASTER_HEIGHT,
            packed=False,
            tile_size=_RASTER_TILE_SIZE,
            render_mode="RGB",
            sh_degree=1,
            supa_native_options=native_options,
            supa_use_torch_fallback=True,
        )
        loss = render.square().mean() + alpha.square().mean()
        loss.backward()

    if sync_after:
        synchronize(device)
    return {
        "render": render.detach(),
        "alpha": alpha.detach(),
        "loss": float(loss.detach().cpu().item()),
        "meta": meta,
        "grads": {
            key: scene[key].grad.detach().clone()
            for key in _GRAD_KEYS
            if scene[key].grad is not None
        },
    }


def _assert_finite_tensors(result: dict[str, Any]) -> None:
    tensors = [
        value
        for value in result.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    ]
    tensors.extend(
        value
        for value in result.get("grads", {}).values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )
    for tensor in tensors:
        assert bool(torch.isfinite(tensor).all().detach().cpu().item())


def _assert_raster_meta(meta: dict[str, Any], expected_variant: str) -> None:
    assert meta.get("backend") == "torch_fallback"
    assert meta.get("projection_backend") == "supa_native"
    assert meta.get("sh_backend") == "supa_native"
    assert meta.get("compact_backend") == "supa_native"
    assert meta.get("tile_count_backend") == "supa_native"
    assert meta.get("composite_backend") == "supa_native"
    assert meta.get("composite_variant") == expected_variant
    assert int(meta.get("culled_gaussians", 0)) > 0
    if expected_variant == "tiled":
        assert meta.get("tiled_metadata_backend") == "supa_native_offsets"


def _assert_raster_matches(
    native: dict[str, Any],
    reference: dict[str, Any],
    *,
    expected_variant: str,
) -> None:
    _assert_finite_tensors(native)
    _assert_raster_meta(native["meta"], expected_variant)
    assert reference["meta"].get("projection_backend") == "torch"
    assert reference["meta"].get("sh_backend") == "torch"
    assert reference["meta"].get("compact_backend") == "torch"
    assert reference["meta"].get("composite_backend") == "torch"
    assert native["meta"].get("visible_gaussians") == reference["meta"].get(
        "visible_gaussians"
    )
    assert native["meta"].get("culled_gaussians") == reference["meta"].get(
        "culled_gaussians"
    )

    torch.testing.assert_close(
        native["render"].cpu(),
        reference["render"].cpu(),
        rtol=1e-2,
        atol=1e-3,
    )
    torch.testing.assert_close(
        native["alpha"].cpu(),
        reference["alpha"].cpu(),
        rtol=1e-2,
        atol=1e-3,
    )
    for key in _GRAD_KEYS:
        torch.testing.assert_close(
            native["grads"][key].cpu(),
            reference["grads"][key].cpu(),
            rtol=2e-2,
            atol=2e-3,
        )


def _assert_speed_guard(
    case_name: str,
    native_stats: dict[str, Any],
    reference_stats: dict[str, Any],
    *,
    max_slowdown: float,
) -> None:
    native_median = float(native_stats["median"])
    reference_median = float(reference_stats["median"])
    assert math.isfinite(native_median) and native_median > 0.0
    assert math.isfinite(reference_median) and reference_median > 0.0
    budget = max(0.10, reference_median * max_slowdown)
    assert native_median <= budget, (
        f"{case_name} native operator path is too slow: "
        f"native median={native_median:.6f}s, "
        f"reference median={reference_median:.6f}s, budget={budget:.6f}s, "
        f"native samples={native_stats['samples']}, "
        f"reference samples={reference_stats['samples']}"
    )


@pytest.mark.parametrize(
    ("case_name", "max_slowdown"),
    [
        pytest.param("raster_full_scan", 25.0, id="raster-full-scan"),
        pytest.param("raster_tiled_train", 35.0, id="raster-tiled-train"),
    ],
)
def test_supa_native_operator_guard_correctness_and_speed(
    case_name: str,
    max_slowdown: float,
) -> None:
    """Catch native operator functional drift while keeping a coarse speed guard."""

    device = _require_native_supa()
    if case_name == "raster_full_scan":
        base = _make_raster_scene()
        native_runner = lambda sync_after: _run_rasterization_case(
            base,
            device,
            _NATIVE_FULL_SCAN_OPTIONS,
            sync_after=sync_after,
        )
        reference_runner = lambda sync_after: _run_rasterization_case(
            base,
            device,
            _FALLBACK_OPTIONS,
            sync_after=sync_after,
        )
        expected_variant = "full_scan"
    elif case_name == "raster_tiled_train":
        base = _make_raster_scene()
        native_runner = lambda sync_after: _run_rasterization_case(
            base,
            device,
            _NATIVE_TILED_TRAIN_OPTIONS,
            sync_after=sync_after,
        )
        reference_runner = lambda sync_after: _run_rasterization_case(
            base,
            device,
            _FALLBACK_OPTIONS,
            sync_after=sync_after,
        )
        expected_variant = "tiled"
    else:  # pragma: no cover - parameter table controls this.
        raise AssertionError(f"unknown operator guard case {case_name!r}")

    native_probe = native_runner(True)
    reference_probe = reference_runner(True)
    _assert_raster_matches(
        native_probe,
        reference_probe,
        expected_variant=expected_variant,
    )

    native_stats, _ = _time_sync(lambda: native_runner(False), device)
    reference_stats, _ = _time_sync(lambda: reference_runner(False), device)
    _assert_speed_guard(
        case_name,
        native_stats,
        reference_stats,
        max_slowdown=max_slowdown,
    )
