# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the Biren SUPA PyTorch rasterization fallback."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


def _supa_device() -> torch.device:
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pytest.skip("torch_br is not installed")

    if not hasattr(torch, "supa") or not torch.supa.is_available():
        pytest.skip("SUPA device is not available")
    return torch.device("supa:0")


def _make_basic_scene(device: torch.device | str):
    tensors = {
        "means": [
            [-0.2, -0.1, 2.0],
            [0.15, 0.05, 2.4],
            [0.0, 0.2, 3.0],
        ],
        "quats": [
            [1.0, 0.0, 0.0, 0.0],
            [0.98, 0.05, 0.1, 0.02],
            [0.9, 0.2, 0.1, 0.05],
        ],
        "scales": [
            [0.08, 0.07, 0.09],
            [0.1, 0.08, 0.07],
            [0.06, 0.09, 0.08],
        ],
        "opacities": [0.7, 0.55, 0.65],
        "colors": [
            [0.9, 0.1, 0.2],
            [0.2, 0.8, 0.1],
            [0.1, 0.3, 0.9],
        ],
    }
    scene = {
        name: torch.tensor(value, dtype=torch.float32).to(device)
        for name, value in tensors.items()
    }
    for tensor in scene.values():
        tensor.requires_grad_(True)

    scene["viewmats"] = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).to(device)
    scene["Ks"] = torch.tensor(
        [[[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    ).to(device)
    return scene


def _make_two_camera_scene(device: torch.device | str):
    scene = _make_basic_scene(device)
    viewmat0 = torch.eye(4, dtype=torch.float32).to(device)
    viewmat1 = torch.eye(4, dtype=torch.float32).to(device)
    viewmat1[0, 3] = 0.08
    scene["viewmats"] = torch.stack((viewmat0, viewmat1), dim=0)
    scene["Ks"] = torch.tensor(
        [
            [[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]],
            [[7.5, 0.0, 3.8], [0.0, 8.5, 4.2], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    ).to(device)
    return scene


def _test_env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _test_native_options_from_env() -> dict | None:
    env_names = [
        "GSPLAT_USE_SUPA_NATIVE",
        "GSPLAT_USE_SUPA_NATIVE_COMPACT",
        "GSPLAT_USE_SUPA_NATIVE_COMPOSITE",
        "GSPLAT_USE_SUPA_NATIVE_COVARIANCE",
        "GSPLAT_USE_SUPA_NATIVE_PROJECTION",
        "GSPLAT_USE_SUPA_NATIVE_SH",
        "GSPLAT_USE_SUPA_NATIVE_TILE_COUNT",
        "GSPLAT_USE_SUPA_NATIVE_TILED",
        "GSPLAT_USE_SUPA_NATIVE_TILED_METADATA",
        "GSPLAT_USE_SUPA_NATIVE_TILED_TRAIN",
    ]
    if not any(name in os.environ for name in env_names):
        return None
    native = bool(_test_env_bool("GSPLAT_USE_SUPA_NATIVE"))
    tiled_value = os.getenv("GSPLAT_USE_SUPA_NATIVE_TILED", "false").strip().lower()
    if tiled_value in {"1", "true", "yes", "on"}:
        tiled = "true"
    elif tiled_value == "auto":
        tiled = "auto"
    else:
        tiled = "false"
    options: dict[str, object] = {
        "native": native,
        "native_tiled": tiled,
        "native_tiled_metadata": bool(_test_env_bool("GSPLAT_USE_SUPA_NATIVE_TILED_METADATA")),
        "native_tiled_train": bool(_test_env_bool("GSPLAT_USE_SUPA_NATIVE_TILED_TRAIN")),
        "native_tiled_backward_mode": "auto",
        "native_alpha_threshold": 0.0,
    }
    feature_env = {
        "compact": "GSPLAT_USE_SUPA_NATIVE_COMPACT",
        "composite": "GSPLAT_USE_SUPA_NATIVE_COMPOSITE",
        "covariance": "GSPLAT_USE_SUPA_NATIVE_COVARIANCE",
        "projection": "GSPLAT_USE_SUPA_NATIVE_PROJECTION",
        "sh": "GSPLAT_USE_SUPA_NATIVE_SH",
        "tile_count": "GSPLAT_USE_SUPA_NATIVE_TILE_COUNT",
    }
    for feature, env_name in feature_env.items():
        value = _test_env_bool(env_name)
        if value is not None:
            options[f"native_{feature}"] = value
    return options


def _call_public_rasterization(scene, **kwargs):
    from gsplat import rasterization

    means = kwargs.pop("means", scene["means"])
    quats = kwargs.pop("quats", scene["quats"])
    scales = kwargs.pop("scales", scene["scales"])
    opacities = kwargs.pop("opacities", scene["opacities"])
    colors = kwargs.pop("colors", scene["colors"])
    viewmats = kwargs.pop("viewmats", scene["viewmats"])
    Ks = kwargs.pop("Ks", scene["Ks"])
    width = kwargs.pop("width", 8)
    height = kwargs.pop("height", 8)
    kwargs.setdefault("supa_native_options", _test_native_options_from_env())
    return rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width=width,
        height=height,
        packed=False,
        **kwargs,
    )


def _call_reference_fallback(scene, **kwargs):
    from gsplat._torch_fallback import rasterization as torch_fallback_rasterization

    width = kwargs.pop("width", 8)
    height = kwargs.pop("height", 8)
    options = {
        "colors": scene["colors"],
        "sh_degree": None,
        "backgrounds": None,
        "render_mode": "RGB+D",
        "rasterize_mode": "classic",
        "covars": None,
        "extra_signals": None,
        "extra_signals_sh_degree": None,
        "chunk_size": 2,
        "camera_model": "pinhole",
        "supa_native_options": _test_native_options_from_env(),
    }
    options.update(kwargs)
    return torch_fallback_rasterization(
        means=scene["means"],
        quats=scene["quats"],
        scales=scene["scales"],
        opacities=scene["opacities"],
        colors=options.pop("colors"),
        viewmats=scene["viewmats"],
        Ks=scene["Ks"],
        width=width,
        height=height,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        eps2d=0.3,
        sh_degree=options.pop("sh_degree"),
        tile_size=16,
        backgrounds=options.pop("backgrounds"),
        render_mode=options.pop("render_mode"),
        rasterize_mode=options.pop("rasterize_mode"),
        camera_model=options.pop("camera_model"),
        covars=options.pop("covars"),
        extra_signals=options.pop("extra_signals"),
        extra_signals_sh_degree=options.pop("extra_signals_sh_degree"),
        **options,
    )


def _assert_full_covar_grad_matches_triu(
    full_grad: torch.Tensor,
    triu_grad: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    tri = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
    offdiag = ([1, 2, 4], [1, 2, 2], [0, 0, 1])
    full_grad = full_grad.cpu()
    combined = full_grad[..., tri[0], tri[1]].clone()
    combined[..., offdiag[0]] += full_grad[..., offdiag[1], offdiag[2]]
    torch.testing.assert_close(combined, triu_grad, rtol=rtol, atol=atol)


def test_supa_native_options_normalize_tiled_backward_mode_names():
    from gsplat._torch_fallback import _native_options, _native_tiled_backward_mode

    options = _native_options(
        {
            "native": True,
            "native_tiled_backward_mode": "tile-reduce",
        }
    )

    assert options["native_tiled_backward_mode"] == "tile_reduce"
    assert _native_tiled_backward_mode(options) == "tile_reduce"
    assert (
        _native_tiled_backward_mode({"native_tiled_backward_mode": "tile_reduce"})
        == "tile_reduce"
    )
    assert (
        _native_tiled_backward_mode({"native_tiled_backward_mode": "parallel"})
        == "parallel"
    )
    assert (
        _native_tiled_backward_mode({"native_tiled_backward_mode": "unknown"})
        == "auto"
    )


def test_supa_tiled_backward_parallel_uses_stable_tile_reduce_beyond_auto_gate(monkeypatch):
    import gsplat.supa._wrapper as wrapper

    class FakeBackend:
        def __init__(self):
            self.calls = []

        def rasterize_to_pixels_tiled_bwd(self, *args):
            self.calls.append("pixel")
            return self._zeros(*args)

        def rasterize_to_pixels_tiled_bwd_tile_reduce(self, *args):
            self.calls.append("tile_reduce")
            return self._zeros(*args)

        @staticmethod
        def _zeros(*args):
            means2d, conics, features, opacities, backgrounds = args[:5]
            return (
                torch.zeros_like(means2d),
                torch.zeros_like(conics),
                torch.zeros_like(features),
                torch.zeros_like(opacities),
                torch.zeros_like(backgrounds) if backgrounds is not None else None,
            )

    fake = FakeBackend()
    monkeypatch.setattr(wrapper, "load_supa_backend", lambda: fake)
    monkeypatch.setattr(
        wrapper,
        "is_supa_device",
        lambda value: isinstance(value, torch.Tensor),
    )

    means2d = torch.zeros((1, 2, 2), dtype=torch.float32)
    conics = torch.zeros((1, 2, 3), dtype=torch.float32)
    features = torch.zeros((1, 2, 3), dtype=torch.float32)
    opacities = torch.zeros((1, 2), dtype=torch.float32)
    tile_offsets = torch.zeros((1, 1, 1), dtype=torch.int32)
    v_render = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    def run(flatten_count: int, *, backward_mode: str):
        flatten_ids = torch.zeros((flatten_count,), dtype=torch.int32)
        return wrapper.rasterize_to_pixels_tiled_bwd(
            means2d,
            conics,
            features,
            opacities,
            1,
            1,
            16,
            tile_offsets,
            flatten_ids,
            None,
            v_render,
            None,
            None,
            backward_mode=backward_mode,
        )

    assert run(
        wrapper._TILED_BWD_AUTO_TILE_REDUCE_MAX_RANGE,
        backward_mode="parallel",
    ) is not None
    assert fake.calls[-1] == "tile_reduce"

    assert run(
        wrapper._TILED_BWD_AUTO_TILE_REDUCE_MAX_RANGE + 1,
        backward_mode="parallel",
    ) is not None
    assert fake.calls[-1] == "tile_reduce"

    assert run(
        wrapper._TILED_BWD_AUTO_TILE_REDUCE_MAX_RANGE + 1,
        backward_mode="auto",
    ) is not None
    assert fake.calls[-1] == "pixel"


def test_torch_fallback_compacts_invisible_gaussians_preserves_meta_shape():
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pass

    scene = _make_basic_scene("cpu")
    offscreen = {
        "means": torch.tensor([[100.0, 0.0, 2.0]], dtype=torch.float32),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "scales": torch.tensor([[0.02, 0.02, 0.02]], dtype=torch.float32),
        "opacities": torch.tensor([0.8], dtype=torch.float32),
        "colors": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
    }
    for name, value in offscreen.items():
        scene[name] = torch.cat((scene[name].detach(), value), dim=0)
        scene[name].requires_grad_(True)

    render, _, meta = _call_reference_fallback(scene, render_mode="RGB")
    render.sum().backward()

    assert meta["visible_gaussians"] == 3
    assert meta["culled_gaussians"] == 1
    assert tuple(meta["radii"].shape) == (1, 4, 2)
    assert tuple(meta["tiles_per_gauss"].shape) == (1, 4)
    assert int(meta["tiles_per_gauss"][0, -1].item()) == 0
    assert torch.all(scene["means"].grad[-1] == 0)


def test_supa_compact_select_backward_avoids_large_index_add_failure():
    supa_device = _supa_device()

    from gsplat._torch_fallback import _compact_select

    n_gaussians = 65535
    ids = torch.arange(0, n_gaussians, 2, dtype=torch.long, device=supa_device)
    x2 = torch.randn(
        (1, n_gaussians, 2), dtype=torch.float32, device=supa_device, requires_grad=True
    )
    x3 = torch.randn(
        (1, n_gaussians, 3), dtype=torch.float32, device=supa_device, requires_grad=True
    )

    loss = _compact_select(x2, ids, -2).sum() + _compact_select(x3, ids, -2).sum()
    loss.backward()
    torch.supa.synchronize()

    assert x2.grad is not None
    assert x3.grad is not None
    assert tuple(x2.grad.shape) == (1, n_gaussians, 2)
    assert tuple(x3.grad.shape) == (1, n_gaussians, 3)
    torch.testing.assert_close(
        x2.grad[0, 0].cpu(),
        torch.ones(2),
    )
    torch.testing.assert_close(
        x2.grad[0, 1].cpu(),
        torch.zeros(2),
    )
    torch.testing.assert_close(
        x3.grad[0, 0].cpu(),
        torch.ones(3),
    )
    torch.testing.assert_close(
        x3.grad[0, 1].cpu(),
        torch.zeros(3),
    )


def test_supa_native_visible_compact_ids_match_reference(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa._wrapper import visible_compact_ids

    radii = torch.tensor(
        [
            [[1, 1], [0, 1], [1, 1], [1, 1], [0, 0]],
            [[0, 0], [0, 0], [2, 1], [1, 1], [3, 3]],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [
            [1.0, 2.0, float("inf"), 3.0, 4.0],
            [5.0, 6.0, 7.0, float("nan"), 8.0],
        ],
        dtype=torch.float32,
    )
    valid = (radii > 0).all(dim=-1) & torch.isfinite(depths)
    expected = torch.nonzero(valid.any(dim=0), as_tuple=False).flatten()

    native = visible_compact_ids(
        radii.to(supa_device).contiguous(),
        depths.to(supa_device).contiguous(),
    )
    if native is None:
        pytest.skip("SUPA native visible compact is not available")
    compact_ids, visible_count = native

    assert visible_count == expected.numel()
    torch.testing.assert_close(
        torch.sort(compact_ids.cpu().to(torch.long)).values,
        expected,
        rtol=0,
        atol=0,
    )


def test_supa_native_compact_select_forward_backward_matches_reference(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa._wrapper import compact_select

    x_cpu = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    ids_cpu = torch.tensor([4, 0, 2], dtype=torch.long)
    weight_cpu = torch.linspace(0.1, 1.0, 2 * 3 * 3, dtype=torch.float32).reshape(
        2, 3, 3
    )
    x_ref = x_cpu.clone().requires_grad_(True)
    y_ref = x_ref.index_select(1, ids_cpu)
    (y_ref * weight_cpu).sum().backward()

    x_supa = x_cpu.to(supa_device).contiguous().requires_grad_(True)
    ids_supa = ids_cpu.to(torch.int32).to(supa_device).contiguous()
    native = compact_select(x_supa, ids_supa, 1)
    if native is None:
        pytest.skip("SUPA native compact select is not available")
    (native * weight_cpu.to(supa_device)).sum().backward()
    torch.supa.synchronize()

    torch.testing.assert_close(native.cpu(), y_ref, rtol=0, atol=0)
    torch.testing.assert_close(x_supa.grad.cpu(), x_ref.grad, rtol=1e-6, atol=1e-6)


def test_supa_native_compact_select_backward_handles_large_tensors(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_COMPACT", "1")

    from gsplat.supa._wrapper import compact_select

    n_gaussians = 65535
    ids = torch.arange(0, n_gaussians, 3, dtype=torch.int32, device=supa_device)
    x2 = torch.randn(
        (1, n_gaussians, 2), dtype=torch.float32, device=supa_device, requires_grad=True
    )
    x3 = torch.randn(
        (1, n_gaussians, 3), dtype=torch.float32, device=supa_device, requires_grad=True
    )

    y2 = compact_select(x2, ids, -2)
    y3 = compact_select(x3, ids, -2)
    if y2 is None or y3 is None:
        pytest.skip("SUPA native compact select is not available")
    (y2.sum() + y3.sum()).backward()
    torch.supa.synchronize()

    assert x2.grad is not None
    assert x3.grad is not None
    torch.testing.assert_close(x2.grad[0, 0].cpu(), torch.ones(2))
    torch.testing.assert_close(x2.grad[0, 1].cpu(), torch.zeros(2))
    torch.testing.assert_close(x3.grad[0, 0].cpu(), torch.ones(3))
    torch.testing.assert_close(x3.grad[0, 1].cpu(), torch.zeros(3))


def test_supa_native_compact_path_preserves_public_metadata_shape(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_COMPACT", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_COVARIANCE", "0")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_PROJECTION", "0")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_COMPOSITE", "0")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILE_COUNT", "0")

    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)
    offscreen = {
        "means": torch.tensor([[100.0, 0.0, 2.0]], dtype=torch.float32),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "scales": torch.tensor([[0.02, 0.02, 0.02]], dtype=torch.float32),
        "opacities": torch.tensor([0.8], dtype=torch.float32),
        "colors": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
    }
    for name, value in offscreen.items():
        cpu_scene[name] = torch.cat((cpu_scene[name].detach(), value), dim=0)
        cpu_scene[name].requires_grad_(True)
        supa_scene[name] = torch.cat(
            (supa_scene[name].detach(), value.to(supa_device)), dim=0
        ).contiguous()
        supa_scene[name].requires_grad_(True)

    cpu_render, cpu_alpha, _ = _call_reference_fallback(cpu_scene, render_mode="RGB")
    supa_render, supa_alpha, supa_meta = _call_public_rasterization(
        supa_scene,
        render_mode="RGB",
        channel_chunk=2,
    )
    if supa_meta["compact_backend"] != "supa_native":
        pytest.skip("SUPA native compact path is not available")

    assert supa_meta["visible_gaussians"] == 3
    assert supa_meta["culled_gaussians"] == 1
    assert tuple(supa_meta["radii"].shape) == (1, 4, 2)
    assert tuple(supa_meta["tiles_per_gauss"].shape) == (1, 4)
    assert int(supa_meta["tiles_per_gauss"][0, -1].item()) == 0
    torch.testing.assert_close(supa_render.cpu(), cpu_render, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(supa_alpha.cpu(), cpu_alpha, rtol=1e-3, atol=1e-4)

    (cpu_render.sum() + cpu_alpha.sum()).backward()
    (supa_render.sum() + supa_alpha.sum()).backward()
    _assert_scene_grads_close(
        cpu_scene,
        supa_scene,
        ("means", "quats", "scales", "opacities", "colors"),
    )
    assert torch.all(supa_scene["means"].grad[-1].cpu() == 0)


def test_supa_native_alpha_composite_backward_workspace_is_chunked():
    kernel_source = (
        Path(__file__).resolve().parents[1] / "gsplat" / "supa" / "kernels.su"
    ).read_text(encoding="utf-8")
    binding_source = (
        Path(__file__).resolve().parents[1] / "gsplat" / "supa" / "binding.cpp"
    ).read_text(encoding="utf-8")

    assert "int workspace_base = idx * prefix_checkpoint_count" not in kernel_source
    assert "int workspace_base = local_idx * prefix_checkpoint_count" in kernel_source
    assert "int idx = pixel_offset + local_idx" in kernel_source
    assert "kAlphaCompositeBwdWorkspaceTargetBytes" in binding_source
    assert "at::empty({chunk_pixels, prefix_checkpoint_count}" in binding_source
    assert "for (int64_t pixel_offset = 0; pixel_offset < total;" in binding_source


def test_supa_tiled_backward_launch_uses_tuned_block_size():
    kernel_source = (
        Path(__file__).resolve().parents[1] / "gsplat" / "supa" / "kernels.su"
    ).read_text(encoding="utf-8")
    launch_source = kernel_source.split(
        'extern "C" suError_t gsplat_rasterize_to_pixels_tiled_bwd_launch',
        maxsplit=1,
    )[1].split(
        'extern "C" __global__ void gsplat_alpha_composite_bwd_kernel',
        maxsplit=1,
    )[0]

    assert "int threads = 64;" in launch_source


def test_supa_tiled_backward_uses_prefix_checkpoint_workspace():
    kernel_source = (
        Path(__file__).resolve().parents[1] / "gsplat" / "supa" / "kernels.su"
    ).read_text(encoding="utf-8")
    binding_source = (
        Path(__file__).resolve().parents[1] / "gsplat" / "supa" / "binding.cpp"
    ).read_text(encoding="utf-8")
    launch_source = kernel_source.split(
        'extern "C" __global__ void gsplat_rasterize_to_pixels_tiled_bwd_kernel',
        maxsplit=1,
    )[1].split(
        'extern "C" suError_t gsplat_rasterize_to_pixels_tiled_bwd_launch',
        maxsplit=1,
    )[0]

    assert "prefix_checkpoints" in launch_source
    assert "int workspace_base = local_idx * prefix_checkpoint_count" in launch_source
    assert "for (int prev = range_start; prev < chunk_start;" not in launch_source
    assert "rasterize_to_pixels_tiled_bwd prefix checkpoint workspace is too large" in binding_source


def test_supa_tiled_backward_has_tile_reduce_rgb_path():
    kernel_source = (
        Path(__file__).resolve().parents[1] / "gsplat" / "supa" / "kernels.su"
    ).read_text(encoding="utf-8")
    binding_source = (
        Path(__file__).resolve().parents[1] / "gsplat" / "supa" / "binding.cpp"
    ).read_text(encoding="utf-8")

    assert "gsplat_rasterize_to_pixels_tiled_bwd_tile_reduce_kernel" in kernel_source
    assert "tile_reduce_atomic_add_rgb_grads" in kernel_source
    assert "GSPLAT_TILED_BWD_TILE_REDUCE_SIZE = 16" in kernel_source
    assert "forward_alpha != nullptr" in kernel_source
    assert "tile_has_low_transmittance" not in kernel_source
    assert "tile-reduce tiled backward currently supports RGB channels only" in binding_source
    assert "rasterize_to_pixels_tiled_bwd_tile_reduce" in binding_source


def test_supa_projection_autograd_backward_avoids_host_sync():
    wrapper_source = (
        Path(__file__).resolve().parents[1] / "gsplat" / "supa" / "_wrapper.py"
    ).read_text(encoding="utf-8")
    autograd_source = wrapper_source.split(
        "class _ProjectionEWA3DGSPinhole",
        maxsplit=1,
    )[1].split(
        "def projection_ewa_3dgs_pinhole",
        maxsplit=1,
    )[0]

    assert "synchronize_launch=False" in autograd_source
    assert "synchronize_launch: bool = True" in wrapper_source


def test_supa_native_alpha_composite_backward_chunked_large_workspace_smoke():
    supa_device = _supa_device()

    from gsplat.supa._wrapper import alpha_composite_bwd

    n_images, n_gaussians, height, width, channels = 1, 65536, 128, 128, 1
    means2d = torch.zeros((n_images, n_gaussians, 2), dtype=torch.float32, device=supa_device)
    conics = torch.zeros((n_images, n_gaussians, 3), dtype=torch.float32, device=supa_device)
    features = torch.zeros(
        (n_images, n_gaussians, channels), dtype=torch.float32, device=supa_device
    )
    opacities = torch.zeros((n_images, n_gaussians), dtype=torch.float32, device=supa_device)
    depths = torch.zeros((n_images, n_gaussians), dtype=torch.float32, device=supa_device)
    radii = torch.zeros((n_images, n_gaussians, 2), dtype=torch.int32, device=supa_device)
    sorted_ids = torch.full(
        (n_images, n_gaussians), -1, dtype=torch.int32, device=supa_device
    )
    v_render = torch.zeros(
        (n_images, height, width, channels), dtype=torch.float32, device=supa_device
    )
    v_alpha = torch.zeros((n_images, height, width, 1), dtype=torch.float32, device=supa_device)

    grads = alpha_composite_bwd(
        means2d.contiguous(),
        conics.contiguous(),
        features.contiguous(),
        opacities.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        sorted_ids.contiguous(),
        width,
        height,
        None,
        v_render.contiguous(),
        v_alpha.contiguous(),
    )
    if grads is None:
        pytest.skip("SUPA native composite backward is not available")
    assert [tuple(grad.shape) if grad is not None else None for grad in grads] == [
        (n_images, n_gaussians, 2),
        (n_images, n_gaussians, 3),
        (n_images, n_gaussians, channels),
        (n_images, n_gaussians),
        (n_images, n_gaussians),
        None,
    ]


def _assert_scene_grads_close(cpu_scene, supa_scene, names, *, rtol=2e-3, atol=1e-3):
    for name in names:
        assert supa_scene[name].grad is not None
        torch.testing.assert_close(
            supa_scene[name].grad.cpu(),
            cpu_scene[name].grad,
            rtol=rtol,
            atol=atol,
        )


def _make_sh_colors(scene):
    colors = torch.zeros(
        scene["colors"].shape[:-1] + (4, 3),
        dtype=scene["colors"].dtype,
        device=scene["colors"].device,
    )
    colors[..., 0, :] = scene["colors"].detach() - 0.5
    colors.requires_grad_(True)
    return colors


def _make_extra_signals(scene):
    extra = torch.tensor(
        [[0.1, 0.2], [0.4, 0.1], [0.3, 0.7]],
        dtype=scene["means"].dtype,
        device=scene["means"].device,
    )
    extra.requires_grad_(True)
    return extra


def _make_garden_subset_scene(device: torch.device | str, *, num_points: int = 128):
    data_path = (
        Path(__file__).resolve().parents[1]
        / ".agents"
        / "dataset"
        / "test_garden"
        / "test_garden.npz"
    )
    if not data_path.exists():
        pytest.skip(f"optional garden test data is missing: {data_path}")

    from gsplat._helper import load_test_data
    from gsplat._math import _quat_scale_to_covar_preci
    from gsplat._torch_impl import _fully_fused_projection

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        (
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            orig_width,
            orig_height,
        ) = load_test_data(
            device="cpu",
            data_path=str(data_path),
        )
    width, height, num_cameras = 32, 24, 2
    viewmats = viewmats[:num_cameras].contiguous()
    Ks = Ks[:num_cameras].clone().contiguous()
    Ks[:, 0, :] *= width / orig_width
    Ks[:, 1, :] *= height / orig_height

    covars, _ = _quat_scale_to_covar_preci(quats, scales, True, False, triu=False)
    radii, _, _, _, _ = _fully_fused_projection(
        means,
        covars,
        viewmats,
        Ks,
        width,
        height,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        calc_compensations=False,
        camera_model="pinhole",
    )
    visible_ids = torch.where((radii > 0).all(dim=-1).any(dim=0))[0][:num_points]
    assert visible_ids.numel() == num_points

    scene = {
        "means": means[visible_ids].contiguous(),
        "quats": quats[visible_ids].contiguous(),
        "scales": scales[visible_ids].contiguous(),
        "opacities": opacities[visible_ids].contiguous(),
        "colors": colors[visible_ids].contiguous(),
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
    }
    for key in ("means", "quats", "scales", "opacities", "colors"):
        scene[key] = scene[key].to(device).requires_grad_(True)
    scene["viewmats"] = scene["viewmats"].to(device)
    scene["Ks"] = scene["Ks"].to(device)
    return scene


def test_alpha_composite_avoids_cumprod_and_is_chunk_stable(monkeypatch):
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pass

    from gsplat._torch_fallback import _alpha_composite

    def fail_cumprod(*args, **kwargs):
        raise AssertionError("fallback alpha compositing should not call torch.cumprod")

    monkeypatch.setattr(torch, "cumprod", fail_cumprod)

    base = {
        "means2d": torch.tensor(
            [[[2.0, 2.2], [2.8, 2.5], [1.7, 3.1]]], dtype=torch.float32
        ),
        "conics": torch.tensor(
            [[[0.8, 0.05, 0.9], [1.1, -0.04, 0.7], [0.6, 0.02, 1.2]]],
            dtype=torch.float32,
        ),
        "features": torch.tensor(
            [[[0.9, 0.1], [0.2, 0.7], [0.4, 0.5]]], dtype=torch.float32
        ),
        "opacities": torch.tensor([[0.65, 0.55, 0.45]], dtype=torch.float32),
    }
    depths = torch.tensor([[2.0, 1.5, 2.8]], dtype=torch.float32)
    radii = torch.tensor([[[2, 2], [2, 2], [2, 2]]], dtype=torch.int32)

    def run(chunk_size):
        tensors = {
            name: value.clone().requires_grad_(True) for name, value in base.items()
        }
        renders, alphas = _alpha_composite(
            tensors["means2d"],
            tensors["conics"],
            tensors["features"],
            tensors["opacities"],
            depths,
            radii,
            width=5,
            height=4,
            backgrounds=None,
            chunk_size=chunk_size,
        )
        (renders.sum() + alphas.sum()).backward()
        grads = {name: tensor.grad.clone() for name, tensor in tensors.items()}
        return renders, alphas, grads

    renders_one, alphas_one, grads_one = run(chunk_size=1)
    renders_many, alphas_many, grads_many = run(chunk_size=3)

    torch.testing.assert_close(renders_many, renders_one)
    torch.testing.assert_close(alphas_many, alphas_one)
    for name in grads_one:
        torch.testing.assert_close(grads_many[name], grads_one[name])


def test_supa_fallback_matches_cpu_reference_forward_backward():
    supa_device = _supa_device()
    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(cpu_scene, chunk_size=2)
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene, render_mode="RGB+D", channel_chunk=2
    )

    assert supa_meta["backend"] == "torch_fallback"
    assert supa_renders.shape == (1, 8, 8, 4)
    assert supa_alphas.shape == (1, 8, 8, 1)
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=1e-3, atol=1e-4)

    (cpu_renders.sum() + cpu_alphas.sum()).backward()
    (supa_renders.sum() + supa_alphas.sum()).backward()

    _assert_scene_grads_close(
        cpu_scene, supa_scene, ("means", "quats", "scales", "opacities", "colors")
    )


@pytest.mark.parametrize(
    "name,kwargs,color_grad",
    [
        ("rgb", {"render_mode": "RGB"}, True),
        ("depth", {"render_mode": "D", "colors": None}, False),
        ("expected_depth", {"render_mode": "ED", "colors": None}, False),
        ("rgb_expected_depth", {"render_mode": "RGB+ED"}, True),
        ("antialiased_rgb", {"render_mode": "RGB", "rasterize_mode": "antialiased"}, True),
        (
            "rgb_background",
            {
                "render_mode": "RGB",
                "backgrounds": torch.tensor([[0.05, 0.1, 0.15]], dtype=torch.float32),
            },
            True,
        ),
    ],
)
def test_supa_fallback_basic_render_options_match_cpu(name, kwargs, color_grad):
    del name
    supa_device = _supa_device()
    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)
    cpu_kwargs = dict(kwargs)
    supa_kwargs = dict(kwargs)
    if "backgrounds" in kwargs:
        supa_kwargs["backgrounds"] = kwargs["backgrounds"].to(supa_device)

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(cpu_scene, **cpu_kwargs)
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene,
        channel_chunk=2,
        **supa_kwargs,
    )

    assert supa_meta["backend"] == "torch_fallback"
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=1e-3, atol=1e-4)

    (cpu_renders.sum() + cpu_alphas.sum()).backward()
    (supa_renders.sum() + supa_alphas.sum()).backward()

    grad_names = ["means", "quats", "scales", "opacities"]
    if color_grad:
        grad_names.append("colors")
    _assert_scene_grads_close(cpu_scene, supa_scene, grad_names)


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_supa_native_spherical_harmonics_matches_torch_reference(monkeypatch, degree):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_SH", "1")

    from gsplat._torch_impl import _spherical_harmonics
    from gsplat.supa._wrapper import spherical_harmonics

    torch.manual_seed(1234 + degree)
    dirs_cpu = torch.randn(2, 4, 3, dtype=torch.float32).requires_grad_(True)
    coeffs_cpu = torch.randn(2, 4, 25, 3, dtype=torch.float32).requires_grad_(True)
    v_colors_cpu = torch.randn(2, 4, 3, dtype=torch.float32)

    dirs_supa = dirs_cpu.detach().to(supa_device).contiguous().requires_grad_(True)
    coeffs_supa = coeffs_cpu.detach().to(supa_device).contiguous().requires_grad_(True)
    v_colors_supa = v_colors_cpu.to(supa_device).contiguous()

    native = spherical_harmonics(degree, dirs_supa, coeffs_supa)
    if native is None:
        pytest.skip("SUPA native spherical harmonics is not available")
    ref = _spherical_harmonics(degree, dirs_cpu, coeffs_cpu)

    torch.testing.assert_close(native.cpu(), ref, rtol=1e-4, atol=1e-4)

    (native * v_colors_supa).sum().backward()
    (ref * v_colors_cpu).sum().backward()

    torch.testing.assert_close(
        coeffs_supa.grad.cpu(), coeffs_cpu.grad, rtol=1e-4, atol=1e-4
    )
    if degree > 0:
        torch.testing.assert_close(
            dirs_supa.grad.cpu(), dirs_cpu.grad, rtol=1e-4, atol=1e-4
        )


def test_supa_fallback_sh_matches_cpu_reference_forward_backward(monkeypatch):
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "0")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_SH", "1")
    supa_device = _supa_device()
    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)
    cpu_colors = _make_sh_colors(cpu_scene)
    supa_colors = _make_sh_colors(supa_scene)

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene, colors=cpu_colors, sh_degree=1, render_mode="RGB"
    )
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene,
        colors=supa_colors,
        sh_degree=1,
        render_mode="RGB",
        channel_chunk=2,
    )

    assert supa_meta["backend"] == "torch_fallback"
    if supa_meta["sh_backend"] != "supa_native":
        pytest.skip("SUPA native spherical harmonics is not available")
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=1e-3, atol=1e-4)

    (cpu_renders.sum() + cpu_alphas.sum()).backward()
    (supa_renders.sum() + supa_alphas.sum()).backward()

    _assert_scene_grads_close(
        cpu_scene, supa_scene, ("means", "quats", "scales", "opacities")
    )
    torch.testing.assert_close(
        supa_colors.grad.cpu(), cpu_colors.grad, rtol=2e-3, atol=1e-3
    )


def test_supa_fallback_extra_signals_match_cpu_reference_forward_backward():
    supa_device = _supa_device()
    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)
    cpu_extra = _make_extra_signals(cpu_scene)
    supa_extra = _make_extra_signals(supa_scene)

    cpu_renders, cpu_alphas, cpu_meta = _call_reference_fallback(
        cpu_scene, extra_signals=cpu_extra, render_mode="RGB"
    )
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene,
        extra_signals=supa_extra,
        render_mode="RGB",
        channel_chunk=2,
    )

    assert supa_meta["backend"] == "torch_fallback"
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(
        supa_meta["render_extra_signals"].cpu(),
        cpu_meta["render_extra_signals"],
        rtol=1e-3,
        atol=1e-4,
    )

    cpu_loss = (
        cpu_renders.sum()
        + cpu_alphas.sum()
        + cpu_meta["render_extra_signals"].sum()
    )
    supa_loss = (
        supa_renders.sum()
        + supa_alphas.sum()
        + supa_meta["render_extra_signals"].sum()
    )
    cpu_loss.backward()
    supa_loss.backward()

    _assert_scene_grads_close(
        cpu_scene, supa_scene, ("means", "quats", "scales", "opacities", "colors")
    )
    torch.testing.assert_close(
        supa_extra.grad.cpu(), cpu_extra.grad, rtol=2e-3, atol=1e-3
    )


def test_supa_fallback_explicit_covars_match_cpu_reference_forward_backward():
    supa_device = _supa_device()
    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)

    from gsplat._math import _quat_scale_to_covar_preci

    cpu_covars, _ = _quat_scale_to_covar_preci(
        cpu_scene["quats"].detach(),
        cpu_scene["scales"].detach(),
        True,
        False,
        triu=False,
    )
    supa_covars, _ = _quat_scale_to_covar_preci(
        supa_scene["quats"].detach(),
        supa_scene["scales"].detach(),
        True,
        False,
        triu=False,
    )
    tri = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
    cpu_covars_triu = cpu_covars[..., tri[0], tri[1]].detach().requires_grad_(True)
    supa_covars.requires_grad_(True)

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene, covars=cpu_covars_triu, render_mode="RGB"
    )
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene,
        covars=supa_covars,
        render_mode="RGB",
        channel_chunk=2,
    )

    assert supa_meta["backend"] == "torch_fallback"
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=1e-3, atol=1e-4)

    (cpu_renders.sum() + cpu_alphas.sum()).backward()
    (supa_renders.sum() + supa_alphas.sum()).backward()

    _assert_scene_grads_close(cpu_scene, supa_scene, ("means", "opacities", "colors"))
    _assert_full_covar_grad_matches_triu(
        supa_covars.grad,
        cpu_covars_triu.grad,
        rtol=2e-3,
        atol=1e-3,
    )


def test_supa_fallback_rejects_unsupported_packed_mode():
    supa_scene = _make_basic_scene(_supa_device())
    from gsplat import rasterization

    with pytest.raises(NotImplementedError, match="packed=True"):
        rasterization(
            supa_scene["means"],
            supa_scene["quats"],
            supa_scene["scales"],
            supa_scene["opacities"],
            supa_scene["colors"],
            supa_scene["viewmats"],
            supa_scene["Ks"],
            width=8,
            height=8,
            packed=True,
        )


def test_supa_public_surface_capability_flags_and_removed_stubs():
    import gsplat

    assert gsplat.has_3dgs() is True
    assert gsplat.has_2dgs() is False
    assert gsplat.has_3dgut() is False
    assert gsplat.has_adam() is False
    assert gsplat.has_camera_wrappers() is False
    assert gsplat.has_reloc() is False

    with pytest.raises(NotImplementedError, match="rasterization_2dgs"):
        gsplat.rasterization_2dgs()
    with pytest.raises(NotImplementedError, match="SUPA-only"):
        gsplat.UnscentedTransformParameters()
    assert gsplat.BivariateWindshieldModelParameters.MAX_ORDER == 5
    assert gsplat.BivariateWindshieldModelParameters.MAX_COEFFS == 21
    assert gsplat.ExternalDistortionReferencePolynomial.FORWARD.value == 1
    assert gsplat.FThetaPolynomialType.ANGLE_TO_PIXELDIST.value == 1


@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
def test_supa_public_rasterization_camera_models_match_cpu(camera_model):
    supa_device = _supa_device()
    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene,
        render_mode="RGB+D",
        camera_model=camera_model,
    )
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene,
        render_mode="RGB+D",
        camera_model=camera_model,
        channel_chunk=2,
    )

    assert supa_meta["backend"] == "torch_fallback"
    if camera_model != "pinhole":
        assert supa_meta["projection_backend"] == "torch"
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=1e-3, atol=1e-4)

    (cpu_renders.sum() + cpu_alphas.sum()).backward()
    (supa_renders.sum() + supa_alphas.sum()).backward()
    _assert_scene_grads_close(
        cpu_scene,
        supa_scene,
        ("means", "quats", "scales", "opacities", "colors"),
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda scene: {"means": scene["means"][..., :2]}, "means"),
        (lambda scene: {"quats": scene["quats"][..., :3]}, "quats"),
        (lambda scene: {"scales": scene["scales"][..., :2]}, "scales"),
        (lambda scene: {"opacities": scene["opacities"].unsqueeze(-1)}, "opacities"),
        (lambda scene: {"colors": scene["colors"][:2]}, "colors"),
        (lambda scene: {"viewmats": scene["viewmats"][..., :3, :4]}, "viewmats"),
        (lambda scene: {"Ks": scene["Ks"][..., :2, :3]}, "Ks"),
        (lambda scene: {"backgrounds": torch.zeros((2, 3))}, "backgrounds"),
        (lambda scene: {"tile_size": 0}, "tile_size"),
        (lambda scene: {"channel_chunk": 0}, "channel_chunk"),
        (lambda scene: {"rasterize_mode": "invalid"}, "rasterize_mode"),
    ],
)
def test_supa_public_rasterization_rejects_invalid_surface_inputs(mutate, match):
    scene = _make_basic_scene("cpu")
    kwargs = mutate(scene)
    with pytest.raises(ValueError, match=match):
        _call_public_rasterization(scene, **kwargs)


def test_supa_public_rasterization_rejects_invalid_sh_surface_inputs():
    scene = _make_basic_scene("cpu")
    bad_sh = torch.zeros((3, 3, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="at least 4 SH coefficients"):
        _call_public_rasterization(scene, colors=bad_sh, sh_degree=1)


def test_supa_native_quat_scale_kernel_matches_torch_explicit_grad(monkeypatch):
    _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._math import _quat_scale_to_covar_preci
    from gsplat.supa._wrapper import quat_scale_to_covar_preci

    q_supa = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.98, 0.05, 0.1, 0.02]],
        dtype=torch.float32,
        device="supa",
        requires_grad=True,
    )
    s_supa = torch.tensor(
        [[0.08, 0.07, 0.09], [0.1, 0.08, 0.07]],
        dtype=torch.float32,
        device="supa",
        requires_grad=True,
    )
    native = quat_scale_to_covar_preci(q_supa, s_supa, True, True, False)
    if native is None:
        pytest.skip("SUPA native extension is not available")
    covar_supa, preci_supa = native

    q_cpu = q_supa.detach().cpu().requires_grad_(True)
    s_cpu = s_supa.detach().cpu().requires_grad_(True)
    covar_cpu, preci_cpu = _quat_scale_to_covar_preci(
        q_cpu, s_cpu, True, True, False
    )

    torch.testing.assert_close(covar_supa.cpu(), covar_cpu, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(preci_supa.cpu(), preci_cpu, rtol=1e-5, atol=1e-4)

    (covar_supa.sum() + preci_supa.sum()).backward()
    (covar_cpu.sum() + preci_cpu.sum()).backward()

    torch.testing.assert_close(q_supa.grad.cpu(), q_cpu.grad, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(s_supa.grad.cpu(), s_cpu.grad, rtol=1e-5, atol=1e-3)


def test_supa_native_covariance_opt_in_keeps_rasterization_correct(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_PROJECTION", "0")
    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(cpu_scene, chunk_size=2)
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene, render_mode="RGB+D", channel_chunk=2
    )

    assert supa_meta["backend"] == "torch_fallback"
    assert supa_meta["projection_backend"] == "torch"
    assert supa_meta["covariance_backend"] == "supa_native"
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=1e-3, atol=1e-4)

    (cpu_renders.sum() + cpu_alphas.sum()).backward()
    (supa_renders.sum() + supa_alphas.sum()).backward()

    _assert_scene_grads_close(
        cpu_scene, supa_scene, ("means", "quats", "scales", "opacities", "colors")
    )


@pytest.mark.parametrize("input_kind", ["quats_scales", "covars"])
@pytest.mark.parametrize("rasterize_mode", ["classic", "antialiased"])
def test_supa_native_projection_opt_in_keeps_public_rasterization_correct(
    monkeypatch, input_kind, rasterize_mode
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._math import _quat_scale_to_covar_preci

    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)
    cpu_covars = None
    cpu_covars_triu = None
    supa_covars = None
    if input_kind == "covars":
        cpu_covars, _ = _quat_scale_to_covar_preci(
            cpu_scene["quats"].detach(),
            cpu_scene["scales"].detach(),
            True,
            False,
            triu=False,
        )
        tri = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        cpu_covars_triu = cpu_covars[..., tri[0], tri[1]].detach().requires_grad_(True)
        supa_covars = cpu_covars.detach().to(supa_device).contiguous().requires_grad_(True)

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene,
        covars=cpu_covars_triu,
        render_mode="RGB+D",
        rasterize_mode=rasterize_mode,
        chunk_size=2,
    )
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene,
        covars=supa_covars,
        render_mode="RGB+D",
        rasterize_mode=rasterize_mode,
        channel_chunk=2,
    )

    if supa_meta["projection_backend"] != "supa_native":
        pytest.skip("SUPA native projection is not available")
    assert supa_meta["backend"] == "torch_fallback"
    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["covariance_backend"] in (None, "provided")
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=3e-3, atol=2e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=3e-3, atol=2e-4)

    (cpu_renders.sum() + cpu_alphas.sum()).backward()
    (supa_renders.sum() + supa_alphas.sum()).backward()

    if input_kind == "covars":
        _assert_scene_grads_close(cpu_scene, supa_scene, ("means", "opacities", "colors"))
        assert supa_covars.grad is not None
        assert cpu_covars_triu is not None and cpu_covars_triu.grad is not None
        _assert_full_covar_grad_matches_triu(
            supa_covars.grad,
            cpu_covars_triu.grad,
            rtol=4e-3,
            atol=1e-3,
        )
    else:
        _assert_scene_grads_close(
            cpu_scene, supa_scene, ("means", "quats", "scales", "opacities", "colors")
        )


def test_supa_native_alpha_composite_forward_matches_torch_reference(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._torch_fallback import _alpha_composite
    from gsplat.supa._wrapper import alpha_composite_fwd

    means2d = torch.tensor(
        [[[2.0, 2.2], [2.8, 2.5], [1.7, 3.1]]], dtype=torch.float32
    )
    conics = torch.tensor(
        [[[0.8, 0.05, 0.9], [1.1, -0.04, 0.7], [0.6, 0.02, 1.2]]],
        dtype=torch.float32,
    )
    features = torch.tensor(
        [[[0.9, 0.1, 2.0], [0.2, 0.7, 1.5], [0.4, 0.5, 2.8]]],
        dtype=torch.float32,
    )
    opacities = torch.tensor([[0.65, 0.55, 0.45]], dtype=torch.float32)
    depths = torch.tensor([[2.0, 1.5, 2.8]], dtype=torch.float32)
    radii = torch.tensor([[[2, 2], [2, 2], [2, 2]]], dtype=torch.int32)

    ref_renders, ref_alphas = _alpha_composite(
        means2d,
        conics,
        features,
        opacities,
        depths,
        radii,
        width=5,
        height=4,
        backgrounds=None,
        chunk_size=2,
    )
    native = alpha_composite_fwd(
        means2d.to(supa_device).contiguous(),
        conics.to(supa_device).contiguous(),
        features.to(supa_device).contiguous(),
        opacities.to(supa_device).contiguous(),
        depths.to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        width=5,
        height=4,
    )
    if native is None:
        pytest.skip("SUPA native extension is not available")
    native_renders, native_alphas = native

    torch.testing.assert_close(native_renders.cpu(), ref_renders, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(native_alphas.cpu(), ref_alphas, rtol=1e-5, atol=1e-6)


def test_supa_native_alpha_composite_forward_matches_torch_reference_64_channels(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._torch_fallback import _alpha_composite
    from gsplat.supa._wrapper import alpha_composite_fwd

    channels = 64
    channel_t = torch.linspace(0.0, 1.0, channels, dtype=torch.float32)
    means2d = torch.tensor(
        [[[2.0, 2.2], [2.8, 2.5], [1.7, 3.1]]], dtype=torch.float32
    )
    conics = torch.tensor(
        [[[0.8, 0.05, 0.9], [1.1, -0.04, 0.7], [0.6, 0.02, 1.2]]],
        dtype=torch.float32,
    )
    features = torch.stack(
        (
            0.1 + 0.4 * channel_t,
            0.3 + 0.2 * torch.sin(3.0 * channel_t),
            0.6 - 0.3 * channel_t.square(),
        ),
        dim=0,
    ).unsqueeze(0)
    opacities = torch.tensor([[0.65, 0.55, 0.45]], dtype=torch.float32)
    depths = torch.tensor([[2.0, 1.5, 2.8]], dtype=torch.float32)
    radii = torch.tensor([[[2, 2], [2, 2], [2, 2]]], dtype=torch.int32)
    backgrounds = (0.02 + 0.03 * channel_t).unsqueeze(0)

    ref_renders, ref_alphas = _alpha_composite(
        means2d,
        conics,
        features,
        opacities,
        depths,
        radii,
        width=5,
        height=4,
        backgrounds=backgrounds,
        chunk_size=2,
    )
    native = alpha_composite_fwd(
        means2d.to(supa_device).contiguous(),
        conics.to(supa_device).contiguous(),
        features.to(supa_device).contiguous(),
        opacities.to(supa_device).contiguous(),
        depths.to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        width=5,
        height=4,
        backgrounds=backgrounds.to(supa_device).contiguous(),
    )
    if native is None:
        pytest.skip("SUPA native extension is not available")
    native_renders, native_alphas = native

    torch.testing.assert_close(native_renders.cpu(), ref_renders, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(native_alphas.cpu(), ref_alphas, rtol=1e-5, atol=1e-6)


def test_supa_native_alpha_composite_backward_matches_torch_reference(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._torch_fallback import _alpha_composite
    from gsplat.supa._wrapper import alpha_composite

    base = {
        "means2d": torch.tensor(
            [[[2.0, 2.2], [2.8, 2.5], [1.7, 3.1]]], dtype=torch.float32
        ),
        "conics": torch.tensor(
            [[[0.8, 0.05, 0.9], [1.1, -0.04, 0.7], [0.6, 0.02, 1.2]]],
            dtype=torch.float32,
        ),
        "features": torch.tensor(
            [[[0.9, 0.1, 2.0], [0.2, 0.7, 1.5], [0.4, 0.5, 2.8]]],
            dtype=torch.float32,
        ),
        "opacities": torch.tensor([[0.65, 0.55, 0.45]], dtype=torch.float32),
        "depths": torch.tensor([[2.0, 1.5, 2.8]], dtype=torch.float32),
    }
    radii = torch.tensor([[[2, 2], [2, 2], [2, 2]]], dtype=torch.int32)
    backgrounds = torch.tensor([[0.05, 0.1, 0.15]], dtype=torch.float32)

    cpu = {name: tensor.clone().requires_grad_(True) for name, tensor in base.items()}
    cpu_backgrounds = backgrounds.clone().requires_grad_(True)
    ref_renders, ref_alphas = _alpha_composite(
        cpu["means2d"],
        cpu["conics"],
        cpu["features"],
        cpu["opacities"],
        cpu["depths"],
        radii,
        width=5,
        height=4,
        backgrounds=cpu_backgrounds,
        chunk_size=2,
    )
    (ref_renders.sum() + 0.7 * ref_alphas.sum()).backward()

    supa = {
        name: tensor.to(supa_device).contiguous().requires_grad_(True)
        for name, tensor in base.items()
    }
    supa_backgrounds = backgrounds.to(supa_device).contiguous().requires_grad_(True)
    native = alpha_composite(
        supa["means2d"],
        supa["conics"],
        supa["features"],
        supa["opacities"],
        supa["depths"],
        radii.to(supa_device).contiguous(),
        width=5,
        height=4,
        backgrounds=supa_backgrounds,
    )
    if native is None:
        pytest.skip("SUPA native extension is not available")
    native_renders, native_alphas = native
    (native_renders.sum() + 0.7 * native_alphas.sum()).backward()

    torch.testing.assert_close(native_renders.cpu(), ref_renders, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(native_alphas.cpu(), ref_alphas, rtol=1e-5, atol=1e-6)
    for name in ("means2d", "conics", "features", "opacities"):
        torch.testing.assert_close(
            supa[name].grad.cpu(), cpu[name].grad, rtol=2e-5, atol=5e-6
        )
    assert cpu["depths"].grad is None
    assert supa["depths"].grad is None
    torch.testing.assert_close(
        supa_backgrounds.grad.cpu(),
        cpu_backgrounds.grad,
        rtol=2e-5,
        atol=5e-6,
    )


def test_supa_native_alpha_composite_backward_matches_torch_reference_checkpointed(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._torch_fallback import _alpha_composite
    from gsplat.supa._wrapper import alpha_composite

    num_points = 65
    channels = 4
    t = torch.linspace(0.0, 1.0, num_points, dtype=torch.float32)
    means2d = torch.stack(
        (
            2.0 + 0.7 * torch.sin(6.0 * t),
            2.0 + 0.6 * torch.cos(5.0 * t),
        ),
        dim=-1,
    ).unsqueeze(0)
    conics = torch.stack(
        (
            torch.full_like(t, 0.45),
            torch.linspace(-0.03, 0.03, num_points, dtype=torch.float32),
            torch.full_like(t, 0.55),
        ),
        dim=-1,
    ).unsqueeze(0)
    features = torch.stack(
        (
            0.2 + 0.3 * t,
            0.6 - 0.2 * t,
            0.1 + 0.4 * torch.sin(3.0 * t).square(),
            1.0 + t,
        ),
        dim=-1,
    ).unsqueeze(0)
    opacities = torch.linspace(0.015, 0.045, num_points, dtype=torch.float32).unsqueeze(0)
    depths = torch.linspace(1.0, 3.0, num_points, dtype=torch.float32).unsqueeze(0)
    radii = torch.full((1, num_points, 2), 3, dtype=torch.int32)
    backgrounds = torch.tensor([[0.04, 0.08, 0.12, 0.0]], dtype=torch.float32)

    cpu = {
        "means2d": means2d.clone().requires_grad_(True),
        "conics": conics.clone().requires_grad_(True),
        "features": features.clone().requires_grad_(True),
        "opacities": opacities.clone().requires_grad_(True),
    }
    cpu_backgrounds = backgrounds.clone().requires_grad_(True)
    ref_renders, ref_alphas = _alpha_composite(
        cpu["means2d"],
        cpu["conics"],
        cpu["features"],
        cpu["opacities"],
        depths,
        radii,
        width=5,
        height=4,
        backgrounds=cpu_backgrounds,
        chunk_size=17,
    )
    (0.7 * ref_renders.square().mean() + 0.3 * ref_alphas.square().mean()).backward()

    supa = {
        name: tensor.detach().to(supa_device).contiguous().requires_grad_(True)
        for name, tensor in cpu.items()
    }
    supa_backgrounds = backgrounds.to(supa_device).contiguous().requires_grad_(True)
    native = alpha_composite(
        supa["means2d"],
        supa["conics"],
        supa["features"],
        supa["opacities"],
        depths.to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        width=5,
        height=4,
        backgrounds=supa_backgrounds,
    )
    if native is None:
        pytest.skip("SUPA native extension is not available")
    native_renders, native_alphas = native
    (0.7 * native_renders.square().mean() + 0.3 * native_alphas.square().mean()).backward()

    torch.testing.assert_close(native_renders.cpu(), ref_renders, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(native_alphas.cpu(), ref_alphas, rtol=1e-5, atol=1e-6)
    for name in cpu:
        torch.testing.assert_close(
            supa[name].grad.cpu(), cpu[name].grad, rtol=2e-4, atol=2e-5
        )
    torch.testing.assert_close(
        supa_backgrounds.grad.cpu(),
        cpu_backgrounds.grad,
        rtol=2e-4,
        atol=2e-5,
    )


def test_supa_native_alpha_composite_prunes_invalid_sorted_tail(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._torch_fallback import _alpha_composite
    from gsplat.supa._wrapper import _make_composite_sorted_ids, alpha_composite

    means2d = torch.tensor(
        [[[2.0, 2.1], [2.6, 1.9], [1.8, 2.4], [2.3, 2.7], [2.1, 1.7], [1.6, 2.2]]],
        dtype=torch.float32,
    )
    conics = torch.tensor(
        [[[0.7, 0.02, 0.8], [0.9, -0.03, 0.6], [0.5, 0.04, 0.7],
          [0.8, 0.01, 0.9], [0.6, -0.02, 0.5], [0.75, 0.03, 0.65]]],
        dtype=torch.float32,
    )
    features = torch.tensor(
        [[[0.8, 0.1], [0.2, 0.5], [0.4, 0.7], [0.6, 0.2], [0.1, 0.9], [0.3, 0.4]]],
        dtype=torch.float32,
    )
    opacities = torch.tensor([[0.4, 0.7, 0.5, 0.6, 0.45, 0.35]], dtype=torch.float32)
    depths = torch.tensor([[2.0, 0.4, 1.0, float("nan"), 0.2, 1.5]], dtype=torch.float32)
    radii = torch.tensor(
        [[[2, 2], [0, 2], [2, 2], [2, 2], [-1, 2], [2, 2]]], dtype=torch.int32
    )
    backgrounds = torch.tensor([[0.03, 0.07]], dtype=torch.float32)

    sorted_ids = _make_composite_sorted_ids(
        depths.to(supa_device).contiguous(), radii.to(supa_device).contiguous()
    )
    torch.testing.assert_close(
        sorted_ids.cpu(),
        torch.tensor([[2, 5, 0, -1, -1, -1]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )

    cpu = {
        "means2d": means2d.clone().requires_grad_(True),
        "conics": conics.clone().requires_grad_(True),
        "features": features.clone().requires_grad_(True),
        "opacities": opacities.clone().requires_grad_(True),
    }
    cpu_backgrounds = backgrounds.clone().requires_grad_(True)
    ref_renders, ref_alphas = _alpha_composite(
        cpu["means2d"],
        cpu["conics"],
        cpu["features"],
        cpu["opacities"],
        depths,
        radii,
        width=5,
        height=4,
        backgrounds=cpu_backgrounds,
        chunk_size=2,
    )
    (0.6 * ref_renders.square().mean() + 0.4 * ref_alphas.square().mean()).backward()

    supa = {
        name: tensor.detach().to(supa_device).contiguous().requires_grad_(True)
        for name, tensor in cpu.items()
    }
    supa_backgrounds = backgrounds.to(supa_device).contiguous().requires_grad_(True)
    native = alpha_composite(
        supa["means2d"],
        supa["conics"],
        supa["features"],
        supa["opacities"],
        depths.to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        width=5,
        height=4,
        backgrounds=supa_backgrounds,
    )
    if native is None:
        pytest.skip("SUPA native extension is not available")
    native_renders, native_alphas = native
    (0.6 * native_renders.square().mean() + 0.4 * native_alphas.square().mean()).backward()

    torch.testing.assert_close(native_renders.cpu(), ref_renders, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(native_alphas.cpu(), ref_alphas, rtol=1e-5, atol=1e-6)
    for name in cpu:
        torch.testing.assert_close(
            supa[name].grad.cpu(), cpu[name].grad, rtol=2e-4, atol=2e-5
        )
    torch.testing.assert_close(
        supa_backgrounds.grad.cpu(), cpu_backgrounds.grad, rtol=2e-4, atol=2e-5
    )


def test_supa_native_tile_count_unpacked_matches_torch_reference(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._torch_fallback import _tiles_per_gauss
    from gsplat.supa._wrapper import tile_count_unpacked

    means2d = torch.tensor(
        [
            [[2.0, 2.0], [7.5, 1.5], [-2.0, 3.0], [4.0, 8.0]],
            [[1.0, 1.0], [3.5, 5.5], [8.5, 4.0], [6.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [
            [[2, 2], [3, 1], [2, 2], [0, 3]],
            [[1, 1], [4, 2], [2, 2], [3, 3]],
        ],
        dtype=torch.int32,
    )
    tile_size = 4
    tile_width = 3
    tile_height = 2
    ref = _tiles_per_gauss(means2d, radii, tile_size, tile_width, tile_height)

    native = tile_count_unpacked(
        means2d.to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        tile_size,
        tile_width,
        tile_height,
    )
    if native is None:
        pytest.skip("SUPA native tile count is not available")
    torch.testing.assert_close(native.cpu(), ref, rtol=0, atol=0)


@pytest.mark.parametrize("sort", [False, True])
def test_supa_native_isect_tiles_unpacked_matches_torch_reference(monkeypatch, sort):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._torch_impl import _isect_tiles
    from gsplat.supa._wrapper import isect_tiles_unpacked

    means2d = torch.tensor(
        [
            [[2.0, 2.0], [7.5, 1.5], [-2.0, 3.0], [4.0, 8.0]],
            [[1.0, 1.0], [3.5, 5.5], [8.5, 4.0], [6.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [
            [[2, 2], [3, 1], [2, 2], [0, 3]],
            [[1, 1], [4, 2], [2, 2], [3, 3]],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [
            [1.0, 0.5, 2.0, 1.5],
            [1.25, 0.75, 2.5, 0.25],
        ],
        dtype=torch.float32,
    )
    tile_size = 4
    tile_width = 3
    tile_height = 2

    ref_tiles, ref_isect_ids, ref_flatten_ids = _isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, sort=sort
    )
    native = isect_tiles_unpacked(
        means2d.to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        depths.to(supa_device).contiguous(),
        tile_size,
        tile_width,
        tile_height,
        sort=sort,
    )
    if native is None:
        pytest.skip("SUPA native tile intersection is not available")
    tiles_per_gauss, isect_ids, flatten_ids = native
    torch.testing.assert_close(tiles_per_gauss.cpu(), ref_tiles, rtol=0, atol=0)
    torch.testing.assert_close(isect_ids.cpu(), ref_isect_ids, rtol=0, atol=0)
    torch.testing.assert_close(flatten_ids.cpu(), ref_flatten_ids, rtol=0, atol=0)


def test_supa_isect_offset_encode_matches_torch_reference(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._torch_impl import _isect_offset_encode, _isect_tiles
    from gsplat.supa._wrapper import isect_offset_encode, isect_tiles_unpacked

    means2d = torch.tensor(
        [
            [[2.0, 2.0], [7.5, 1.5], [-2.0, 3.0], [4.0, 8.0]],
            [[1.0, 1.0], [3.5, 5.5], [8.5, 4.0], [6.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [
            [[2, 2], [3, 1], [2, 2], [0, 3]],
            [[1, 1], [4, 2], [2, 2], [3, 3]],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [
            [1.0, 0.5, 2.0, 1.5],
            [1.25, 0.75, 2.5, 0.25],
        ],
        dtype=torch.float32,
    )
    tile_size = 4
    tile_width = 3
    tile_height = 2
    n_images = means2d.shape[0]

    _, ref_isect_ids, _ = _isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, sort=True
    )
    ref_offsets = _isect_offset_encode(ref_isect_ids, n_images, tile_width, tile_height)
    native = isect_tiles_unpacked(
        means2d.to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        depths.to(supa_device).contiguous(),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )
    if native is None:
        pytest.skip("SUPA tile intersection prototype is not available")
    native_offsets = isect_offset_encode(native[1], n_images, tile_width, tile_height)
    if native_offsets is None:
        pytest.skip("SUPA tile offset prototype is not available")
    torch.testing.assert_close(native_offsets, ref_offsets, rtol=0, atol=0)


def test_supa_isect_tiles_unpacked_offsets_matches_torch_reference(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._torch_impl import _isect_offset_encode, _isect_tiles
    from gsplat.supa._wrapper import isect_tiles_unpacked_offsets

    means2d = torch.tensor(
        [
            [[2.0, 2.0], [7.5, 1.5], [-2.0, 3.0], [4.0, 8.0]],
            [[1.0, 1.0], [3.5, 5.5], [8.5, 4.0], [6.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [
            [[2, 2], [3, 1], [2, 2], [0, 3]],
            [[1, 1], [4, 2], [2, 2], [3, 3]],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [
            [1.0, 0.5, 2.0, 1.5],
            [1.25, 0.75, 2.5, 0.25],
        ],
        dtype=torch.float32,
    )
    tile_size = 4
    tile_width = 3
    tile_height = 2
    n_images = means2d.shape[0]

    ref_tiles, ref_isect_ids, ref_flatten_ids = _isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, sort=True
    )
    ref_offsets = _isect_offset_encode(ref_isect_ids, n_images, tile_width, tile_height)
    native = isect_tiles_unpacked_offsets(
        means2d.to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        depths.to(supa_device).contiguous(),
        tile_size,
        tile_width,
        tile_height,
    )
    if native is None:
        pytest.skip("SUPA tile offset prototype is not available")
    tiles_per_gauss, tile_offsets, flatten_ids = native
    torch.testing.assert_close(tiles_per_gauss.cpu(), ref_tiles, rtol=0, atol=0)
    torch.testing.assert_close(tile_offsets, ref_offsets, rtol=0, atol=0)
    torch.testing.assert_close(flatten_ids, ref_flatten_ids, rtol=0, atol=0)


def test_supa_isect_tiles_unpacked_offsets_alpha_threshold_reduces_metadata(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa import _wrapper as supa_wrapper

    means2d = torch.tensor([[[8.0, 8.0], [8.0, 8.0]]], dtype=torch.float32)
    radii = torch.tensor([[[8, 8], [8, 8]]], dtype=torch.int32)
    depths = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    conics = torch.tensor([[[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]], dtype=torch.float32)
    opacities = torch.tensor([[0.1, 0.001]], dtype=torch.float32)
    tile_size = 4
    tile_width = 4
    tile_height = 4

    supa_means2d = means2d.to(supa_device).contiguous()
    supa_radii = radii.to(supa_device).contiguous()
    supa_depths = depths.to(supa_device).contiguous()
    supa_conics = conics.to(supa_device).contiguous()
    supa_opacities = opacities.to(supa_device).contiguous()
    exact = supa_wrapper.isect_tiles_unpacked_offsets(
        supa_means2d,
        supa_radii,
        supa_depths,
        tile_size,
        tile_width,
        tile_height,
        return_backend=True,
    )
    if exact is None:
        pytest.skip("SUPA native tile metadata is not available")
    thresholded = supa_wrapper.isect_tiles_unpacked_offsets(
        supa_means2d,
        supa_radii,
        supa_depths,
        tile_size,
        tile_width,
        tile_height,
        return_backend=True,
        conics=supa_conics,
        opacities=supa_opacities,
        alpha_threshold=1.0 / 255.0,
    )
    if thresholded is None:
        pytest.skip("SUPA native alpha-threshold tile metadata is not available")

    exact_tiles, _, exact_flatten_ids, exact_backend = exact
    threshold_tiles, _, threshold_flatten_ids, threshold_backend = thresholded
    assert exact_backend == "cpu_numpy"
    assert threshold_backend == "cpu_numpy_alpha_threshold"
    assert int(exact_tiles.detach().cpu().sum().item()) == int(exact_flatten_ids.numel())
    assert int(threshold_tiles.detach().cpu().sum().item()) == int(
        threshold_flatten_ids.numel()
    )
    assert int(threshold_flatten_ids.numel()) < int(exact_flatten_ids.numel())
    assert int(threshold_flatten_ids.numel()) == 4


def test_supa_isect_tiles_unpacked_offsets_alpha_threshold_can_use_native_metadata(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa import _wrapper as supa_wrapper

    means2d = torch.tensor([[[8.0, 8.0], [8.0, 8.0]]], dtype=torch.float32)
    radii = torch.tensor([[[8, 8], [8, 8]]], dtype=torch.int32)
    depths = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    conics = torch.tensor([[[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]], dtype=torch.float32)
    opacities = torch.tensor([[0.1, 0.001]], dtype=torch.float32)
    tile_size = 4
    tile_width = 4
    tile_height = 4

    supa_means2d = means2d.to(supa_device).contiguous()
    supa_radii = radii.to(supa_device).contiguous()
    supa_depths = depths.to(supa_device).contiguous()
    supa_conics = conics.to(supa_device).contiguous()
    supa_opacities = opacities.to(supa_device).contiguous()
    cpu_ref = supa_wrapper.isect_tiles_unpacked_offsets(
        supa_means2d,
        supa_radii,
        supa_depths,
        tile_size,
        tile_width,
        tile_height,
        return_backend=True,
        conics=supa_conics,
        opacities=supa_opacities,
        alpha_threshold=1.0 / 255.0,
    )
    native = supa_wrapper.isect_tiles_unpacked_offsets(
        supa_means2d,
        supa_radii,
        supa_depths,
        tile_size,
        tile_width,
        tile_height,
        return_backend=True,
        use_native_offsets=True,
        conics=supa_conics,
        opacities=supa_opacities,
        alpha_threshold=1.0 / 255.0,
    )
    if cpu_ref is None or native is None:
        pytest.skip("SUPA native alpha-threshold tile metadata is not available")

    ref_tiles, ref_offsets, ref_flatten_ids, ref_backend = cpu_ref
    native_tiles, native_offsets, native_flatten_ids, native_backend = native
    assert ref_backend == "cpu_numpy_alpha_threshold"
    assert native_backend == "supa_native_alpha_threshold"
    torch.testing.assert_close(native_tiles.cpu(), ref_tiles.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(native_offsets.cpu(), ref_offsets.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(
        native_flatten_ids.cpu(),
        ref_flatten_ids.cpu(),
        rtol=0,
        atol=0,
    )


def test_supa_isect_tiles_unpacked_lanes_match_numpy_reference(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa._wrapper import (
        _expand_isect_tiles_unpacked_np,
        isect_tiles_unpacked_lanes,
        tile_count_unpacked,
    )

    means2d = torch.tensor(
        [
            [[2.0, 2.0], [7.5, 1.5], [-2.0, 3.0], [4.0, 8.0]],
            [[1.0, 1.0], [3.5, 5.5], [8.5, 4.0], [6.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [
            [[2, 2], [3, 1], [2, 2], [0, 3]],
            [[1, 1], [4, 2], [2, 2], [3, 3]],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [
            [1.0, 0.5, 2.0, 1.5],
            [1.25, 0.75, 2.5, 0.25],
        ],
        dtype=torch.float32,
    )
    tile_size = 4
    tile_width = 3
    tile_height = 2

    supa_means2d = means2d.to(supa_device).contiguous()
    supa_radii = radii.to(supa_device).contiguous()
    supa_depths = depths.to(supa_device).contiguous()
    tiles_per_gauss = tile_count_unpacked(
        supa_means2d, supa_radii, tile_size, tile_width, tile_height
    )
    if tiles_per_gauss is None:
        pytest.skip("SUPA native tile count is not available")
    lanes = isect_tiles_unpacked_lanes(
        supa_means2d,
        supa_radii,
        supa_depths,
        tiles_per_gauss,
        tile_size,
        tile_width,
        tile_height,
    )
    if lanes is None:
        pytest.skip("SUPA native tile lane expansion is not available")
    dense_tile_ids, flatten_ids, isect_ids_hi, isect_ids_lo = lanes

    (
        _,
        n_isects,
        ref_flatten_ids,
        ref_dense_tile_ids,
        ref_isect_ids_hi,
        ref_isect_ids_lo,
    ) = _expand_isect_tiles_unpacked_np(
        means2d,
        radii,
        depths,
        tiles_per_gauss.cpu(),
        tile_size,
        tile_width,
        tile_height,
    )
    assert dense_tile_ids.numel() == n_isects
    torch.testing.assert_close(
        dense_tile_ids, torch.from_numpy(ref_dense_tile_ids), rtol=0, atol=0
    )
    torch.testing.assert_close(
        flatten_ids, torch.from_numpy(ref_flatten_ids), rtol=0, atol=0
    )
    torch.testing.assert_close(
        isect_ids_hi, torch.from_numpy(ref_isect_ids_hi), rtol=0, atol=0
    )
    torch.testing.assert_close(
        isect_ids_lo, torch.from_numpy(ref_isect_ids_lo), rtol=0, atol=0
    )

    empty_tiles = torch.zeros_like(tiles_per_gauss[:1])
    empty_lanes = isect_tiles_unpacked_lanes(
        supa_means2d[:1].contiguous(),
        torch.zeros_like(supa_radii[:1]).contiguous(),
        supa_depths[:1].contiguous(),
        empty_tiles.contiguous(),
        tile_size,
        tile_width,
        tile_height,
    )
    if empty_lanes is None:
        pytest.skip("SUPA native tile lane expansion is not available")
    for tensor in empty_lanes:
        assert tensor.numel() == 0


def test_supa_isect_tiles_unpacked_lanes_can_use_native_starts(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa import _wrapper as supa_wrapper
    from gsplat.supa._backend import load_supa_backend

    backend = load_supa_backend()
    if backend is None or not hasattr(backend, "isect_tiles_unpacked_lanes_from_counts"):
        pytest.skip("SUPA native tile lane starts are not available")

    means2d = torch.tensor(
        [
            [[2.0, 2.0], [7.5, 1.5], [-2.0, 3.0], [4.0, 8.0]],
            [[1.0, 1.0], [3.5, 5.5], [8.5, 4.0], [6.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [
            [[2, 2], [3, 1], [2, 2], [0, 3]],
            [[1, 1], [4, 2], [2, 2], [3, 3]],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [
            [1.0, 0.5, 2.0, 1.5],
            [1.25, 0.75, 2.5, 0.25],
        ],
        dtype=torch.float32,
    )
    tile_size = 4
    tile_width = 3
    tile_height = 2

    supa_means2d = means2d.to(supa_device).contiguous()
    supa_radii = radii.to(supa_device).contiguous()
    supa_depths = depths.to(supa_device).contiguous()
    tiles_per_gauss = supa_wrapper.tile_count_unpacked(
        supa_means2d, supa_radii, tile_size, tile_width, tile_height
    )
    if tiles_per_gauss is None:
        pytest.skip("SUPA native tile count is not available")

    def _fail_cpu_starts(*_args, **_kwargs):
        raise AssertionError("CPU lane starts helper should not be used")

    monkeypatch.setattr(
        supa_wrapper,
        "_isect_tiles_unpacked_lanes_cpu_starts",
        _fail_cpu_starts,
    )

    lanes = supa_wrapper.isect_tiles_unpacked_lanes(
        supa_means2d,
        supa_radii,
        supa_depths,
        tiles_per_gauss,
        tile_size,
        tile_width,
        tile_height,
        return_cpu=False,
    )
    if lanes is None:
        pytest.skip("SUPA native tile lane expansion is not available")
    dense_tile_ids, flatten_ids, isect_ids_hi, isect_ids_lo = lanes

    (
        _,
        _,
        ref_flatten_ids,
        ref_dense_tile_ids,
        ref_isect_ids_hi,
        ref_isect_ids_lo,
    ) = supa_wrapper._expand_isect_tiles_unpacked_np(
        means2d,
        radii,
        depths,
        tiles_per_gauss.cpu(),
        tile_size,
        tile_width,
        tile_height,
    )
    torch.testing.assert_close(
        dense_tile_ids.cpu(), torch.from_numpy(ref_dense_tile_ids), rtol=0, atol=0
    )
    torch.testing.assert_close(
        flatten_ids.cpu(), torch.from_numpy(ref_flatten_ids), rtol=0, atol=0
    )
    torch.testing.assert_close(
        isect_ids_hi.cpu(), torch.from_numpy(ref_isect_ids_hi), rtol=0, atol=0
    )
    torch.testing.assert_close(
        isect_ids_lo.cpu(), torch.from_numpy(ref_isect_ids_lo), rtol=0, atol=0
    )


def test_supa_tiled_offsets_from_lanes_match_torch_reference(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._torch_impl import _isect_offset_encode, _isect_tiles
    from gsplat.supa._wrapper import (
        isect_tiles_unpacked_lanes,
        tile_count_unpacked,
        tiled_offsets_from_lanes,
    )

    means2d = torch.tensor(
        [
            [[2.0, 2.0], [2.5, 2.5], [2.25, 2.25], [7.5, 1.5], [5.5, 5.5]],
            [[1.0, 1.0], [3.5, 5.5], [8.5, 4.0], [6.0, -1.0], [2.5, 2.5]],
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [
            [[2, 2], [2, 2], [2, 2], [3, 1], [1, 1]],
            [[1, 1], [4, 2], [2, 2], [3, 3], [2, 2]],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [
            [1.0, 0.5, 0.625, 1.5, 2.0],
            [1.25, 0.75, 2.5, 0.25, 0.875],
        ],
        dtype=torch.float32,
    )
    tile_size = 4
    tile_width = 3
    tile_height = 2
    n_images = means2d.shape[0]

    ref_tiles, ref_isect_ids, ref_flatten_ids = _isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, sort=True
    )
    ref_offsets = _isect_offset_encode(ref_isect_ids, n_images, tile_width, tile_height)

    supa_means2d = means2d.to(supa_device).contiguous()
    supa_radii = radii.to(supa_device).contiguous()
    supa_depths = depths.to(supa_device).contiguous()
    tiles_per_gauss = tile_count_unpacked(
        supa_means2d, supa_radii, tile_size, tile_width, tile_height
    )
    if tiles_per_gauss is None:
        pytest.skip("SUPA native tile count is not available")
    lanes = isect_tiles_unpacked_lanes(
        supa_means2d,
        supa_radii,
        supa_depths,
        tiles_per_gauss,
        tile_size,
        tile_width,
        tile_height,
        return_cpu=False,
    )
    if lanes is None:
        pytest.skip("SUPA native tile lane expansion is not available")

    native_offsets = tiled_offsets_from_lanes(
        lanes[0],
        lanes[1],
        lanes[3],
        n_images,
        tile_width,
        tile_height,
    )
    if native_offsets is None:
        pytest.skip("SUPA native tiled offsets are not available")
    tile_offsets, flatten_ids = native_offsets

    torch.testing.assert_close(tiles_per_gauss.cpu(), ref_tiles, rtol=0, atol=0)
    torch.testing.assert_close(tile_offsets.cpu(), ref_offsets, rtol=0, atol=0)
    torch.testing.assert_close(flatten_ids.cpu(), ref_flatten_ids, rtol=0, atol=0)


def test_supa_isect_tiles_unpacked_offsets_can_use_lane_metadata(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILED_METADATA", "1")

    from gsplat._torch_impl import _isect_offset_encode, _isect_tiles
    from gsplat.supa import _wrapper as supa_wrapper

    means2d = torch.tensor(
        [
            [[2.0, 2.0], [7.5, 1.5], [-2.0, 3.0], [4.0, 8.0]],
            [[1.0, 1.0], [3.5, 5.5], [8.5, 4.0], [6.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [
            [[2, 2], [3, 1], [2, 2], [0, 3]],
            [[1, 1], [4, 2], [2, 2], [3, 3]],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [
            [1.0, 0.5, 2.0, 1.5],
            [1.25, 0.75, 2.5, 0.25],
        ],
        dtype=torch.float32,
    )
    tile_size = 4
    tile_width = 3
    tile_height = 2
    n_images = means2d.shape[0]

    supa_means2d = means2d.to(supa_device).contiguous()
    supa_radii = radii.to(supa_device).contiguous()
    supa_depths = depths.to(supa_device).contiguous()
    tiles_per_gauss = supa_wrapper.tile_count_unpacked(
        supa_means2d,
        supa_radii,
        tile_size,
        tile_width,
        tile_height,
    )
    if tiles_per_gauss is None:
        pytest.skip("SUPA native tile count is not available")
    lanes = supa_wrapper.isect_tiles_unpacked_lanes(
        supa_means2d,
        supa_radii,
        supa_depths,
        tiles_per_gauss,
        tile_size,
        tile_width,
        tile_height,
    )
    if lanes is None:
        pytest.skip("SUPA native tile lane expansion is not available")

    def _fail_cpu_offsets(*_args, **_kwargs):
        raise AssertionError("CPU tiled offsets helper should not be used")

    monkeypatch.setattr(
        supa_wrapper,
        "_isect_tiles_unpacked_offsets_cpu",
        _fail_cpu_offsets,
    )
    monkeypatch.setattr(
        supa_wrapper,
        "_isect_tiles_unpacked_offsets_lanes_cpu",
        _fail_cpu_offsets,
    )

    def _fail_old_native_offsets(*_args, **_kwargs):
        raise AssertionError("old custom native tiled offsets helper should not be used")

    monkeypatch.setattr(
        supa_wrapper,
        "tiled_offsets_from_lanes",
        _fail_old_native_offsets,
    )

    ref_tiles, ref_isect_ids, ref_flatten_ids = _isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, sort=True
    )
    ref_offsets = _isect_offset_encode(ref_isect_ids, n_images, tile_width, tile_height)
    native = supa_wrapper.isect_tiles_unpacked_offsets(
        supa_means2d,
        supa_radii,
        supa_depths,
        tile_size,
        tile_width,
        tile_height,
        use_native_offsets=True,
    )
    if native is None:
        pytest.skip("SUPA tile offset prototype is not available")
    native_tiles, tile_offsets, flatten_ids = native
    torch.testing.assert_close(native_tiles.cpu(), ref_tiles, rtol=0, atol=0)
    torch.testing.assert_close(tile_offsets.cpu(), ref_offsets, rtol=0, atol=0)
    torch.testing.assert_close(flatten_ids.cpu(), ref_flatten_ids, rtol=0, atol=0)


def test_supa_isect_tiles_unpacked_offsets_contract(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa._wrapper import isect_tiles_unpacked_offsets

    means2d = torch.tensor(
        [
            [[1.0, 1.0], [6.0, 1.5], [10.0, 4.0], [-3.0, 2.0], [4.0, 8.0]],
            [[2.0, 2.0], [6.5, 5.5], [11.5, 1.5], [1.5, 6.5], [8.0, 7.0]],
            [[2.5, 6.5], [7.5, 2.5], [12.0, 7.0], [4.0, -2.0], [9.5, 4.5]],
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [
            [[0, 0], [0, 2], [2, 0], [0, 0], [0, 0]],
            [[2, 2], [3, 2], [2, 1], [1, 3], [2, 2]],
            [[3, 1], [2, 3], [1, 1], [2, 2], [4, 2]],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [
            [2.0, 1.0, 3.0, 4.0, 5.0],
            [1.25, 0.75, 2.5, 1.75, 2.0],
            [0.5, 1.5, 2.5, 3.5, 1.0],
        ],
        dtype=torch.float32,
    )
    tile_size = 4
    tile_width = 4
    tile_height = 3

    native = isect_tiles_unpacked_offsets(
        means2d.to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        depths.to(supa_device).contiguous(),
        tile_size,
        tile_width,
        tile_height,
    )
    if native is None:
        pytest.skip("SUPA tile offset prototype is not available")
    tiles_per_gauss, tile_offsets, flatten_ids = native

    assert tile_offsets.dtype == torch.int32
    assert flatten_ids.dtype == torch.int32
    assert tile_offsets.shape == (means2d.shape[0], tile_height, tile_width)
    assert flatten_ids.numel() == int(tiles_per_gauss.cpu().sum().item())
    assert int(tiles_per_gauss.cpu()[0].sum().item()) == 0

    flat_offsets = tile_offsets.reshape(-1)
    range_ends = torch.cat(
        (flat_offsets[1:], torch.tensor([flatten_ids.numel()], dtype=torch.int32))
    )
    assert bool((range_ends >= flat_offsets).all().item())
    assert int(flat_offsets.min().item()) >= 0
    assert int(range_ends.max().item()) <= flatten_ids.numel()

    tile_means2d = means2d / tile_size
    tile_radii = radii.float() / tile_size
    tile_mins = torch.floor(tile_means2d - tile_radii).int()
    tile_maxs = torch.ceil(tile_means2d + tile_radii).int()
    tile_mins[..., 0] = torch.clamp(tile_mins[..., 0], 0, tile_width)
    tile_mins[..., 1] = torch.clamp(tile_mins[..., 1], 0, tile_height)
    tile_maxs[..., 0] = torch.clamp(tile_maxs[..., 0], 0, tile_width)
    tile_maxs[..., 1] = torch.clamp(tile_maxs[..., 1], 0, tile_height)

    n_gaussians = means2d.shape[1]
    for tile_linear, (start_tensor, end_tensor) in enumerate(zip(flat_offsets, range_ends)):
        start = int(start_tensor.item())
        end = int(end_tensor.item())
        if start == end:
            continue
        image_id = tile_linear // (tile_width * tile_height)
        local_tile = tile_linear % (tile_width * tile_height)
        tile_y = local_tile // tile_width
        tile_x = local_tile % tile_width
        ids = flatten_ids[start:end].long()
        gaussian_ids = ids % n_gaussians
        image_ids = ids // n_gaussians
        assert bool((image_ids == image_id).all().item())
        assert bool((tile_mins[image_id, gaussian_ids, 0] <= tile_x).all().item())
        assert bool((tile_maxs[image_id, gaussian_ids, 0] > tile_x).all().item())
        assert bool((tile_mins[image_id, gaussian_ids, 1] <= tile_y).all().item())
        assert bool((tile_maxs[image_id, gaussian_ids, 1] > tile_y).all().item())
        range_depths = depths.reshape(-1)[ids]
        assert bool((range_depths[1:] >= range_depths[:-1]).all().item())

    empty = isect_tiles_unpacked_offsets(
        means2d[:1].to(supa_device).contiguous(),
        torch.zeros_like(radii[:1]).to(supa_device).contiguous(),
        depths[:1].to(supa_device).contiguous(),
        tile_size,
        tile_width,
        tile_height,
    )
    if empty is None:
        pytest.skip("SUPA tile offset prototype is not available")
    empty_tiles, empty_offsets, empty_flatten_ids = empty
    assert int(empty_tiles.cpu().sum().item()) == 0
    assert int(empty_offsets.sum().item()) == 0
    assert empty_flatten_ids.numel() == 0


def _tiled_rasterize_reference(
    means2d,
    conics,
    features,
    opacities,
    width,
    height,
    tile_size,
    tile_offsets,
    flatten_ids,
    backgrounds=None,
):
    I, _, channels = features.shape
    flat_means2d = means2d.reshape(-1, 2)
    flat_conics = conics.reshape(-1, 3)
    flat_features = features.reshape(-1, channels)
    flat_opacities = opacities.reshape(-1)
    render = torch.zeros((I, height, width, channels), dtype=features.dtype)
    alpha = torch.zeros((I, height, width, 1), dtype=features.dtype)
    tile_height, tile_width = tile_offsets.shape[-2:]
    flat_offsets = tile_offsets.reshape(-1)
    n_isects = flatten_ids.numel()
    for image_id in range(I):
        for y in range(height):
            for x in range(width):
                tile_x = x // tile_size
                tile_y = y // tile_size
                tile_linear = (image_id * tile_height + tile_y) * tile_width + tile_x
                range_start = int(flat_offsets[tile_linear].item())
                range_end = (
                    n_isects
                    if tile_linear + 1 == flat_offsets.numel()
                    else int(flat_offsets[tile_linear + 1].item())
                )
                px = float(x) + 0.5
                py = float(y) + 0.5
                transmittance = 1.0
                alpha_acc = 0.0
                for idx_isect in range(range_start, range_end):
                    gidx = int(flatten_ids[idx_isect].item())
                    dx = px - float(flat_means2d[gidx, 0])
                    dy = py - float(flat_means2d[gidx, 1])
                    c = flat_conics[gidx]
                    sigma = (
                        0.5 * (c[0] * dx * dx + c[2] * dy * dy)
                        + c[1] * dx * dy
                    )
                    alpha_i = torch.clamp(flat_opacities[gidx] * torch.exp(-sigma), 0.0, 0.99)
                    weight = transmittance * float(alpha_i.item())
                    render[image_id, y, x] += weight * flat_features[gidx]
                    alpha_acc += weight
                    transmittance *= 1.0 - float(alpha_i.item())
                if backgrounds is not None:
                    render[image_id, y, x] += transmittance * backgrounds[image_id]
                alpha[image_id, y, x, 0] = alpha_acc
    return render, alpha


def _tiled_rasterize_autograd_reference(
    means2d,
    conics,
    features,
    opacities,
    width,
    height,
    tile_size,
    tile_offsets,
    flatten_ids,
    backgrounds=None,
):
    I, _, channels = features.shape
    flat_means2d = means2d.reshape(-1, 2)
    flat_conics = conics.reshape(-1, 3)
    flat_features = features.reshape(-1, channels)
    flat_opacities = opacities.reshape(-1)
    tile_height, tile_width = tile_offsets.shape[-2:]
    flat_offsets = tile_offsets.reshape(-1)
    n_isects = flatten_ids.numel()
    render_images = []
    alpha_images = []
    for image_id in range(I):
        render_rows = []
        alpha_rows = []
        for y in range(height):
            render_pixels = []
            alpha_pixels = []
            for x in range(width):
                tile_x = x // tile_size
                tile_y = y // tile_size
                tile_linear = (image_id * tile_height + tile_y) * tile_width + tile_x
                range_start = int(flat_offsets[tile_linear].item())
                range_end = (
                    n_isects
                    if tile_linear + 1 == flat_offsets.numel()
                    else int(flat_offsets[tile_linear + 1].item())
                )
                px = float(x) + 0.5
                py = float(y) + 0.5
                transmittance = torch.ones((), dtype=features.dtype, device=features.device)
                pixel_render = torch.zeros((channels,), dtype=features.dtype, device=features.device)
                pixel_alpha = torch.zeros((), dtype=features.dtype, device=features.device)
                for idx_isect in range(range_start, range_end):
                    gidx = int(flatten_ids[idx_isect].item())
                    dx = px - flat_means2d[gidx, 0]
                    dy = py - flat_means2d[gidx, 1]
                    c = flat_conics[gidx]
                    sigma = 0.5 * (c[0] * dx * dx + c[2] * dy * dy) + c[1] * dx * dy
                    alpha_i = torch.clamp(flat_opacities[gidx] * torch.exp(-sigma), 0.0, 0.99)
                    weight = transmittance * alpha_i
                    pixel_render = pixel_render + weight * flat_features[gidx]
                    pixel_alpha = pixel_alpha + weight
                    transmittance = transmittance * (1.0 - alpha_i)
                if backgrounds is not None:
                    pixel_render = pixel_render + transmittance * backgrounds[image_id]
                render_pixels.append(pixel_render)
                alpha_pixels.append(pixel_alpha.reshape(1))
            render_rows.append(torch.stack(render_pixels, dim=0))
            alpha_rows.append(torch.stack(alpha_pixels, dim=0))
        render_images.append(torch.stack(render_rows, dim=0))
        alpha_images.append(torch.stack(alpha_rows, dim=0))
    return torch.stack(render_images, dim=0), torch.stack(alpha_images, dim=0)


def test_supa_rasterize_to_pixels_tiled_forward_matches_cpu_reference(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa._wrapper import (
        isect_offset_encode,
        isect_tiles_unpacked,
        rasterize_to_pixels_tiled_fwd,
    )

    means2d = torch.tensor(
        [
            [[2.0, 2.0], [7.5, 1.5], [-2.0, 3.0], [4.0, 8.0]],
            [[1.0, 1.0], [3.5, 5.5], [8.5, 4.0], [6.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [
            [[2, 2], [3, 1], [2, 2], [0, 3]],
            [[1, 1], [4, 2], [2, 2], [3, 3]],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [
            [1.0, 0.5, 2.0, 1.5],
            [1.25, 0.75, 2.5, 0.25],
        ],
        dtype=torch.float32,
    )
    conics = torch.tensor(
        [
            [[0.35, 0.02, 0.45], [0.5, -0.03, 0.4], [0.3, 0.0, 0.3], [0.2, 0.01, 0.2]],
            [[0.4, 0.01, 0.5], [0.45, -0.02, 0.35], [0.25, 0.0, 0.25], [0.3, 0.02, 0.4]],
        ],
        dtype=torch.float32,
    )
    features = torch.tensor(
        [
            [[0.9, 0.1, 0.2], [0.2, 0.8, 0.1], [0.1, 0.3, 0.9], [0.7, 0.6, 0.2]],
            [[0.8, 0.2, 0.1], [0.1, 0.7, 0.3], [0.2, 0.2, 0.8], [0.6, 0.4, 0.2]],
        ],
        dtype=torch.float32,
    )
    opacities = torch.tensor(
        [[0.7, 0.55, 0.65, 0.4], [0.5, 0.6, 0.45, 0.35]], dtype=torch.float32
    )
    backgrounds = torch.tensor([[0.05, 0.1, 0.15], [0.2, 0.05, 0.1]], dtype=torch.float32)
    width = 9
    height = 7
    tile_size = 4
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size

    isects = isect_tiles_unpacked(
        means2d.to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        depths.to(supa_device).contiguous(),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )
    if isects is None:
        pytest.skip("SUPA tile intersection prototype is not available")
    _, isect_ids, flatten_ids = isects
    offsets = isect_offset_encode(isect_ids, means2d.shape[0], tile_width, tile_height)
    if offsets is None:
        pytest.skip("SUPA tile offset prototype is not available")
    ref_render, ref_alpha = _tiled_rasterize_reference(
        means2d,
        conics,
        features,
        opacities,
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        backgrounds,
    )

    native = rasterize_to_pixels_tiled_fwd(
        means2d.to(supa_device).contiguous(),
        conics.to(supa_device).contiguous(),
        features.to(supa_device).contiguous(),
        opacities.to(supa_device).contiguous(),
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        backgrounds.to(supa_device).contiguous(),
    )
    if native is None:
        pytest.skip("SUPA tiled rasterization prototype is not available")
    native_render, native_alpha = native
    torch.testing.assert_close(native_render.cpu(), ref_render, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(native_alpha.cpu(), ref_alpha, rtol=1e-5, atol=1e-6)


def test_supa_rasterize_to_pixels_tiled_forward_single_channel_matches_cpu_reference(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa._wrapper import (
        isect_offset_encode,
        isect_tiles_unpacked,
        rasterize_to_pixels_tiled_fwd,
    )

    means2d = torch.tensor(
        [
            [[2.0, 2.0], [7.5, 1.5], [-2.0, 3.0], [4.0, 8.0]],
            [[1.0, 1.0], [3.5, 5.5], [8.5, 4.0], [6.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor(
        [
            [[2, 2], [3, 1], [2, 2], [0, 3]],
            [[1, 1], [4, 2], [2, 2], [3, 3]],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [
            [1.0, 0.5, 2.0, 1.5],
            [1.25, 0.75, 2.5, 0.25],
        ],
        dtype=torch.float32,
    )
    conics = torch.tensor(
        [
            [[0.35, 0.02, 0.45], [0.5, -0.03, 0.4], [0.3, 0.0, 0.3], [0.2, 0.01, 0.2]],
            [[0.4, 0.01, 0.5], [0.45, -0.02, 0.35], [0.25, 0.0, 0.25], [0.3, 0.02, 0.4]],
        ],
        dtype=torch.float32,
    )
    features = depths[..., None].contiguous()
    opacities = torch.tensor(
        [[0.7, 0.55, 0.65, 0.4], [0.5, 0.6, 0.45, 0.35]], dtype=torch.float32
    )
    backgrounds = torch.tensor([[0.05], [0.2]], dtype=torch.float32)
    width = 9
    height = 7
    tile_size = 4
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size

    isects = isect_tiles_unpacked(
        means2d.to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        depths.to(supa_device).contiguous(),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )
    if isects is None:
        pytest.skip("SUPA tile intersection prototype is not available")
    _, isect_ids, flatten_ids = isects
    offsets = isect_offset_encode(isect_ids, means2d.shape[0], tile_width, tile_height)
    if offsets is None:
        pytest.skip("SUPA tile offset prototype is not available")
    ref_render, ref_alpha = _tiled_rasterize_reference(
        means2d,
        conics,
        features,
        opacities,
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        backgrounds,
    )

    native = rasterize_to_pixels_tiled_fwd(
        means2d.to(supa_device).contiguous(),
        conics.to(supa_device).contiguous(),
        features.to(supa_device).contiguous(),
        opacities.to(supa_device).contiguous(),
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        backgrounds.to(supa_device).contiguous(),
    )
    if native is None:
        pytest.skip("SUPA tiled rasterization prototype is not available")
    native_render, native_alpha = native
    torch.testing.assert_close(native_render.cpu(), ref_render, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(native_alpha.cpu(), ref_alpha, rtol=1e-5, atol=1e-6)


def test_supa_rasterize_to_pixels_tiled_backward_matches_autograd_reference(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa._wrapper import (
        isect_offset_encode,
        isect_tiles_unpacked,
        rasterize_to_pixels_tiled_fwd,
        rasterize_to_pixels_tiled_bwd,
    )

    means2d = torch.tensor(
        [
            [[2.0, 2.0], [7.5, 1.5], [-2.0, 3.0], [4.0, 8.0]],
            [[1.0, 1.0], [3.5, 5.5], [8.5, 4.0], [6.0, -1.0]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    radii = torch.tensor(
        [
            [[2, 2], [3, 1], [2, 2], [0, 3]],
            [[1, 1], [4, 2], [2, 2], [3, 3]],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [
            [1.0, 0.5, 2.0, 1.5],
            [1.25, 0.75, 2.5, 0.25],
        ],
        dtype=torch.float32,
    )
    conics = torch.tensor(
        [
            [[0.35, 0.02, 0.45], [0.5, -0.03, 0.4], [0.3, 0.0, 0.3], [0.2, 0.01, 0.2]],
            [[0.4, 0.01, 0.5], [0.45, -0.02, 0.35], [0.25, 0.0, 0.25], [0.3, 0.02, 0.4]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    features = torch.tensor(
        [
            [[0.9, 0.1, 0.2], [0.2, 0.8, 0.1], [0.1, 0.3, 0.9], [0.7, 0.6, 0.2]],
            [[0.8, 0.2, 0.1], [0.1, 0.7, 0.3], [0.2, 0.2, 0.8], [0.6, 0.4, 0.2]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    opacities = torch.tensor(
        [[0.7, 0.55, 0.65, 0.4], [0.5, 0.6, 0.45, 0.35]],
        dtype=torch.float32,
        requires_grad=True,
    )
    backgrounds = torch.tensor(
        [[0.05, 0.1, 0.15], [0.2, 0.05, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    width = 9
    height = 7
    tile_size = 4
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size

    isects = isect_tiles_unpacked(
        means2d.detach().to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        depths.to(supa_device).contiguous(),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )
    if isects is None:
        pytest.skip("SUPA tile intersection prototype is not available")
    _, isect_ids, flatten_ids = isects
    offsets = isect_offset_encode(isect_ids, means2d.shape[0], tile_width, tile_height)
    if offsets is None:
        pytest.skip("SUPA tile offset prototype is not available")

    ref_render, ref_alpha = _tiled_rasterize_autograd_reference(
        means2d,
        conics,
        features,
        opacities,
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        backgrounds,
    )
    v_render = torch.linspace(
        -0.2, 0.3, ref_render.numel(), dtype=torch.float32
    ).reshape_as(ref_render)
    v_alpha = torch.linspace(
        0.15, -0.1, ref_alpha.numel(), dtype=torch.float32
    ).reshape_as(ref_alpha)
    (ref_render * v_render).sum().add((ref_alpha * v_alpha).sum()).backward()

    native = rasterize_to_pixels_tiled_bwd(
        means2d.detach().to(supa_device).contiguous(),
        conics.detach().to(supa_device).contiguous(),
        features.detach().to(supa_device).contiguous(),
        opacities.detach().to(supa_device).contiguous(),
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        backgrounds.detach().to(supa_device).contiguous(),
        v_render.to(supa_device).contiguous(),
        v_alpha.to(supa_device).contiguous(),
    )
    if native is None:
        pytest.skip("SUPA tiled rasterization backward prototype is not available")
    v_means2d, v_conics, v_features, v_opacities, v_backgrounds = native
    torch.testing.assert_close(v_means2d.cpu(), means2d.grad, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(v_conics.cpu(), conics.grad, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(v_features.cpu(), features.grad, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(v_opacities.cpu(), opacities.grad, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(v_backgrounds.cpu(), backgrounds.grad, rtol=3e-4, atol=3e-5)


def test_supa_rasterize_to_pixels_tiled_backward_tile_reduce_matches_pixel_kernel(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa._wrapper import (
        isect_offset_encode,
        isect_tiles_unpacked,
        rasterize_to_pixels_tiled_fwd,
        rasterize_to_pixels_tiled_bwd,
    )

    means2d = torch.tensor(
        [[[4.0, 4.0], [10.5, 5.5], [15.0, 13.0], [2.5, 15.0], [18.0, 2.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    radii = torch.tensor([[[5, 5], [6, 4], [5, 6], [4, 5], [3, 3]]], dtype=torch.int32)
    depths = torch.tensor([[0.3, 1.2, 0.8, 1.8, 0.6]], dtype=torch.float32)
    conics = torch.tensor(
        [[[0.08, 0.01, 0.10], [0.06, -0.01, 0.09], [0.07, 0.02, 0.05], [0.09, 0.0, 0.07], [0.12, 0.01, 0.12]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    features = torch.tensor(
        [[[0.8, 0.1, 0.2], [0.2, 0.7, 0.1], [0.1, 0.4, 0.9], [0.6, 0.5, 0.2], [0.3, 0.2, 0.8]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    opacities = torch.tensor([[0.65, 0.55, 0.7, 0.45, 0.5]], dtype=torch.float32, requires_grad=True)
    backgrounds = torch.tensor([[0.02, 0.04, 0.06]], dtype=torch.float32, requires_grad=True)
    width = 19
    height = 18
    tile_size = 16
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size

    isects = isect_tiles_unpacked(
        means2d.detach().to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        depths.to(supa_device).contiguous(),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )
    if isects is None:
        pytest.skip("SUPA tile intersection prototype is not available")
    _, isect_ids, flatten_ids = isects
    offsets = isect_offset_encode(isect_ids, means2d.shape[0], tile_width, tile_height)
    if offsets is None:
        pytest.skip("SUPA tile offset prototype is not available")

    ref_render, ref_alpha = _tiled_rasterize_autograd_reference(
        means2d,
        conics,
        features,
        opacities,
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        backgrounds,
    )
    v_render = torch.linspace(-0.1, 0.2, ref_render.numel(), dtype=torch.float32).reshape_as(ref_render)
    v_alpha = torch.linspace(0.08, -0.04, ref_alpha.numel(), dtype=torch.float32).reshape_as(ref_alpha)
    (ref_render * v_render).sum().add((ref_alpha * v_alpha).sum()).backward()

    common_args = (
        means2d.detach().to(supa_device).contiguous(),
        conics.detach().to(supa_device).contiguous(),
        features.detach().to(supa_device).contiguous(),
        opacities.detach().to(supa_device).contiguous(),
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        backgrounds.detach().to(supa_device).contiguous(),
        v_render.to(supa_device).contiguous(),
        v_alpha.to(supa_device).contiguous(),
    )
    pixel = rasterize_to_pixels_tiled_bwd(*common_args, backward_mode="pixel")
    tile_reduce = rasterize_to_pixels_tiled_bwd(*common_args, backward_mode="tile_reduce")
    forward = rasterize_to_pixels_tiled_fwd(
        common_args[0],
        common_args[1],
        common_args[2],
        common_args[3],
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        common_args[9],
    )
    if forward is None:
        pytest.skip("SUPA tiled rasterization forward prototype is not available")
    _, forward_alpha = forward
    tile_reduce_with_alpha = rasterize_to_pixels_tiled_bwd(
        *common_args,
        forward_alpha=forward_alpha,
        backward_mode="tile_reduce",
    )
    parallel = rasterize_to_pixels_tiled_bwd(
        *common_args,
        forward_alpha=forward_alpha,
        backward_mode="parallel",
    )
    if pixel is None or tile_reduce is None or tile_reduce_with_alpha is None or parallel is None:
        pytest.skip("SUPA tiled rasterization backward prototypes are not available")

    for actual, expected in zip(tile_reduce, pixel):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=5e-4, atol=8e-5)
    for actual, expected in zip(tile_reduce_with_alpha, pixel):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=5e-4, atol=8e-5)
    for actual, expected in zip(parallel, pixel):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=5e-4, atol=8e-5)
    torch.testing.assert_close(tile_reduce_with_alpha[0].cpu(), means2d.grad, rtol=5e-4, atol=8e-5)
    torch.testing.assert_close(tile_reduce_with_alpha[1].cpu(), conics.grad, rtol=5e-4, atol=8e-5)
    torch.testing.assert_close(tile_reduce_with_alpha[2].cpu(), features.grad, rtol=5e-4, atol=8e-5)
    torch.testing.assert_close(tile_reduce_with_alpha[3].cpu(), opacities.grad, rtol=5e-4, atol=8e-5)
    torch.testing.assert_close(tile_reduce_with_alpha[4].cpu(), backgrounds.grad, rtol=5e-4, atol=8e-5)

    zero_v_alpha = torch.zeros_like(common_args[-1])
    common_zero_alpha_args = (*common_args[:-1], zero_v_alpha)
    common_none_alpha_args = (*common_args[:-1], None)
    pixel_zero_alpha = rasterize_to_pixels_tiled_bwd(
        *common_zero_alpha_args,
        forward_alpha=forward_alpha,
        backward_mode="pixel",
    )
    pixel_none_alpha = rasterize_to_pixels_tiled_bwd(
        *common_none_alpha_args,
        forward_alpha=forward_alpha,
        backward_mode="pixel",
    )
    tile_reduce_zero_alpha = rasterize_to_pixels_tiled_bwd(
        *common_zero_alpha_args,
        forward_alpha=forward_alpha,
        backward_mode="tile_reduce",
    )
    tile_reduce_none_alpha = rasterize_to_pixels_tiled_bwd(
        *common_none_alpha_args,
        forward_alpha=forward_alpha,
        backward_mode="tile_reduce",
    )
    if (
        pixel_zero_alpha is None
        or pixel_none_alpha is None
        or tile_reduce_zero_alpha is None
        or tile_reduce_none_alpha is None
    ):
        pytest.skip("SUPA tiled rasterization backward prototypes are not available")
    for actual, expected in zip(pixel_none_alpha, pixel_zero_alpha):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=5e-4, atol=8e-5)
    for actual, expected in zip(tile_reduce_none_alpha, tile_reduce_zero_alpha):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=5e-4, atol=8e-5)


def test_supa_rasterize_to_pixels_tiled_backward_tile_reduce_long_range_is_finite(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa._wrapper import (
        _TILED_BWD_AUTO_TILE_REDUCE_MAX_RANGE,
        rasterize_to_pixels_tiled_bwd,
        rasterize_to_pixels_tiled_fwd,
    )

    width = 16
    height = 16
    tile_size = 16
    n_gaussians = _TILED_BWD_AUTO_TILE_REDUCE_MAX_RANGE + 64
    ids = torch.arange(n_gaussians, dtype=torch.float32)
    phase = ids / float(n_gaussians)
    means2d = torch.stack(
        (
            torch.full_like(phase, 8.0) + 0.25 * torch.sin(phase * 17.0),
            torch.full_like(phase, 8.0) + 0.25 * torch.cos(phase * 13.0),
        ),
        dim=-1,
    ).reshape(1, n_gaussians, 2)
    conics = torch.stack(
        (
            torch.full_like(phase, 0.0020),
            torch.zeros_like(phase),
            torch.full_like(phase, 0.0025),
        ),
        dim=-1,
    ).reshape(1, n_gaussians, 3)
    features = torch.stack(
        (
            0.1 + torch.remainder(ids, 17.0) / 21.0,
            0.2 + torch.remainder(ids, 31.0) / 37.0,
            0.3 + torch.remainder(ids, 47.0) / 53.0,
        ),
        dim=-1,
    ).reshape(1, n_gaussians, 3)
    opacities = torch.full((1, n_gaussians), 0.002, dtype=torch.float32)
    backgrounds = torch.tensor([[0.03, 0.04, 0.05]], dtype=torch.float32)
    tile_offsets = torch.zeros((1, 1, 1), dtype=torch.int32)
    flatten_ids = torch.arange(n_gaussians, dtype=torch.int32)
    v_render = torch.linspace(
        -0.03,
        0.04,
        width * height * 3,
        dtype=torch.float32,
    ).reshape(1, height, width, 3)
    v_alpha = torch.linspace(
        0.02,
        -0.01,
        width * height,
        dtype=torch.float32,
    ).reshape(1, height, width, 1)

    supa_args = (
        means2d.to(supa_device).contiguous(),
        conics.to(supa_device).contiguous(),
        features.to(supa_device).contiguous(),
        opacities.to(supa_device).contiguous(),
        width,
        height,
        tile_size,
        tile_offsets,
        flatten_ids,
        backgrounds.to(supa_device).contiguous(),
        v_render.to(supa_device).contiguous(),
        v_alpha.to(supa_device).contiguous(),
    )
    forward = rasterize_to_pixels_tiled_fwd(
        supa_args[0],
        supa_args[1],
        supa_args[2],
        supa_args[3],
        width,
        height,
        tile_size,
        tile_offsets,
        flatten_ids,
        supa_args[9],
    )
    if forward is None:
        pytest.skip("SUPA tiled rasterization forward prototype is not available")
    _, forward_alpha = forward

    pixel = rasterize_to_pixels_tiled_bwd(
        *supa_args,
        forward_alpha=forward_alpha,
        backward_mode="pixel",
    )
    tile_reduce = rasterize_to_pixels_tiled_bwd(
        *supa_args,
        forward_alpha=forward_alpha,
        backward_mode="tile_reduce",
    )
    if pixel is None or tile_reduce is None:
        pytest.skip("SUPA tiled rasterization backward prototypes are not available")

    for actual, expected in zip(tile_reduce, pixel):
        assert actual is not None and expected is not None
        actual_cpu = actual.cpu()
        expected_cpu = expected.cpu()
        assert bool(torch.isfinite(actual_cpu).all().item())
        torch.testing.assert_close(actual_cpu, expected_cpu, rtol=2e-3, atol=2e-4)


def test_supa_rasterize_to_pixels_tiled_backward_tile_reduce_low_transmittance_edge_tile(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa._wrapper import (
        isect_offset_encode,
        isect_tiles_unpacked,
        rasterize_to_pixels_tiled_fwd,
        rasterize_to_pixels_tiled_bwd,
    )

    n_gaussians = 48
    means2d = torch.full((1, n_gaussians, 2), 15.0, dtype=torch.float32)
    radii = torch.full((1, n_gaussians, 2), 20, dtype=torch.int32)
    depths = torch.linspace(0.2, 2.0, n_gaussians, dtype=torch.float32).unsqueeze(0)
    conics = torch.zeros((1, n_gaussians, 3), dtype=torch.float32)
    conics[..., 0] = 0.02
    conics[..., 2] = 0.02
    features = torch.linspace(0.05, 0.95, n_gaussians * 3, dtype=torch.float32).reshape(1, n_gaussians, 3)
    opacities = torch.full((1, n_gaussians), 0.95, dtype=torch.float32)
    backgrounds = torch.tensor([[0.02, 0.04, 0.06]], dtype=torch.float32)
    width = 19
    height = 18
    tile_size = 16
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size

    isects = isect_tiles_unpacked(
        means2d.to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        depths.to(supa_device).contiguous(),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )
    if isects is None:
        pytest.skip("SUPA tile intersection prototype is not available")
    _, isect_ids, flatten_ids = isects
    offsets = isect_offset_encode(isect_ids, means2d.shape[0], tile_width, tile_height)
    if offsets is None:
        pytest.skip("SUPA tile offset prototype is not available")

    native_means = means2d.to(supa_device).contiguous()
    native_conics = conics.to(supa_device).contiguous()
    native_features = features.to(supa_device).contiguous()
    native_opacities = opacities.to(supa_device).contiguous()
    native_backgrounds = backgrounds.to(supa_device).contiguous()
    forward = rasterize_to_pixels_tiled_fwd(
        native_means,
        native_conics,
        native_features,
        native_opacities,
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        native_backgrounds,
    )
    if forward is None:
        pytest.skip("SUPA tiled rasterization forward prototype is not available")
    _, forward_alpha = forward
    assert float(forward_alpha.cpu().max().item()) > 0.999999

    v_render = torch.linspace(
        -0.1,
        0.2,
        width * height * 3,
        dtype=torch.float32,
    ).reshape(1, height, width, 3)
    v_alpha = torch.linspace(
        0.08,
        -0.04,
        width * height,
        dtype=torch.float32,
    ).reshape(1, height, width, 1)
    common_args = (
        native_means,
        native_conics,
        native_features,
        native_opacities,
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        native_backgrounds,
        v_render.to(supa_device).contiguous(),
        v_alpha.to(supa_device).contiguous(),
    )
    pixel = rasterize_to_pixels_tiled_bwd(*common_args, backward_mode="pixel")
    tile_reduce = rasterize_to_pixels_tiled_bwd(
        *common_args,
        forward_alpha=forward_alpha,
        backward_mode="tile_reduce",
    )
    if pixel is None or tile_reduce is None:
        pytest.skip("SUPA tiled rasterization backward prototypes are not available")
    for actual, expected in zip(tile_reduce, pixel):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-3, atol=2e-4)


def test_supa_rasterize_to_pixels_tiled_backward_low_transmittance_matches_reference(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa._wrapper import (
        isect_offset_encode,
        isect_tiles_unpacked,
        rasterize_to_pixels_tiled_bwd,
    )

    n_gaussians = 48
    means2d = torch.full((1, n_gaussians, 2), 4.0, dtype=torch.float32)
    means2d.requires_grad_(True)
    radii = torch.full((1, n_gaussians, 2), 8, dtype=torch.int32)
    depths = torch.linspace(0.25, 2.0, n_gaussians, dtype=torch.float32).unsqueeze(0)
    conics = torch.zeros((1, n_gaussians, 3), dtype=torch.float32)
    conics[..., 0] = 0.08
    conics[..., 2] = 0.08
    conics.requires_grad_(True)
    features = torch.linspace(
        0.05,
        0.95,
        n_gaussians * 3,
        dtype=torch.float32,
    ).reshape(1, n_gaussians, 3)
    features.requires_grad_(True)
    opacities = torch.full((1, n_gaussians), 0.95, dtype=torch.float32)
    opacities.requires_grad_(True)
    backgrounds = torch.tensor([[0.02, 0.04, 0.06]], dtype=torch.float32)
    backgrounds.requires_grad_(True)
    width = 8
    height = 8
    tile_size = 8
    tile_width = 1
    tile_height = 1

    isects = isect_tiles_unpacked(
        means2d.detach().to(supa_device).contiguous(),
        radii.to(supa_device).contiguous(),
        depths.to(supa_device).contiguous(),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )
    if isects is None:
        pytest.skip("SUPA tile intersection prototype is not available")
    _, isect_ids, flatten_ids = isects
    offsets = isect_offset_encode(isect_ids, means2d.shape[0], tile_width, tile_height)
    if offsets is None:
        pytest.skip("SUPA tile offset prototype is not available")

    ref_render, ref_alpha = _tiled_rasterize_autograd_reference(
        means2d,
        conics,
        features,
        opacities,
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        backgrounds,
    )
    v_render = torch.linspace(
        -0.15,
        0.25,
        ref_render.numel(),
        dtype=torch.float32,
    ).reshape_as(ref_render)
    v_alpha = torch.linspace(
        0.2,
        -0.05,
        ref_alpha.numel(),
        dtype=torch.float32,
    ).reshape_as(ref_alpha)
    (ref_render * v_render).sum().add((ref_alpha * v_alpha).sum()).backward()

    native = rasterize_to_pixels_tiled_bwd(
        means2d.detach().to(supa_device).contiguous(),
        conics.detach().to(supa_device).contiguous(),
        features.detach().to(supa_device).contiguous(),
        opacities.detach().to(supa_device).contiguous(),
        width,
        height,
        tile_size,
        offsets,
        flatten_ids,
        backgrounds.detach().to(supa_device).contiguous(),
        v_render.to(supa_device).contiguous(),
        v_alpha.to(supa_device).contiguous(),
    )
    if native is None:
        pytest.skip("SUPA tiled rasterization backward prototype is not available")
    v_means2d, v_conics, v_features, v_opacities, v_backgrounds = native
    torch.testing.assert_close(v_means2d.cpu(), means2d.grad, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(v_conics.cpu(), conics.grad, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(v_features.cpu(), features.grad, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(v_opacities.cpu(), opacities.grad, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(v_backgrounds.cpu(), backgrounds.grad, rtol=2e-3, atol=2e-4)


def test_supa_native_composite_opt_in_keeps_public_rasterization_forward_correct(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)
    for name in ("means", "quats", "scales", "opacities", "colors"):
        supa_scene[name] = supa_scene[name].detach()

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene, render_mode="RGB+D", chunk_size=2
    )
    with torch.no_grad():
        supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
            supa_scene, render_mode="RGB+D", channel_chunk=2
        )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    assert supa_meta["backend"] == "torch_fallback"
    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    assert supa_meta["composite_variant"] == "full_scan"
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=3e-3, atol=2e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=3e-3, atol=2e-4)


def test_supa_native_tiled_opt_in_keeps_public_rasterization_forward_correct(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILED", "1")

    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)
    for name in ("means", "quats", "scales", "opacities", "colors"):
        supa_scene[name] = supa_scene[name].detach()

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene, render_mode="RGB+D", chunk_size=2
    )
    with torch.no_grad():
        supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
            supa_scene, render_mode="RGB+D", channel_chunk=2
        )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    if supa_meta["composite_variant"] != "tiled":
        pytest.skip("SUPA native tiled composite is not available")
    assert supa_meta["backend"] == "torch_fallback"
    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=3e-3, atol=2e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=3e-3, atol=2e-4)


def test_supa_native_tiled_opt_in_keeps_public_rgb_forward_correct(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILED", "1")

    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)
    for name in ("means", "quats", "scales", "opacities", "colors"):
        supa_scene[name] = supa_scene[name].detach()

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene, render_mode="RGB", chunk_size=2
    )
    with torch.no_grad():
        supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
            supa_scene, render_mode="RGB", channel_chunk=2
        )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    if supa_meta["composite_variant"] != "tiled":
        pytest.skip("SUPA native tiled composite is not available")
    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=3e-3, atol=2e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=3e-3, atol=2e-4)


def test_supa_native_no_grad_rgb_materializes_elementwise_features(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILED", "1")

    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)
    for name in ("means", "quats", "scales"):
        cpu_scene[name] = cpu_scene[name].detach()
        supa_scene[name] = supa_scene[name].detach()

    cpu_color_logits = torch.logit(cpu_scene["colors"].detach().clamp(1e-4, 1.0 - 1e-4))
    cpu_opacity_logits = torch.logit(
        cpu_scene["opacities"].detach().clamp(1e-4, 1.0 - 1e-4)
    )
    supa_color_logits = torch.logit(
        supa_scene["colors"].detach().clamp(1e-4, 1.0 - 1e-4)
    )
    supa_opacity_logits = torch.logit(
        supa_scene["opacities"].detach().clamp(1e-4, 1.0 - 1e-4)
    )

    cpu_scene["colors"] = torch.sigmoid(cpu_color_logits)
    cpu_scene["opacities"] = torch.sigmoid(cpu_opacity_logits)
    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene, render_mode="RGB", chunk_size=2
    )

    with torch.no_grad():
        supa_scene["colors"] = torch.sigmoid(supa_color_logits)
        supa_scene["opacities"] = torch.sigmoid(supa_opacity_logits)
        supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
            supa_scene, render_mode="RGB", channel_chunk=2
        )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=3e-3, atol=2e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=3e-3, atol=2e-4)


@pytest.mark.parametrize("render_mode", ["D", "ED"])
@pytest.mark.parametrize("tiled", [False, True])
def test_supa_native_opt_in_keeps_public_depth_forward_correct(
    monkeypatch, tiled, render_mode
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILED", "1" if tiled else "0")

    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)
    for name in ("means", "quats", "scales", "opacities", "colors"):
        supa_scene[name] = supa_scene[name].detach()

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene, colors=None, render_mode=render_mode, chunk_size=2
    )
    with torch.no_grad():
        supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
            supa_scene, colors=None, render_mode=render_mode, channel_chunk=2
        )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    expected_variant = "tiled" if tiled else "full_scan"
    if supa_meta["composite_variant"] != expected_variant:
        pytest.skip(f"SUPA native {expected_variant} composite is not available")
    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=3e-3, atol=2e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=3e-3, atol=2e-4)


def test_supa_native_tiled_reuses_composite_tile_counts(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILED", "1")

    import gsplat._torch_fallback as torch_fallback

    calls = {"count": 0}

    def count_extra_tile_count(*_args, **_kwargs):
        calls["count"] += 1
        return None

    monkeypatch.setattr(
        torch_fallback,
        "_try_native_tiles_per_gauss",
        count_extra_tile_count,
    )

    supa_scene = _make_basic_scene(supa_device)
    for name in ("means", "quats", "scales", "opacities", "colors"):
        supa_scene[name] = supa_scene[name].detach()

    with torch.no_grad():
        _, _, supa_meta = _call_public_rasterization(
            supa_scene, render_mode="RGB", channel_chunk=2
        )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    if supa_meta["composite_variant"] != "tiled":
        pytest.skip("SUPA native tiled composite is not available")
    assert calls["count"] == 0
    assert supa_meta["tile_count_backend"] == "supa_native"
    assert supa_meta["tiled_metadata_backend"] == "cpu_numpy"
    ref_tiles = torch_fallback._tiles_per_gauss(
        supa_meta["means2d"],
        supa_meta["radii"],
        supa_meta["tile_size"],
        supa_meta["tile_width"],
        supa_meta["tile_height"],
    )
    torch.testing.assert_close(
        supa_meta["tiles_per_gauss"].cpu(),
        ref_tiles.cpu(),
        rtol=0,
        atol=0,
    )


def test_supa_native_tiled_records_native_metadata_backend(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILED", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILED_METADATA", "1")

    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)
    for name in ("means", "quats", "scales", "opacities", "colors"):
        supa_scene[name] = supa_scene[name].detach()

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene, render_mode="RGB", chunk_size=2
    )
    with torch.no_grad():
        supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
            supa_scene, render_mode="RGB", channel_chunk=2
        )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    if supa_meta["composite_variant"] != "tiled":
        pytest.skip("SUPA native tiled composite is not available")
    assert supa_meta["tile_count_backend"] == "supa_native"
    assert supa_meta["tiled_metadata_backend"] == "supa_native_offsets"
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=3e-3, atol=2e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=3e-3, atol=2e-4)


def test_supa_native_tiled_auto_is_conservative(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILED", "auto")

    supa_scene = _make_basic_scene(supa_device)
    for name in ("means", "quats", "scales", "opacities", "colors"):
        supa_scene[name] = supa_scene[name].detach()

    with torch.no_grad():
        _, _, small_meta = _call_public_rasterization(
            supa_scene, render_mode="RGB", channel_chunk=2
        )
    if small_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    assert small_meta["composite_variant"] == "full_scan"

    import gsplat._torch_fallback as fallback

    monkeypatch.setattr(fallback, "_SUPA_TILED_AUTO_MIN_GAUSSIANS", 3)
    monkeypatch.setattr(fallback, "_SUPA_TILED_AUTO_MIN_PIXELS", 8 * 8)
    with torch.no_grad():
        _, _, auto_meta = _call_public_rasterization(
            supa_scene, render_mode="RGB", channel_chunk=2
        )
    if auto_meta["composite_variant"] != "tiled":
        pytest.skip("SUPA native tiled composite is not available")
    assert auto_meta["composite_backend"] == "supa_native"
    assert auto_meta["composite_variant"] == "tiled"


def test_supa_native_composite_opt_in_preserves_public_rasterization_backward(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene, render_mode="RGB+D", chunk_size=2
    )
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene, render_mode="RGB+D", channel_chunk=2
    )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    assert supa_meta["composite_variant"] == "full_scan"

    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=3e-3, atol=2e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=3e-3, atol=2e-4)

    (cpu_renders.sum() + cpu_alphas.sum()).backward()
    (supa_renders.sum() + supa_alphas.sum()).backward()

    _assert_scene_grads_close(
        cpu_scene,
        supa_scene,
        ("means", "quats", "scales", "opacities", "colors"),
        rtol=4e-3,
        atol=1e-3,
    )


def test_supa_native_tiled_train_opt_in_preserves_public_rasterization_backward(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILED_TRAIN", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILED_METADATA", "1")
    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene, render_mode="RGB+D", chunk_size=2
    )
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene, render_mode="RGB+D", channel_chunk=2
    )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    if supa_meta["composite_variant"] != "tiled":
        pytest.skip("SUPA native tiled training composite is not available")
    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    assert supa_meta["tile_count_backend"] == "supa_native"
    assert supa_meta["tiled_metadata_backend"] == "supa_native_offsets"

    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=3e-3, atol=2e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=3e-3, atol=2e-4)

    (cpu_renders.sum() + cpu_alphas.sum()).backward()
    (supa_renders.sum() + supa_alphas.sum()).backward()

    _assert_scene_grads_close(
        cpu_scene,
        supa_scene,
        ("means", "quats", "scales", "opacities", "colors"),
        rtol=5e-3,
        atol=2e-3,
    )


def test_supa_native_tiled_train_alpha_threshold_uses_native_metadata(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "0")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILED_TRAIN", "0")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILED_METADATA", "0")

    supa_scene = _make_basic_scene(supa_device)
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene,
        render_mode="RGB",
        channel_chunk=2,
        supa_native_options={
            "native": True,
            "native_tiled": "false",
            "native_tiled_metadata": True,
            "native_tiled_train": True,
            "native_alpha_threshold": 1.0 / 255.0,
        },
    )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    if supa_meta["composite_variant"] != "tiled":
        pytest.skip("SUPA native tiled training composite is not available")
    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    assert supa_meta["tile_count_backend"] == "supa_native"
    assert supa_meta["tiled_metadata_backend"] == "supa_native_alpha_threshold"

    (supa_renders.sum() + supa_alphas.sum()).backward()
    for name in ("means", "quats", "scales", "opacities", "colors"):
        assert torch.isfinite(supa_scene[name].grad).all()


def test_supa_native_composite_backward_does_not_recompute_with_torch(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    supa_scene = _make_basic_scene(supa_device)

    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene, render_mode="RGB+D", channel_chunk=2
    )
    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")

    import gsplat._torch_fallback as torch_fallback

    def fail_recompute(*args, **kwargs):
        raise AssertionError("native composite backward should not recompute in PyTorch")

    monkeypatch.setattr(torch_fallback, "_alpha_composite", fail_recompute)
    (supa_renders.sum() + supa_alphas.sum()).backward()
    for name in ("means", "quats", "scales", "opacities", "colors"):
        assert supa_scene[name].grad is not None


def test_supa_native_opt_in_handles_two_cameras_backgrounds_extra_and_expected_depth(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    cpu_scene = _make_two_camera_scene("cpu")
    supa_scene = _make_two_camera_scene(supa_device)
    cpu_extra = _make_extra_signals(cpu_scene)
    supa_extra = _make_extra_signals(supa_scene)
    cpu_backgrounds = torch.tensor(
        [[0.05, 0.10, 0.15], [0.15, 0.08, 0.03]], dtype=torch.float32
    )

    cpu_renders, cpu_alphas, cpu_meta = _call_reference_fallback(
        cpu_scene,
        render_mode="RGB+ED",
        extra_signals=cpu_extra,
        backgrounds=cpu_backgrounds,
        chunk_size=2,
    )
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene,
        render_mode="RGB+ED",
        extra_signals=supa_extra,
        backgrounds=cpu_backgrounds.to(supa_device),
        channel_chunk=2,
    )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    assert supa_renders.shape == (2, 8, 8, 4)
    assert supa_meta["render_extra_signals"].shape == (2, 8, 8, 2)

    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=4e-3, atol=3e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=4e-3, atol=3e-4)
    torch.testing.assert_close(
        supa_meta["render_extra_signals"].cpu(),
        cpu_meta["render_extra_signals"],
        rtol=4e-3,
        atol=3e-4,
    )

    cpu_loss = (
        cpu_renders.sum()
        + cpu_alphas.sum()
        + cpu_meta["render_extra_signals"].sum()
    )
    supa_loss = (
        supa_renders.sum()
        + supa_alphas.sum()
        + supa_meta["render_extra_signals"].sum()
    )
    cpu_loss.backward()
    supa_loss.backward()

    _assert_scene_grads_close(
        cpu_scene,
        supa_scene,
        ("means", "quats", "scales", "opacities", "colors"),
        rtol=5e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(
        supa_extra.grad.cpu(), cpu_extra.grad, rtol=5e-3, atol=2e-3
    )


def test_supa_native_tiled_train_handles_two_cameras_backgrounds_extra_and_expected_depth(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE_TILED_TRAIN", "1")

    cpu_scene = _make_two_camera_scene("cpu")
    supa_scene = _make_two_camera_scene(supa_device)
    cpu_extra = _make_extra_signals(cpu_scene)
    supa_extra = _make_extra_signals(supa_scene)
    cpu_backgrounds = torch.tensor(
        [[0.05, 0.10, 0.15], [0.15, 0.08, 0.03]], dtype=torch.float32
    )

    cpu_renders, cpu_alphas, cpu_meta = _call_reference_fallback(
        cpu_scene,
        render_mode="RGB+ED",
        extra_signals=cpu_extra,
        backgrounds=cpu_backgrounds,
        chunk_size=2,
    )
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene,
        render_mode="RGB+ED",
        extra_signals=supa_extra,
        backgrounds=cpu_backgrounds.to(supa_device),
        channel_chunk=2,
    )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    if supa_meta["composite_variant"] != "tiled":
        pytest.skip("SUPA native tiled training composite is not available")
    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    assert supa_meta["tile_count_backend"] == "supa_native"
    assert supa_renders.shape == (2, 8, 8, 4)
    assert supa_meta["render_extra_signals"].shape == (2, 8, 8, 2)

    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=4e-3, atol=3e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=4e-3, atol=3e-4)
    torch.testing.assert_close(
        supa_meta["render_extra_signals"].cpu(),
        cpu_meta["render_extra_signals"],
        rtol=4e-3,
        atol=3e-4,
    )

    cpu_loss = (
        cpu_renders.sum()
        + cpu_alphas.sum()
        + cpu_meta["render_extra_signals"].sum()
    )
    supa_loss = (
        supa_renders.sum()
        + supa_alphas.sum()
        + supa_meta["render_extra_signals"].sum()
    )
    cpu_loss.backward()
    supa_loss.backward()

    _assert_scene_grads_close(
        cpu_scene,
        supa_scene,
        ("means", "quats", "scales", "opacities", "colors"),
        rtol=5e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(
        supa_extra.grad.cpu(), cpu_extra.grad, rtol=5e-3, atol=2e-3
    )


def test_supa_native_opt_in_handles_per_view_colors(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    cpu_scene = _make_two_camera_scene("cpu")
    supa_scene = _make_two_camera_scene(supa_device)
    cpu_colors = torch.stack(
        (
            cpu_scene["colors"].detach(),
            torch.flip(cpu_scene["colors"].detach(), dims=(0,)),
        ),
        dim=0,
    ).requires_grad_(True)
    supa_colors = cpu_colors.detach().to(supa_device).requires_grad_(True)

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene, colors=cpu_colors, render_mode="RGB", chunk_size=2
    )
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene,
        colors=supa_colors,
        render_mode="RGB",
        channel_chunk=2,
    )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    assert supa_renders.shape == (2, 8, 8, 3)

    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=4e-3, atol=3e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=4e-3, atol=3e-4)

    (cpu_renders.sum() + cpu_alphas.sum()).backward()
    (supa_renders.sum() + supa_alphas.sum()).backward()

    _assert_scene_grads_close(
        cpu_scene,
        supa_scene,
        ("means", "quats", "scales", "opacities"),
        rtol=5e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(
        supa_colors.grad.cpu(), cpu_colors.grad, rtol=5e-3, atol=2e-3
    )


def _make_repeated_basic_scene(device: torch.device | str, num_points: int):
    scene = _make_basic_scene(device)
    repeats = (num_points + scene["means"].shape[0] - 1) // scene["means"].shape[0]
    for name in ("means", "quats", "scales", "opacities", "colors"):
        tensor = scene[name].detach()
        repeat_shape = (repeats,) + (1,) * (tensor.dim() - 1)
        scene[name] = tensor.repeat(repeat_shape)[:num_points].contiguous()
    jitter = torch.linspace(
        -0.05,
        0.05,
        num_points,
        dtype=scene["means"].dtype,
        device=scene["means"].device,
    )
    scene["means"][:, 0] = scene["means"][:, 0] + jitter
    scene["means"][:, 1] = scene["means"][:, 1] - jitter
    for name in ("means", "quats", "scales", "opacities", "colors"):
        scene[name].requires_grad_(True)
    return scene


def test_supa_native_composite_forward_only_allows_more_than_backward_limit(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    num_points = 1025
    cpu_scene = _make_repeated_basic_scene("cpu", num_points)
    supa_scene = _make_repeated_basic_scene(supa_device, num_points)

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene, render_mode="RGB", width=4, height=4, chunk_size=64
    )
    with torch.no_grad():
        supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
            supa_scene,
            render_mode="RGB",
            width=4,
            height=4,
            channel_chunk=64,
        )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite forward is not available")
    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    assert supa_meta["composite_variant"] == "full_scan"
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=4e-3, atol=3e-4)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=4e-3, atol=3e-4)


def test_supa_native_composite_training_allows_more_than_old_backward_limit(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    supa_scene = _make_repeated_basic_scene(supa_device, 1025)
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene,
        render_mode="RGB",
        width=4,
        height=4,
        channel_chunk=64,
    )

    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    assert supa_meta["composite_variant"] == "full_scan"
    (supa_renders.sum() + supa_alphas.sum()).backward()
    for name in ("means", "quats", "scales", "opacities", "colors"):
        assert supa_scene[name].grad is not None


def test_supa_native_composite_training_allows_2048_gaussians(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    supa_scene = _make_repeated_basic_scene(supa_device, 2048)
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene,
        render_mode="RGB",
        width=4,
        height=4,
        channel_chunk=64,
    )

    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    assert supa_meta["composite_variant"] == "full_scan"
    (supa_renders.sum() + supa_alphas.sum()).backward()
    for name in ("means", "quats", "scales", "opacities", "colors"):
        assert supa_scene[name].grad is not None


def test_supa_native_opt_in_matches_garden_asset_subset(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._torch_fallback import _tiles_per_gauss

    cpu_scene = _make_garden_subset_scene("cpu")
    supa_scene = _make_garden_subset_scene(supa_device)
    backgrounds = torch.tensor(
        [[0.03, 0.04, 0.05], [0.07, 0.02, 0.01]], dtype=torch.float32
    )

    cpu_renders, cpu_alphas, _ = _call_reference_fallback(
        cpu_scene,
        backgrounds=backgrounds,
        render_mode="RGB+D",
        chunk_size=8,
        width=cpu_scene["width"],
        height=cpu_scene["height"],
    )
    supa_renders, supa_alphas, supa_meta = _call_public_rasterization(
        supa_scene,
        backgrounds=backgrounds.to(supa_device),
        render_mode="RGB+D",
        channel_chunk=8,
        width=supa_scene["width"],
        height=supa_scene["height"],
    )

    if supa_meta["composite_backend"] != "supa_native":
        pytest.skip("SUPA native composite is not available")
    assert supa_meta["projection_backend"] == "supa_native"
    assert supa_meta["composite_backend"] == "supa_native"
    assert supa_meta["tile_count_backend"] == "supa_native"
    ref_tiles = _tiles_per_gauss(
        supa_meta["means2d"],
        supa_meta["radii"],
        supa_meta["tile_size"],
        supa_meta["tile_width"],
        supa_meta["tile_height"],
    )
    torch.testing.assert_close(
        supa_meta["tiles_per_gauss"].cpu(),
        ref_tiles.cpu(),
        rtol=0,
        atol=0,
    )
    assert supa_renders.shape == (2, 24, 32, 4)
    assert supa_alphas.shape == (2, 24, 32, 1)
    torch.testing.assert_close(supa_renders.cpu(), cpu_renders, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(supa_alphas.cpu(), cpu_alphas, rtol=5e-5, atol=5e-5)

    cpu_loss = cpu_renders.square().mean() + 0.3 * cpu_alphas.square().mean()
    supa_loss = supa_renders.square().mean() + 0.3 * supa_alphas.square().mean()
    cpu_loss.backward()
    supa_loss.backward()

    _assert_scene_grads_close(
        cpu_scene,
        supa_scene,
        ("means", "quats", "scales", "opacities", "colors"),
        rtol=5e-4,
        atol=5e-5,
    )


def test_supa_native_tile_count_matches_garden_projection(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._torch_fallback import _tiles_per_gauss
    from gsplat.supa._wrapper import projection_ewa_3dgs_pinhole, tile_count_unpacked

    scene = _make_garden_subset_scene(supa_device, num_points=96)
    projection = projection_ewa_3dgs_pinhole(
        scene["means"].detach().contiguous(),
        None,
        scene["viewmats"].contiguous(),
        scene["Ks"].contiguous(),
        width=scene["width"],
        height=scene["height"],
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        calc_compensations=False,
        quats=scene["quats"].detach().contiguous(),
        scales=scene["scales"].detach().contiguous(),
    )
    if projection is None:
        pytest.skip("SUPA native projection is not available")
    radii, means2d, _, _, _ = projection

    tile_size = 16
    tile_width = (scene["width"] + tile_size - 1) // tile_size
    tile_height = (scene["height"] + tile_size - 1) // tile_size
    native = tile_count_unpacked(
        means2d.reshape(-1, means2d.shape[-2], 2).contiguous(),
        radii.reshape(-1, radii.shape[-2], 2).contiguous(),
        tile_size,
        tile_width,
        tile_height,
    )
    if native is None:
        pytest.skip("SUPA native tile count is not available")

    ref = _tiles_per_gauss(
        means2d,
        radii,
        tile_size,
        tile_width,
        tile_height,
    )
    torch.testing.assert_close(
        native.cpu().reshape(ref.shape),
        ref.cpu(),
        rtol=0,
        atol=0,
    )


def test_supa_native_isect_tiles_matches_garden_projection(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._torch_impl import _isect_tiles
    from gsplat.supa._wrapper import projection_ewa_3dgs_pinhole, isect_tiles_unpacked

    scene = _make_garden_subset_scene(supa_device, num_points=96)
    projection = projection_ewa_3dgs_pinhole(
        scene["means"].detach().contiguous(),
        None,
        scene["viewmats"].contiguous(),
        scene["Ks"].contiguous(),
        width=scene["width"],
        height=scene["height"],
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        calc_compensations=False,
        quats=scene["quats"].detach().contiguous(),
        scales=scene["scales"].detach().contiguous(),
    )
    if projection is None:
        pytest.skip("SUPA native projection is not available")
    radii, means2d, depths, _, _ = projection

    tile_size = 16
    tile_width = (scene["width"] + tile_size - 1) // tile_size
    tile_height = (scene["height"] + tile_size - 1) // tile_size
    native = isect_tiles_unpacked(
        means2d.reshape(-1, means2d.shape[-2], 2).contiguous(),
        radii.reshape(-1, radii.shape[-2], 2).contiguous(),
        depths.reshape(-1, depths.shape[-1]).contiguous(),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )
    if native is None:
        pytest.skip("SUPA native tile intersection is not available")

    ref = _isect_tiles(
        means2d.detach().cpu(),
        radii.detach().cpu(),
        depths.detach().cpu(),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )
    tiles_per_gauss, isect_ids, flatten_ids = native
    torch.testing.assert_close(
        tiles_per_gauss.cpu().reshape(ref[0].shape), ref[0], rtol=0, atol=0
    )
    torch.testing.assert_close(isect_ids.cpu(), ref[1], rtol=0, atol=0)
    torch.testing.assert_close(flatten_ids.cpu(), ref[2], rtol=0, atol=0)


def test_supa_isect_offset_encode_matches_garden_projection(monkeypatch):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._torch_impl import _isect_offset_encode, _isect_tiles
    from gsplat.supa._wrapper import (
        projection_ewa_3dgs_pinhole,
        isect_offset_encode,
        isect_tiles_unpacked,
    )

    scene = _make_garden_subset_scene(supa_device, num_points=96)
    projection = projection_ewa_3dgs_pinhole(
        scene["means"].detach().contiguous(),
        None,
        scene["viewmats"].contiguous(),
        scene["Ks"].contiguous(),
        width=scene["width"],
        height=scene["height"],
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        calc_compensations=False,
        quats=scene["quats"].detach().contiguous(),
        scales=scene["scales"].detach().contiguous(),
    )
    if projection is None:
        pytest.skip("SUPA native projection is not available")
    radii, means2d, depths, _, _ = projection

    tile_size = 16
    tile_width = (scene["width"] + tile_size - 1) // tile_size
    tile_height = (scene["height"] + tile_size - 1) // tile_size
    native = isect_tiles_unpacked(
        means2d.reshape(-1, means2d.shape[-2], 2).contiguous(),
        radii.reshape(-1, radii.shape[-2], 2).contiguous(),
        depths.reshape(-1, depths.shape[-1]).contiguous(),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )
    if native is None:
        pytest.skip("SUPA tile intersection prototype is not available")

    ref = _isect_tiles(
        means2d.detach().cpu(),
        radii.detach().cpu(),
        depths.detach().cpu(),
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )
    n_images = means2d.reshape(-1, means2d.shape[-2], 2).shape[0]
    ref_offsets = _isect_offset_encode(ref[1], n_images, tile_width, tile_height)
    native_offsets = isect_offset_encode(native[1], n_images, tile_width, tile_height)
    if native_offsets is None:
        pytest.skip("SUPA tile offset prototype is not available")
    torch.testing.assert_close(native_offsets, ref_offsets, rtol=0, atol=0)


def test_supa_rasterize_to_pixels_tiled_forward_matches_garden_projection_reference(
    monkeypatch,
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat.supa._wrapper import (
        projection_ewa_3dgs_pinhole,
        isect_offset_encode,
        isect_tiles_unpacked,
        rasterize_to_pixels_tiled_fwd,
    )

    scene = _make_garden_subset_scene(supa_device, num_points=96)
    projection = projection_ewa_3dgs_pinhole(
        scene["means"].detach().contiguous(),
        None,
        scene["viewmats"].contiguous(),
        scene["Ks"].contiguous(),
        width=scene["width"],
        height=scene["height"],
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        calc_compensations=False,
        quats=scene["quats"].detach().contiguous(),
        scales=scene["scales"].detach().contiguous(),
    )
    if projection is None:
        pytest.skip("SUPA native projection is not available")
    radii, means2d, depths, conics, _ = projection

    tile_size = 16
    tile_width = (scene["width"] + tile_size - 1) // tile_size
    tile_height = (scene["height"] + tile_size - 1) // tile_size
    flat_means2d = means2d.reshape(-1, means2d.shape[-2], 2).contiguous()
    flat_radii = radii.reshape(-1, radii.shape[-2], 2).contiguous()
    flat_depths = depths.reshape(-1, depths.shape[-1]).contiguous()
    native_isects = isect_tiles_unpacked(
        flat_means2d,
        flat_radii,
        flat_depths,
        tile_size,
        tile_width,
        tile_height,
        sort=True,
    )
    if native_isects is None:
        pytest.skip("SUPA tile intersection prototype is not available")
    _, isect_ids, flatten_ids = native_isects
    n_images = flat_means2d.shape[0]
    offsets = isect_offset_encode(isect_ids, n_images, tile_width, tile_height)
    if offsets is None:
        pytest.skip("SUPA tile offset prototype is not available")

    flat_conics = conics.reshape(n_images, conics.shape[-2], 3).contiguous()
    flat_colors = scene["colors"][None, ...].expand(n_images, -1, -1).contiguous()
    flat_opacities = scene["opacities"][None, ...].expand(n_images, -1).contiguous()
    backgrounds = torch.tensor(
        [[0.02, 0.04, 0.06], [0.08, 0.03, 0.01]],
        dtype=torch.float32,
        device=supa_device,
    ).contiguous()
    ref_render, ref_alpha = _tiled_rasterize_reference(
        flat_means2d.detach().cpu(),
        flat_conics.detach().cpu(),
        flat_colors.detach().cpu(),
        flat_opacities.detach().cpu(),
        scene["width"],
        scene["height"],
        tile_size,
        offsets,
        flatten_ids,
        backgrounds.cpu(),
    )
    native = rasterize_to_pixels_tiled_fwd(
        flat_means2d,
        flat_conics,
        flat_colors,
        flat_opacities,
        scene["width"],
        scene["height"],
        tile_size,
        offsets,
        flatten_ids,
        backgrounds,
    )
    if native is None:
        pytest.skip("SUPA tiled rasterization prototype is not available")
    native_render, native_alpha = native
    torch.testing.assert_close(native_render.cpu(), ref_render, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(native_alpha.cpu(), ref_alpha, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("calc_compensations", [False, True])
def test_supa_native_projection_pinhole_forward_matches_torch_reference(
    monkeypatch, calc_compensations
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._math import _quat_scale_to_covar_preci
    from gsplat._torch_impl import _fully_fused_projection
    from gsplat.supa._wrapper import projection_ewa_3dgs_pinhole_fwd

    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)
    cpu_covars, _ = _quat_scale_to_covar_preci(
        cpu_scene["quats"].detach(),
        cpu_scene["scales"].detach(),
        True,
        False,
        triu=False,
    )
    supa_covars = cpu_covars.to(supa_device).contiguous()

    native = projection_ewa_3dgs_pinhole_fwd(
        supa_scene["means"].detach().contiguous(),
        supa_covars,
        supa_scene["viewmats"].contiguous(),
        supa_scene["Ks"].contiguous(),
        width=8,
        height=8,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        calc_compensations=calc_compensations,
    )
    if native is None:
        pytest.skip("SUPA native extension is not available")
    radii, means2d, depths, conics, compensations = native

    ref_radii, ref_means2d, ref_depths, ref_conics, ref_compensations = (
        _fully_fused_projection(
            cpu_scene["means"].detach(),
            cpu_covars,
            cpu_scene["viewmats"],
            cpu_scene["Ks"],
            8,
            8,
            eps2d=0.3,
            near_plane=0.01,
            far_plane=1e10,
            calc_compensations=calc_compensations,
            camera_model="pinhole",
        )
    )

    torch.testing.assert_close(radii.cpu(), ref_radii, rtol=0, atol=1)
    torch.testing.assert_close(means2d.cpu(), ref_means2d, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(depths.cpu(), ref_depths, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(conics.cpu(), ref_conics, rtol=1e-4, atol=1e-5)
    if calc_compensations:
        torch.testing.assert_close(
            compensations.cpu(), ref_compensations, rtol=1e-5, atol=1e-5
        )
    else:
        assert compensations is None


@pytest.mark.parametrize("calc_compensations", [False, True])
def test_supa_native_projection_pinhole_backward_matches_torch_reference(
    monkeypatch, calc_compensations
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._math import _quat_scale_to_covar_preci
    from gsplat._torch_impl import _fully_fused_projection
    from gsplat.supa._wrapper import (
        projection_ewa_3dgs_pinhole,
        projection_ewa_3dgs_pinhole_bwd,
    )

    cpu_scene = _make_basic_scene("cpu")
    cpu_covars, _ = _quat_scale_to_covar_preci(
        cpu_scene["quats"].detach(),
        cpu_scene["scales"].detach(),
        True,
        False,
        triu=False,
    )
    cpu_means = cpu_scene["means"].detach().requires_grad_(True)
    cpu_covars = cpu_covars.detach().requires_grad_(True)
    cpu_viewmats = cpu_scene["viewmats"].detach().requires_grad_(True)

    supa_means = cpu_means.detach().to(supa_device).contiguous()
    supa_covars = cpu_covars.detach().to(supa_device).contiguous()
    supa_viewmats = cpu_viewmats.detach().to(supa_device).contiguous()
    supa_Ks = cpu_scene["Ks"].to(supa_device).contiguous()

    native = projection_ewa_3dgs_pinhole(
        supa_means,
        supa_covars,
        supa_viewmats,
        supa_Ks,
        width=8,
        height=8,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        calc_compensations=calc_compensations,
    )
    if native is None:
        pytest.skip("SUPA native extension is not available")
    radii, means2d, depths, conics, compensations = native

    ref_radii, ref_means2d, ref_depths, ref_conics, ref_compensations = (
        _fully_fused_projection(
            cpu_means,
            cpu_covars,
            cpu_viewmats,
            cpu_scene["Ks"],
            8,
            8,
            eps2d=0.3,
            near_plane=0.01,
            far_plane=1e10,
            calc_compensations=calc_compensations,
            camera_model="pinhole",
        )
    )

    assert ref_radii.to(torch.float32).prod(dim=-1).gt(0).all()
    v_means2d = torch.tensor(
        [[[0.2, -0.1], [0.05, 0.3], [-0.4, 0.15]]], dtype=torch.float32
    )
    v_depths = torch.tensor([[0.7, -0.2, 0.4]], dtype=torch.float32)
    v_conics = torch.tensor(
        [[[0.11, -0.03, 0.07], [0.05, 0.02, -0.04], [-0.06, 0.08, 0.03]]],
        dtype=torch.float32,
    )
    loss_cpu = (
        (ref_means2d * v_means2d).sum()
        + (ref_depths * v_depths).sum()
        + (ref_conics * v_conics).sum()
    )
    v_compensations_supa = None
    if calc_compensations:
        v_compensations = torch.tensor([[0.2, -0.1, 0.05]], dtype=torch.float32)
        loss_cpu = loss_cpu + (ref_compensations * v_compensations).sum()
        v_compensations_supa = v_compensations.to(supa_device).contiguous()

    loss_cpu.backward()

    native_grads = projection_ewa_3dgs_pinhole_bwd(
        supa_means,
        supa_covars,
        supa_viewmats,
        supa_Ks,
        width=8,
        height=8,
        eps2d=0.3,
        radii=radii,
        conics=conics,
        compensations=compensations,
        v_means2d=v_means2d.to(supa_device).contiguous(),
        v_depths=v_depths.to(supa_device).contiguous(),
        v_conics=v_conics.to(supa_device).contiguous(),
        v_compensations=v_compensations_supa,
    )
    if native_grads is None:
        pytest.skip("SUPA native extension is not available")
    v_means, v_covars, v_quats, v_scales, v_viewmats = native_grads

    torch.testing.assert_close(v_means.cpu(), cpu_means.grad, rtol=3e-3, atol=1e-4)
    torch.testing.assert_close(v_covars.cpu(), cpu_covars.grad, rtol=3e-3, atol=1e-4)
    assert v_quats.numel() == 0
    assert v_scales.numel() == 0
    torch.testing.assert_close(
        v_viewmats.cpu(), cpu_viewmats.grad, rtol=3e-3, atol=1e-4
    )


@pytest.mark.parametrize("calc_compensations", [False, True])
def test_supa_native_projection_pinhole_forward_accepts_quats_scales(
    monkeypatch, calc_compensations
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._math import _quat_scale_to_covar_preci
    from gsplat._torch_impl import _fully_fused_projection
    from gsplat.supa._wrapper import projection_ewa_3dgs_pinhole_fwd

    cpu_scene = _make_basic_scene("cpu")
    supa_scene = _make_basic_scene(supa_device)
    cpu_covars, _ = _quat_scale_to_covar_preci(
        cpu_scene["quats"].detach(),
        cpu_scene["scales"].detach(),
        True,
        False,
        triu=False,
    )

    native = projection_ewa_3dgs_pinhole_fwd(
        supa_scene["means"].detach().contiguous(),
        None,
        supa_scene["viewmats"].contiguous(),
        supa_scene["Ks"].contiguous(),
        width=8,
        height=8,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        calc_compensations=calc_compensations,
        quats=supa_scene["quats"].detach().contiguous(),
        scales=supa_scene["scales"].detach().contiguous(),
    )
    if native is None:
        pytest.skip("SUPA native extension is not available")
    radii, means2d, depths, conics, compensations = native

    ref_radii, ref_means2d, ref_depths, ref_conics, ref_compensations = (
        _fully_fused_projection(
            cpu_scene["means"].detach(),
            cpu_covars,
            cpu_scene["viewmats"],
            cpu_scene["Ks"],
            8,
            8,
            eps2d=0.3,
            near_plane=0.01,
            far_plane=1e10,
            calc_compensations=calc_compensations,
            camera_model="pinhole",
        )
    )

    torch.testing.assert_close(radii.cpu(), ref_radii, rtol=0, atol=1)
    torch.testing.assert_close(means2d.cpu(), ref_means2d, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(depths.cpu(), ref_depths, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(conics.cpu(), ref_conics, rtol=1e-4, atol=1e-5)
    if calc_compensations:
        torch.testing.assert_close(
            compensations.cpu(), ref_compensations, rtol=1e-5, atol=1e-5
        )
    else:
        assert compensations is None


@pytest.mark.parametrize("calc_compensations", [False, True])
def test_supa_native_projection_pinhole_backward_accepts_quats_scales(
    monkeypatch, calc_compensations
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._math import _quat_scale_to_covar_preci
    from gsplat._torch_impl import _fully_fused_projection
    from gsplat.supa._wrapper import (
        projection_ewa_3dgs_pinhole,
        projection_ewa_3dgs_pinhole_bwd,
    )

    cpu_scene = _make_basic_scene("cpu")
    cpu_means = cpu_scene["means"].detach().requires_grad_(True)
    cpu_quats = cpu_scene["quats"].detach().requires_grad_(True)
    cpu_scales = cpu_scene["scales"].detach().requires_grad_(True)
    cpu_viewmats = cpu_scene["viewmats"].detach().requires_grad_(True)
    cpu_covars, _ = _quat_scale_to_covar_preci(
        cpu_quats,
        cpu_scales,
        True,
        False,
        triu=False,
    )

    supa_means = cpu_means.detach().to(supa_device).contiguous()
    supa_quats = cpu_quats.detach().to(supa_device).contiguous()
    supa_scales = cpu_scales.detach().to(supa_device).contiguous()
    supa_viewmats = cpu_viewmats.detach().to(supa_device).contiguous()
    supa_Ks = cpu_scene["Ks"].to(supa_device).contiguous()

    native = projection_ewa_3dgs_pinhole(
        supa_means,
        None,
        supa_viewmats,
        supa_Ks,
        width=8,
        height=8,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        calc_compensations=calc_compensations,
        quats=supa_quats,
        scales=supa_scales,
    )
    if native is None:
        pytest.skip("SUPA native extension is not available")
    radii, means2d, depths, conics, compensations = native

    ref_radii, ref_means2d, ref_depths, ref_conics, ref_compensations = (
        _fully_fused_projection(
            cpu_means,
            cpu_covars,
            cpu_viewmats,
            cpu_scene["Ks"],
            8,
            8,
            eps2d=0.3,
            near_plane=0.01,
            far_plane=1e10,
            calc_compensations=calc_compensations,
            camera_model="pinhole",
        )
    )

    assert ref_radii.to(torch.float32).prod(dim=-1).gt(0).all()
    v_means2d = torch.tensor(
        [[[0.2, -0.1], [0.05, 0.3], [-0.4, 0.15]]], dtype=torch.float32
    )
    v_depths = torch.tensor([[0.7, -0.2, 0.4]], dtype=torch.float32)
    v_conics = torch.tensor(
        [[[0.11, -0.03, 0.07], [0.05, 0.02, -0.04], [-0.06, 0.08, 0.03]]],
        dtype=torch.float32,
    )
    loss_cpu = (
        (ref_means2d * v_means2d).sum()
        + (ref_depths * v_depths).sum()
        + (ref_conics * v_conics).sum()
    )
    v_compensations_supa = None
    if calc_compensations:
        v_compensations = torch.tensor([[0.2, -0.1, 0.05]], dtype=torch.float32)
        loss_cpu = loss_cpu + (ref_compensations * v_compensations).sum()
        v_compensations_supa = v_compensations.to(supa_device).contiguous()

    loss_cpu.backward()

    native_grads = projection_ewa_3dgs_pinhole_bwd(
        supa_means,
        None,
        supa_viewmats,
        supa_Ks,
        width=8,
        height=8,
        eps2d=0.3,
        radii=radii,
        conics=conics,
        compensations=compensations,
        v_means2d=v_means2d.to(supa_device).contiguous(),
        v_depths=v_depths.to(supa_device).contiguous(),
        v_conics=v_conics.to(supa_device).contiguous(),
        v_compensations=v_compensations_supa,
        quats=supa_quats,
        scales=supa_scales,
    )
    if native_grads is None:
        pytest.skip("SUPA native extension is not available")
    v_means, v_covars, v_quats, v_scales, v_viewmats = native_grads

    no_viewmat_grads = projection_ewa_3dgs_pinhole_bwd(
        supa_means,
        None,
        supa_viewmats,
        supa_Ks,
        width=8,
        height=8,
        eps2d=0.3,
        radii=radii,
        conics=conics,
        compensations=compensations,
        v_means2d=v_means2d.to(supa_device).contiguous(),
        v_depths=v_depths.to(supa_device).contiguous(),
        v_conics=v_conics.to(supa_device).contiguous(),
        v_compensations=v_compensations_supa,
        quats=supa_quats,
        scales=supa_scales,
        compute_v_viewmats=False,
        synchronize_launch=False,
    )
    if no_viewmat_grads is None:
        pytest.skip("SUPA native extension is not available")
    for actual, expected in zip(no_viewmat_grads[:4], native_grads[:4]):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=3e-3, atol=1e-4)
    assert no_viewmat_grads[4].numel() == 0

    zero_depth_grads = projection_ewa_3dgs_pinhole_bwd(
        supa_means,
        None,
        supa_viewmats,
        supa_Ks,
        width=8,
        height=8,
        eps2d=0.3,
        radii=radii,
        conics=conics,
        compensations=compensations,
        v_means2d=v_means2d.to(supa_device).contiguous(),
        v_depths=torch.zeros_like(v_depths).to(supa_device).contiguous(),
        v_conics=v_conics.to(supa_device).contiguous(),
        v_compensations=v_compensations_supa,
        quats=supa_quats,
        scales=supa_scales,
    )
    none_depth_grads = projection_ewa_3dgs_pinhole_bwd(
        supa_means,
        None,
        supa_viewmats,
        supa_Ks,
        width=8,
        height=8,
        eps2d=0.3,
        radii=radii,
        conics=conics,
        compensations=compensations,
        v_means2d=v_means2d.to(supa_device).contiguous(),
        v_depths=None,
        v_conics=v_conics.to(supa_device).contiguous(),
        v_compensations=v_compensations_supa,
        quats=supa_quats,
        scales=supa_scales,
    )
    if zero_depth_grads is None or none_depth_grads is None:
        pytest.skip("SUPA native extension is not available")
    for actual, expected in zip(none_depth_grads, zero_depth_grads):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=3e-3, atol=1e-4)

    torch.testing.assert_close(v_means.cpu(), cpu_means.grad, rtol=3e-3, atol=1e-4)
    assert v_covars.numel() == 0
    torch.testing.assert_close(v_quats.cpu(), cpu_quats.grad, rtol=3e-3, atol=1e-4)
    torch.testing.assert_close(v_scales.cpu(), cpu_scales.grad, rtol=3e-3, atol=1e-4)
    torch.testing.assert_close(
        v_viewmats.cpu(), cpu_viewmats.grad, rtol=3e-3, atol=1e-4
    )


@pytest.mark.parametrize("input_kind", ["covars", "quats_scales"])
@pytest.mark.parametrize("calc_compensations", [False, True])
def test_supa_native_projection_pinhole_autograd_matches_torch_reference(
    monkeypatch, input_kind, calc_compensations
):
    supa_device = _supa_device()
    monkeypatch.setenv("GSPLAT_USE_SUPA_NATIVE", "1")

    from gsplat._math import _quat_scale_to_covar_preci
    from gsplat._torch_impl import _fully_fused_projection
    from gsplat.supa._wrapper import projection_ewa_3dgs_pinhole

    cpu_scene = _make_basic_scene("cpu")
    cpu_means = cpu_scene["means"].detach().requires_grad_(True)
    cpu_quats = cpu_scene["quats"].detach().requires_grad_(True)
    cpu_scales = cpu_scene["scales"].detach().requires_grad_(True)
    cpu_viewmats = cpu_scene["viewmats"].detach().requires_grad_(True)
    cpu_covars, _ = _quat_scale_to_covar_preci(
        cpu_quats,
        cpu_scales,
        True,
        False,
        triu=False,
    )
    if input_kind == "covars":
        cpu_covars = cpu_covars.detach().requires_grad_(True)

    supa_means = cpu_means.detach().to(supa_device).contiguous().requires_grad_(True)
    supa_viewmats = (
        cpu_viewmats.detach().to(supa_device).contiguous().requires_grad_(True)
    )
    supa_Ks = cpu_scene["Ks"].to(supa_device).contiguous()
    supa_covars = None
    supa_quats = None
    supa_scales = None
    if input_kind == "covars":
        supa_covars = (
            cpu_covars.detach().to(supa_device).contiguous().requires_grad_(True)
        )
    else:
        supa_quats = (
            cpu_quats.detach().to(supa_device).contiguous().requires_grad_(True)
        )
        supa_scales = (
            cpu_scales.detach().to(supa_device).contiguous().requires_grad_(True)
        )

    native = projection_ewa_3dgs_pinhole(
        supa_means,
        supa_covars,
        supa_viewmats,
        supa_Ks,
        width=8,
        height=8,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        calc_compensations=calc_compensations,
        quats=supa_quats,
        scales=supa_scales,
    )
    if native is None:
        pytest.skip("SUPA native extension is not available")
    radii, means2d, depths, conics, compensations = native

    ref_radii, ref_means2d, ref_depths, ref_conics, ref_compensations = (
        _fully_fused_projection(
            cpu_means,
            cpu_covars,
            cpu_viewmats,
            cpu_scene["Ks"],
            8,
            8,
            eps2d=0.3,
            near_plane=0.01,
            far_plane=1e10,
            calc_compensations=calc_compensations,
            camera_model="pinhole",
        )
    )

    torch.testing.assert_close(radii.cpu(), ref_radii, rtol=0, atol=1)
    v_means2d = torch.tensor(
        [[[0.2, -0.1], [0.05, 0.3], [-0.4, 0.15]]], dtype=torch.float32
    )
    v_depths = torch.tensor([[0.7, -0.2, 0.4]], dtype=torch.float32)
    v_conics = torch.tensor(
        [[[0.11, -0.03, 0.07], [0.05, 0.02, -0.04], [-0.06, 0.08, 0.03]]],
        dtype=torch.float32,
    )
    loss_cpu = (
        (ref_means2d * v_means2d).sum()
        + (ref_depths * v_depths).sum()
        + (ref_conics * v_conics).sum()
    )
    loss_supa = (
        (means2d * v_means2d.to(supa_device)).sum()
        + (depths * v_depths.to(supa_device)).sum()
        + (conics * v_conics.to(supa_device)).sum()
    )
    if calc_compensations:
        v_compensations = torch.tensor([[0.2, -0.1, 0.05]], dtype=torch.float32)
        loss_cpu = loss_cpu + (ref_compensations * v_compensations).sum()
        loss_supa = loss_supa + (
            compensations * v_compensations.to(supa_device)
        ).sum()

    loss_cpu.backward()
    loss_supa.backward()

    torch.testing.assert_close(
        supa_means.grad.cpu(), cpu_means.grad, rtol=3e-3, atol=1e-4
    )
    torch.testing.assert_close(
        supa_viewmats.grad.cpu(), cpu_viewmats.grad, rtol=3e-3, atol=1e-4
    )
    if input_kind == "covars":
        torch.testing.assert_close(
            supa_covars.grad.cpu(), cpu_covars.grad, rtol=3e-3, atol=1e-4
        )
    else:
        torch.testing.assert_close(
            supa_quats.grad.cpu(), cpu_quats.grad, rtol=3e-3, atol=1e-4
        )
        torch.testing.assert_close(
            supa_scales.grad.cpu(), cpu_scales.grad, rtol=3e-3, atol=1e-4
        )
