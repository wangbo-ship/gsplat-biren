#!/usr/bin/env /usr/bin/python3
# SPDX-License-Identifier: Apache-2.0

"""Operator-level SUPA benchmarks for native gsplat bring-up paths."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch_br  # noqa: F401

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import git_head, time_sync, write_json
from gsplat import rasterization
from gsplat._logging import configure_logging, get_logger, log_info
from gsplat._torch_fallback import (
    _SupaCompactSelect,
    _alpha_composite,
    _visible_gaussian_ids,
)
from gsplat._torch_impl import _spherical_harmonics
from gsplat.supa._wrapper import (
    alpha_composite,
    alpha_composite_fwd,
    compact_select,
    isect_tiles_unpacked_lanes,
    isect_tiles_unpacked_offsets,
    projection_ewa_3dgs_pinhole,
    projection_ewa_3dgs_pinhole_bwd,
    spherical_harmonics,
    tile_count_unpacked,
    tiled_offsets_from_lanes,
    visible_compact_ids,
)


LOGGER = get_logger(__name__)


def _make_projection_scene(
    device: torch.device,
    n_gaussians: int,
    width: int,
    height: int,
    cameras: int,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20260624)
    means = torch.empty((n_gaussians, 3), dtype=torch.float32)
    means[:, 0] = (torch.rand(n_gaussians, generator=gen) - 0.5) * 1.1
    means[:, 1] = (torch.rand(n_gaussians, generator=gen) - 0.5) * 0.8
    means[:, 2] = 2.0 + torch.rand(n_gaussians, generator=gen) * 2.0
    quats = torch.randn((n_gaussians, 4), generator=gen, dtype=torch.float32)
    quats[:, 0] += 2.0
    scales = 0.012 + torch.rand((n_gaussians, 3), generator=gen) * 0.018
    opacities = 0.08 + torch.rand(n_gaussians, generator=gen) * 0.35
    colors = torch.rand((n_gaussians, 3), generator=gen, dtype=torch.float32)
    viewmats = torch.eye(4, dtype=torch.float32).repeat(cameras, 1, 1)
    if cameras > 1:
        viewmats[:, 0, 3] = torch.linspace(-0.08, 0.08, cameras)
    Ks = torch.tensor(
        [
            [float(width), 0.0, width * 0.5],
            [0.0, float(width), height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    ).repeat(cameras, 1, 1)
    return {
        "means": means.to(device).contiguous(),
        "quats": quats.to(device).contiguous(),
        "scales": scales.to(device).contiguous(),
        "opacities": opacities.to(device).contiguous(),
        "colors": colors.to(device).contiguous(),
        "viewmats": viewmats.to(device).contiguous(),
        "Ks": Ks.to(device).contiguous(),
    }


def _project_for_metadata(
    scene: dict[str, torch.Tensor],
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = projection_ewa_3dgs_pinhole(
        scene["means"],
        None,
        scene["viewmats"],
        scene["Ks"],
        width,
        height,
        quats=scene["quats"],
        scales=scene["scales"],
        synchronize_launch=True,
    )
    if out is None:
        raise RuntimeError("native projection is unavailable")
    radii, means2d, depths, conics, _ = out
    return radii.contiguous(), means2d.contiguous(), depths.contiguous(), conics.contiguous()


def _bench_sh(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1201)
    dirs_base = torch.randn(args.sh_items, 3, generator=gen).to(device).contiguous()
    coeffs_base = torch.randn(args.sh_items, 25, 3, generator=gen).to(device).contiguous()
    v_colors = torch.randn(args.sh_items, 3, generator=gen).to(device).contiguous()

    def torch_fwd() -> None:
        _spherical_harmonics(args.sh_degree, dirs_base, coeffs_base)

    def native_fwd() -> None:
        out = spherical_harmonics(
            args.sh_degree,
            dirs_base,
            coeffs_base,
            synchronize_launch=False,
        )
        if out is None:
            raise RuntimeError("native SH is unavailable")

    def torch_bwd() -> None:
        dirs = dirs_base.detach().clone().requires_grad_(True)
        coeffs = coeffs_base.detach().clone().requires_grad_(True)
        (_spherical_harmonics(args.sh_degree, dirs, coeffs) * v_colors).sum().backward()

    def native_bwd() -> None:
        dirs = dirs_base.detach().clone().requires_grad_(True)
        coeffs = coeffs_base.detach().clone().requires_grad_(True)
        out = spherical_harmonics(
            args.sh_degree,
            dirs,
            coeffs,
            synchronize_launch=False,
        )
        if out is None:
            raise RuntimeError("native SH is unavailable")
        (out * v_colors).sum().backward()

    return {
        "operator": "spherical_harmonics",
        "config": {"items": args.sh_items, "degree": args.sh_degree},
        "runs": [
            time_sync("torch_forward", torch_fwd, device, args.warmup, args.iters),
            time_sync("native_forward", native_fwd, device, args.warmup, args.iters),
            time_sync("torch_forward_backward", torch_bwd, device, args.warmup, args.iters),
            time_sync("native_forward_backward", native_bwd, device, args.warmup, args.iters),
        ],
    }


def _bench_compact(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1301)
    radii_cpu = torch.randint(
        0,
        5,
        (args.compact_rows, args.compact_gaussians, 2),
        generator=gen,
        dtype=torch.int32,
    )
    keep = torch.rand((args.compact_rows, args.compact_gaussians), generator=gen) < (
        args.compact_visible_frac
    )
    radii_cpu[..., 0] = torch.where(
        keep,
        torch.clamp_min(radii_cpu[..., 0], 1),
        torch.zeros_like(radii_cpu[..., 0]),
    )
    radii_cpu[..., 1] = torch.where(
        keep,
        torch.clamp_min(radii_cpu[..., 1], 1),
        torch.zeros_like(radii_cpu[..., 1]),
    )
    depths_cpu = torch.rand((args.compact_rows, args.compact_gaussians), generator=gen)
    tensor_cpu = torch.randn(
        args.compact_rows,
        args.compact_gaussians,
        args.compact_channels,
        generator=gen,
    )
    radii = radii_cpu.to(device).contiguous()
    depths = depths_cpu.to(device).contiguous()
    tensor_base = tensor_cpu.to(device).contiguous()
    ids_long, visible_count = _visible_gaussian_ids(radii, depths, args.compact_gaussians)
    ids_i32 = ids_long.to(device=device, dtype=torch.int32).contiguous()

    def torch_visible() -> None:
        _visible_gaussian_ids(radii, depths, args.compact_gaussians)

    def native_visible() -> None:
        out = visible_compact_ids(radii, depths, synchronize_launch=False)
        if out is None:
            raise RuntimeError("native visible_compact_ids is unavailable")
        out[1]

    def torch_select_bwd() -> None:
        tensor = tensor_base.detach().clone().requires_grad_(True)
        _SupaCompactSelect.apply(tensor, ids_long, 1).sum().backward()

    def native_select_bwd() -> None:
        tensor = tensor_base.detach().clone().requires_grad_(True)
        out = compact_select(tensor, ids_i32, 1)
        if out is None:
            raise RuntimeError("native compact_select is unavailable")
        out.sum().backward()

    return {
        "operator": "compact",
        "config": {
            "rows": args.compact_rows,
            "gaussians": args.compact_gaussians,
            "channels": args.compact_channels,
            "visible_count": int(visible_count),
            "visible_frac": float(args.compact_visible_frac),
        },
        "runs": [
            time_sync("torch_visible_ids", torch_visible, device, args.warmup, args.iters),
            time_sync("native_visible_ids", native_visible, device, args.warmup, args.iters),
            time_sync(
                "torch_select_forward_backward",
                torch_select_bwd,
                device,
                args.warmup,
                args.iters,
            ),
            time_sync(
                "native_select_forward_backward",
                native_select_bwd,
                device,
                args.warmup,
                args.iters,
            ),
        ],
    }


def _make_composite_inputs(
    device: torch.device,
    images: int,
    gaussians: int,
    channels: int,
    width: int,
    height: int,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1401)
    means2d = torch.empty((images, gaussians, 2), dtype=torch.float32)
    means2d[..., 0] = torch.rand((images, gaussians), generator=gen) * width
    means2d[..., 1] = torch.rand((images, gaussians), generator=gen) * height
    conics = torch.zeros((images, gaussians, 3), dtype=torch.float32)
    conics[..., 0] = 0.02 + torch.rand((images, gaussians), generator=gen) * 0.05
    conics[..., 2] = 0.02 + torch.rand((images, gaussians), generator=gen) * 0.05
    features = torch.rand((images, gaussians, channels), generator=gen)
    opacities = 0.03 + torch.rand((images, gaussians), generator=gen) * 0.45
    depths = torch.rand((images, gaussians), generator=gen) * 4.0 + 0.5
    radii = torch.full((images, gaussians, 2), 4, dtype=torch.int32)
    return {
        "means2d": means2d.to(device).contiguous(),
        "conics": conics.to(device).contiguous(),
        "features": features.to(device).contiguous(),
        "opacities": opacities.to(device).contiguous(),
        "depths": depths.to(device).contiguous(),
        "radii": radii.to(device).contiguous(),
    }


def _bench_composite(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    data = _make_composite_inputs(
        device,
        args.composite_images,
        args.composite_gaussians,
        args.composite_channels,
        args.width,
        args.height,
    )

    def torch_fwd() -> None:
        with torch.no_grad():
            _alpha_composite(
                data["means2d"],
                data["conics"],
                data["features"],
                data["opacities"],
                data["depths"],
                data["radii"],
                args.width,
                args.height,
                None,
                chunk_size=args.composite_chunk,
            )

    def native_fwd() -> None:
        with torch.no_grad():
            out = alpha_composite_fwd(
                data["means2d"],
                data["conics"],
                data["features"],
                data["opacities"],
                data["depths"],
                data["radii"],
                args.width,
                args.height,
                None,
                synchronize_launch=False,
            )
        if out is None:
            raise RuntimeError("native alpha_composite_fwd is unavailable")

    def native_autograd_fwd() -> None:
        out = alpha_composite(
            data["means2d"],
            data["conics"],
            data["features"],
            data["opacities"],
            data["depths"],
            data["radii"],
            args.width,
            args.height,
            None,
        )
        if out is None:
            raise RuntimeError("native alpha_composite is unavailable")

    def torch_bwd() -> None:
        means2d = data["means2d"].detach().clone().requires_grad_(True)
        conics = data["conics"].detach().clone().requires_grad_(True)
        features = data["features"].detach().clone().requires_grad_(True)
        opacities = data["opacities"].detach().clone().requires_grad_(True)
        render, alpha = _alpha_composite(
            means2d,
            conics,
            features,
            opacities,
            data["depths"],
            data["radii"],
            args.width,
            args.height,
            None,
            chunk_size=args.composite_chunk,
        )
        (render.mean() + alpha.mean()).backward()

    def native_bwd() -> None:
        means2d = data["means2d"].detach().clone().requires_grad_(True)
        conics = data["conics"].detach().clone().requires_grad_(True)
        features = data["features"].detach().clone().requires_grad_(True)
        opacities = data["opacities"].detach().clone().requires_grad_(True)
        out = alpha_composite(
            means2d,
            conics,
            features,
            opacities,
            data["depths"],
            data["radii"],
            args.width,
            args.height,
            None,
        )
        if out is None:
            raise RuntimeError("native alpha_composite is unavailable")
        render, alpha = out
        (render.mean() + alpha.mean()).backward()

    return {
        "operator": "alpha_composite",
        "config": {
            "images": args.composite_images,
            "gaussians": args.composite_gaussians,
            "channels": args.composite_channels,
            "size": [args.width, args.height],
            "chunk_size": args.composite_chunk,
        },
        "runs": [
            time_sync("torch_forward", torch_fwd, device, args.warmup, args.iters),
            time_sync("native_forward", native_fwd, device, args.warmup, args.iters),
            time_sync(
                "native_autograd_forward",
                native_autograd_fwd,
                device,
                args.warmup,
                args.iters,
            ),
            time_sync("torch_forward_backward", torch_bwd, device, args.warmup, args.iters),
            time_sync("native_forward_backward", native_bwd, device, args.warmup, args.iters),
        ],
    }


def _bench_projection(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scene = _make_projection_scene(
        device,
        args.projection_gaussians,
        args.width,
        args.height,
        args.projection_cameras,
    )
    forward = projection_ewa_3dgs_pinhole(
        scene["means"],
        None,
        scene["viewmats"],
        scene["Ks"],
        args.width,
        args.height,
        quats=scene["quats"],
        scales=scene["scales"],
        synchronize_launch=True,
    )
    if forward is None:
        raise RuntimeError("native projection is unavailable")
    radii, means2d, _depths, conics, _compensations = forward
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1601)
    v_means2d = torch.randn(tuple(means2d.shape), generator=gen).to(device).contiguous()
    v_conics = torch.randn(tuple(conics.shape), generator=gen).to(device).contiguous()

    def direct_bwd_all() -> None:
        out = projection_ewa_3dgs_pinhole_bwd(
            scene["means"],
            None,
            scene["viewmats"],
            scene["Ks"],
            args.width,
            args.height,
            0.3,
            radii,
            conics,
            None,
            v_means2d,
            None,
            v_conics,
            None,
            quats=scene["quats"],
            scales=scene["scales"],
            compute_v_viewmats=True,
        )
        if out is None:
            raise RuntimeError("native projection backward is unavailable")

    def direct_bwd_no_viewmats() -> None:
        out = projection_ewa_3dgs_pinhole_bwd(
            scene["means"],
            None,
            scene["viewmats"],
            scene["Ks"],
            args.width,
            args.height,
            0.3,
            radii,
            conics,
            None,
            v_means2d,
            None,
            v_conics,
            None,
            quats=scene["quats"],
            scales=scene["scales"],
            compute_v_viewmats=False,
        )
        if out is None:
            raise RuntimeError("native projection backward is unavailable")

    def make_autograd_inputs(viewmats_require_grad: bool) -> dict[str, torch.Tensor]:
        inputs = {
            key: scene[key].detach().clone()
            for key in ("means", "quats", "scales", "viewmats", "Ks")
        }
        for key in ("means", "quats", "scales"):
            inputs[key].requires_grad_(True)
        inputs["viewmats"].requires_grad_(viewmats_require_grad)
        return inputs

    autograd_no_viewmat = make_autograd_inputs(False)
    autograd_with_viewmat = make_autograd_inputs(True)

    def clear_grads(inputs: dict[str, torch.Tensor]) -> None:
        for key in ("means", "quats", "scales", "viewmats"):
            inputs[key].grad = None

    def autograd_bwd(inputs: dict[str, torch.Tensor]) -> None:
        clear_grads(inputs)
        out = projection_ewa_3dgs_pinhole(
            inputs["means"],
            None,
            inputs["viewmats"],
            inputs["Ks"],
            args.width,
            args.height,
            quats=inputs["quats"],
            scales=inputs["scales"],
            synchronize_launch=False,
        )
        if out is None:
            raise RuntimeError("native projection is unavailable")
        _radii, out_means2d, _out_depths, out_conics, _out_compensations = out
        ((out_means2d * v_means2d).sum() + (out_conics * v_conics).sum()).backward()

    return {
        "operator": "projection_ewa_3dgs_pinhole",
        "config": {
            "gaussians": args.projection_gaussians,
            "cameras": args.projection_cameras,
            "size": [args.width, args.height],
        },
        "runs": [
            time_sync("native_backward_all_grads", direct_bwd_all, device, args.warmup, args.iters),
            time_sync(
                "native_backward_no_viewmats",
                direct_bwd_no_viewmats,
                device,
                args.warmup,
                args.iters,
            ),
            time_sync(
                "native_autograd_no_viewmat_grad",
                lambda: autograd_bwd(autograd_no_viewmat),
                device,
                args.warmup,
                args.iters,
            ),
            time_sync(
                "native_autograd_with_viewmat_grad",
                lambda: autograd_bwd(autograd_with_viewmat),
                device,
                args.warmup,
                args.iters,
            ),
        ],
    }


def _bench_metadata(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scene = _make_projection_scene(
        device,
        args.metadata_gaussians,
        args.width,
        args.height,
        args.metadata_cameras,
    )
    radii, means2d, depths, _ = _project_for_metadata(scene, args.width, args.height)
    tile_width = math.ceil(args.width / float(args.tile_size))
    tile_height = math.ceil(args.height / float(args.tile_size))
    last_meta: dict[str, int] = {}
    setup_tiles = tile_count_unpacked(
        means2d,
        radii,
        args.tile_size,
        tile_width,
        tile_height,
        synchronize_launch=True,
    )
    if setup_tiles is None:
        raise RuntimeError("native tile_count_unpacked is unavailable")
    setup_lanes = isect_tiles_unpacked_lanes(
        means2d,
        radii,
        depths,
        setup_tiles,
        args.tile_size,
        tile_width,
        tile_height,
        synchronize_launch=True,
        return_cpu=False,
    )
    if setup_lanes is None:
        raise RuntimeError("native isect_tiles_unpacked_lanes is unavailable")

    def run_metadata(use_native_offsets: bool) -> None:
        out = isect_tiles_unpacked_offsets(
            means2d,
            radii,
            depths,
            args.tile_size,
            tile_width,
            tile_height,
            synchronize_launch=False,
            return_backend=True,
            use_native_offsets=use_native_offsets,
        )
        if out is None:
            raise RuntimeError("tiled metadata is unavailable")
        tiles_per_gauss, tile_offsets, flatten_ids, backend = out
        last_meta.clear()
        last_meta.update(
            {
                "metadata_backend": backend,
                "tiles_per_gauss_sum": int(tiles_per_gauss.detach().cpu().sum().item()),
                "tile_offsets_numel": int(tile_offsets.numel()),
                "flatten_ids_numel": int(flatten_ids.numel()),
            }
        )

    def native_tile_count_only() -> None:
        out = tile_count_unpacked(
            means2d,
            radii,
            args.tile_size,
            tile_width,
            tile_height,
            synchronize_launch=False,
        )
        if out is None:
            raise RuntimeError("native tile_count_unpacked is unavailable")

    def native_lanes_only() -> None:
        out = isect_tiles_unpacked_lanes(
            means2d,
            radii,
            depths,
            setup_tiles,
            args.tile_size,
            tile_width,
            tile_height,
            synchronize_launch=False,
            return_cpu=False,
        )
        if out is None:
            raise RuntimeError("native isect_tiles_unpacked_lanes is unavailable")

    def native_offsets_only() -> None:
        out = tiled_offsets_from_lanes(
            setup_lanes[0],
            setup_lanes[1],
            setup_lanes[3],
            int(means2d.shape[0]),
            tile_width,
            tile_height,
            synchronize_launch=False,
        )
        if out is None:
            raise RuntimeError("native tiled_offsets_from_lanes is unavailable")

    return {
        "operator": "tiled_metadata",
        "config": {
            "gaussians": args.metadata_gaussians,
            "cameras": args.metadata_cameras,
            "size": [args.width, args.height],
            "tile_size": args.tile_size,
        },
        "runs": [
            time_sync(
                "cpu_numpy_offsets",
                lambda: run_metadata(False),
                device,
                args.warmup,
                args.iters,
            ),
            time_sync(
                "native_offsets",
                lambda: run_metadata(True),
                device,
                args.warmup,
                args.iters,
            ),
            time_sync(
                "native_tile_count_only",
                native_tile_count_only,
                device,
                args.warmup,
                args.iters,
            ),
            time_sync("native_lanes_only", native_lanes_only, device, args.warmup, args.iters),
            time_sync(
                "legacy_custom_native_offsets_from_lanes_only",
                native_offsets_only,
                device,
                args.warmup,
                args.iters,
            ),
        ],
        "meta": last_meta,
    }


def _bench_public_train(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scene = _make_projection_scene(
        device,
        args.public_gaussians,
        args.width,
        args.height,
        args.public_cameras,
    )
    target = torch.rand(
        (args.public_cameras, args.height, args.width, 3),
        generator=torch.Generator(device="cpu").manual_seed(1501),
    ).to(device)
    for key in ("means", "quats", "scales", "opacities", "colors"):
        scene[key].requires_grad_(True)

    def clear_scene_grads() -> None:
        for key in ("means", "quats", "scales", "opacities", "colors"):
            scene[key].grad = None

    def run_public(name: str, native_options: dict[str, object]) -> dict[str, Any]:
        last_meta: dict[str, Any] = {}

        def once() -> None:
            nonlocal last_meta
            clear_scene_grads()
            render, alpha, meta = rasterization(
                scene["means"],
                scene["quats"],
                scene["scales"],
                scene["opacities"],
                scene["colors"],
                scene["viewmats"],
                scene["Ks"],
                width=args.width,
                height=args.height,
                tile_size=args.tile_size,
                packed=False,
                render_mode="RGB",
                supa_native_options=native_options,
            )
            last_meta = meta
            (torch.nn.functional.l1_loss(render, target) + alpha.mean()).backward()

        result = time_sync(name, once, device, args.public_warmup, args.public_iters)
        result["meta"] = {
            "projection_backend": last_meta.get("projection_backend"),
            "covariance_backend": last_meta.get("covariance_backend"),
            "composite_backend": last_meta.get("composite_backend"),
            "composite_variant": last_meta.get("composite_variant"),
            "compact_backend": last_meta.get("compact_backend"),
            "sh_backend": last_meta.get("sh_backend"),
            "tile_count_backend": last_meta.get("tile_count_backend"),
            "visible_gaussians": last_meta.get("visible_gaussians"),
        }
        return result

    return {
        "operator": "public_train",
        "config": {
            "gaussians": args.public_gaussians,
            "cameras": args.public_cameras,
            "size": [args.width, args.height],
            "tile_size": args.tile_size,
            "warmup": args.public_warmup,
            "iters": args.public_iters,
        },
        "runs": [
            run_public(
                "torch_fallback_no_native",
                {
                    "native": False,
                    "native_tiled": "false",
                    "native_tiled_metadata": False,
                    "native_tiled_train": False,
                    "native_tiled_backward_mode": "auto",
                    "native_alpha_threshold": 0.0,
                },
            ),
            run_public(
                "native_core_opt_in",
                {
                    "native": True,
                    "native_tiled": "false",
                    "native_tiled_metadata": False,
                    "native_tiled_train": False,
                    "native_tiled_backward_mode": "auto",
                    "native_alpha_threshold": 0.0,
                },
            ),
        ],
    }


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="supa:0")
    parser.add_argument("--bench", default="sh,compact,composite,projection,metadata,public-train")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=7)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--sh-items", type=int, default=65536)
    parser.add_argument("--sh-degree", type=int, default=4)
    parser.add_argument("--compact-rows", type=int, default=4)
    parser.add_argument("--compact-gaussians", type=int, default=32768)
    parser.add_argument("--compact-channels", type=int, default=8)
    parser.add_argument("--compact-visible-frac", type=float, default=0.45)
    parser.add_argument("--composite-images", type=int, default=1)
    parser.add_argument("--composite-gaussians", type=int, default=384)
    parser.add_argument("--composite-channels", type=int, default=3)
    parser.add_argument("--composite-chunk", type=int, default=32)
    parser.add_argument("--projection-gaussians", type=int, default=65535)
    parser.add_argument("--projection-cameras", type=int, default=1)
    parser.add_argument("--metadata-gaussians", type=int, default=4096)
    parser.add_argument("--metadata-cameras", type=int, default=1)
    parser.add_argument("--public-gaussians", type=int, default=2048)
    parser.add_argument("--public-cameras", type=int, default=1)
    parser.add_argument("--public-warmup", type=int, default=1)
    parser.add_argument("--public-iters", type=int, default=3)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = _parse_args()
    device = torch.device(args.device)
    benchmarks: dict[str, Callable[[argparse.Namespace, torch.device], dict[str, Any]]] = {
        "sh": _bench_sh,
        "compact": _bench_compact,
        "composite": _bench_composite,
        "projection": _bench_projection,
        "metadata": _bench_metadata,
        "public-train": _bench_public_train,
    }
    results = []
    for name in _parse_csv(args.bench):
        if name not in benchmarks:
            raise ValueError(f"unknown benchmark {name!r}; choices: {sorted(benchmarks)}")
        results.append(benchmarks[name](args, device))

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "device": str(device),
        "torch": torch.__version__,
        "benchmarks": results,
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    log_info(LOGGER, text)
    write_json(args.json_out, text)


if __name__ == "__main__":
    main()
