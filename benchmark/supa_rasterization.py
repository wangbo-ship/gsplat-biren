#!/usr/bin/env /usr/bin/python3
# SPDX-License-Identifier: Apache-2.0

"""Benchmark public SUPA rasterization paths and tiled metadata helpers."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch_br  # noqa: F401

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import (
    clear_grads,
    git_head,
    make_scene,
    parse_csv,
    set_requires_grad,
    stats,
    synchronize,
    write_json,
)
from gsplat import rasterization
from gsplat._logging import configure_logging, get_logger, log_info
from gsplat.supa._wrapper import (
    isect_tiles_unpacked_offsets,
    projection_ewa_3dgs_pinhole,
    rasterize_to_pixels_tiled_fwd,
    rasterize_to_pixels_tiled_bwd,
)


LOGGER = get_logger(__name__)
_TILED_BWD_MODE_CHOICES = ("auto", "pixel", "tile-reduce", "parallel")


def _tiled_bwd_modes(args: argparse.Namespace) -> list[str]:
    value = (
        args.tiled_bwd_modes
        if args.tiled_bwd_modes is not None
        else args.tiled_bwd_mode
    )
    modes = parse_csv(value)
    if not modes:
        raise ValueError("at least one tiled backward mode is required")
    invalid = [mode for mode in modes if mode not in _TILED_BWD_MODE_CHOICES]
    if invalid:
        raise ValueError(
            "unknown tiled backward mode(s) "
            f"{invalid}; choices: {list(_TILED_BWD_MODE_CHOICES)}"
        )
    return modes


def _max_tile_range(tile_offsets: torch.Tensor, n_isects: int) -> int:
    if tile_offsets.numel() == 0:
        return 0
    flat_offsets = tile_offsets.reshape(-1)
    final_offset = torch.full(
        (1,),
        int(n_isects),
        dtype=flat_offsets.dtype,
        device=flat_offsets.device,
    )
    range_ends = torch.cat((flat_offsets[1:], final_offset), dim=0)
    return int((range_ends - flat_offsets).max().detach().cpu().item())


def _public_once(
    scene: dict[str, torch.Tensor],
    width: int,
    height: int,
    tile_size: int,
    loss_mode: str,
    need_backward: bool,
    native_options: dict[str, object],
) -> tuple[float, float | None, dict[str, Any]]:
    clear_grads(scene)
    synchronize(scene["means"].device)
    start = time.perf_counter()
    with torch.set_grad_enabled(need_backward):
        render, alpha, meta = rasterization(
            scene["means"],
            scene["quats"],
            scene["scales"],
            scene["opacities"],
            scene["colors"],
            scene["viewmats"],
            scene["Ks"],
            width=width,
            height=height,
            tile_size=tile_size,
            packed=False,
            render_mode="RGB",
            supa_native_options=native_options,
        )
        synchronize(scene["means"].device)
        mid = time.perf_counter()
        backward_time = None
        if need_backward:
            if loss_mode == "l1":
                loss = torch.nn.functional.l1_loss(render, scene["target"])
            elif loss_mode == "render":
                loss = render.mean()
            else:
                loss = render.mean() + alpha.mean()
            loss.backward()
            synchronize(scene["means"].device)
            backward_time = time.perf_counter() - mid
    return mid - start, backward_time, meta


def _projection_for_tiled(
    scene: dict[str, torch.Tensor],
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = projection_ewa_3dgs_pinhole(
        scene["means"].detach().contiguous(),
        None,
        scene["viewmats"].detach().contiguous(),
        scene["Ks"].detach().contiguous(),
        width,
        height,
        quats=scene["quats"].detach().contiguous(),
        scales=scene["scales"].detach().contiguous(),
    )
    if out is None:
        raise RuntimeError("native projection is unavailable")
    radii, means2d, depths, conics, _ = out
    return radii.contiguous(), means2d.contiguous(), depths.contiguous(), conics.contiguous()


def _metadata_once(
    scene: dict[str, torch.Tensor],
    width: int,
    height: int,
    tile_size: int,
) -> tuple[float, dict[str, Any]]:
    synchronize(scene["means"].device)
    start = time.perf_counter()
    radii, means2d, depths, conics = _projection_for_tiled(scene, width, height)
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    isects = isect_tiles_unpacked_offsets(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        synchronize_launch=False,
        use_native_offsets=True,
    )
    if isects is None:
        raise RuntimeError("native tiled metadata is unavailable")
    tiles_per_gauss, tile_offsets, flatten_ids = isects
    synchronize(scene["means"].device)
    return time.perf_counter() - start, {
        "means2d": means2d,
        "conics": conics,
        "tile_offsets": tile_offsets,
        "flatten_ids": flatten_ids,
        "tiles_per_gauss": tiles_per_gauss,
        "n_isects": int(flatten_ids.numel()),
        "tiles_per_gauss_sum": int(tiles_per_gauss.detach().cpu().sum().item()),
        "max_tile_range": _max_tile_range(tile_offsets, int(flatten_ids.numel())),
    }


def _tiled_bwd_once(
    scene: dict[str, torch.Tensor],
    width: int,
    height: int,
    tile_size: int,
    backward_mode: str,
    forward_alpha: bool,
    alpha_grad: bool,
) -> tuple[float, float, dict[str, Any]]:
    metadata_time, metadata = _metadata_once(scene, width, height, tile_size)
    means2d = metadata["means2d"]
    conics = metadata["conics"]
    features = scene["colors"].detach().reshape(1, -1, 3).expand(means2d.shape[0], -1, -1).contiguous()
    opacities = scene["opacities"].detach().reshape(1, -1).expand(means2d.shape[0], -1).contiguous()
    saved_alpha = None
    if forward_alpha:
        tiled = rasterize_to_pixels_tiled_fwd(
            means2d,
            conics,
            features,
            opacities,
            width,
            height,
            tile_size,
            metadata["tile_offsets"],
            metadata["flatten_ids"],
            None,
            synchronize_launch=True,
        )
        if tiled is None:
            raise RuntimeError("native tiled forward is unavailable")
        saved_alpha = tiled[1]
    v_render = torch.ones(
        (means2d.shape[0], height, width, scene["colors"].shape[-1]),
        dtype=torch.float32,
        device=scene["means"].device,
    )
    v_alpha = (
        torch.zeros(
            (means2d.shape[0], height, width, 1),
            dtype=torch.float32,
            device=scene["means"].device,
        )
        if alpha_grad
        else None
    )
    start = time.perf_counter()
    grads = rasterize_to_pixels_tiled_bwd(
        means2d,
        conics,
        features,
        opacities,
        width,
        height,
        tile_size,
        metadata["tile_offsets"],
        metadata["flatten_ids"],
        None,
        v_render,
        v_alpha,
        forward_alpha=saved_alpha,
        synchronize_launch=True,
        backward_mode=backward_mode,
    )
    if grads is None:
        raise RuntimeError("native tiled backward is unavailable")
    return metadata_time, time.perf_counter() - start, {
        "n_isects": metadata["n_isects"],
        "tiles_per_gauss_sum": metadata["tiles_per_gauss_sum"],
        "max_tile_range": metadata["max_tile_range"],
        "backward_mode": backward_mode,
        "forward_alpha": forward_alpha,
        "alpha_grad": alpha_grad,
    }


def _run_public(
    args: argparse.Namespace,
    scene: dict[str, torch.Tensor],
    variant: str,
) -> dict[str, Any]:
    tiled = variant.strip().lower()
    if tiled in {"0", "false", "off"}:
        tiled = "false"
    elif tiled in {"1", "true", "on"}:
        tiled = "true"
    elif tiled != "auto":
        raise ValueError(f"unknown public rasterization variant {variant!r}")
    native_options = {
        "native": True,
        "native_tiled": tiled,
        "native_tiled_metadata": tiled != "false",
        "native_tiled_train": False,
        "native_tiled_backward_mode": "auto",
        "native_alpha_threshold": 0.0,
    }
    need_backward = args.mode == "train"
    set_requires_grad(scene, need_backward)
    forward_times: list[float] = []
    backward_times: list[float] = []
    last_meta: dict[str, Any] = {}
    for _ in range(args.warmup):
        _public_once(
            scene,
            args.width,
            args.height,
            args.tile_size,
            args.loss_mode,
            need_backward,
            native_options,
        )
    for _ in range(args.iters):
        forward, backward, last_meta = _public_once(
            scene,
            args.width,
            args.height,
            args.tile_size,
            args.loss_mode,
            need_backward,
            native_options,
        )
        forward_times.append(forward)
        if backward is not None:
            backward_times.append(backward)

    return {
        "kind": "public_rasterization",
        "variant_request": variant,
        "forward_s": stats(forward_times),
        "backward_s": stats(backward_times),
        "meta": {
            "projection_backend": last_meta.get("projection_backend"),
            "covariance_backend": last_meta.get("covariance_backend"),
            "composite_backend": last_meta.get("composite_backend"),
            "composite_variant": last_meta.get("composite_variant"),
            "compact_backend": last_meta.get("compact_backend"),
            "sh_backend": last_meta.get("sh_backend"),
            "tile_count_backend": last_meta.get("tile_count_backend"),
            "visible_gaussians": last_meta.get("visible_gaussians"),
        },
    }


def _run_metadata(args: argparse.Namespace, scene: dict[str, torch.Tensor]) -> dict[str, Any]:
    set_requires_grad(scene, False)
    metadata_times: list[float] = []
    last_meta: dict[str, Any] = {}
    for _ in range(args.warmup):
        _metadata_once(scene, args.width, args.height, args.tile_size)
    for _ in range(args.iters):
        metadata_time, last_meta = _metadata_once(
            scene,
            args.width,
            args.height,
            args.tile_size,
        )
        metadata_times.append(metadata_time)
    return {
        "kind": "metadata",
        "metadata_s": stats(metadata_times),
        "meta": {
            "n_isects": last_meta.get("n_isects"),
            "tiles_per_gauss_sum": last_meta.get("tiles_per_gauss_sum"),
            "max_tile_range": last_meta.get("max_tile_range"),
        },
    }


def _run_tiled_bwd(
    args: argparse.Namespace,
    scene: dict[str, torch.Tensor],
    backward_mode: str,
) -> dict[str, Any]:
    set_requires_grad(scene, False)
    metadata_times: list[float] = []
    backward_times: list[float] = []
    last_meta: dict[str, Any] = {}
    for _ in range(args.warmup):
        _tiled_bwd_once(
            scene,
            args.width,
            args.height,
            args.tile_size,
            backward_mode,
            args.tiled_bwd_forward_alpha == "true",
            args.tiled_bwd_alpha_grad == "true",
        )
    for _ in range(args.iters):
        metadata, backward, last_meta = _tiled_bwd_once(
            scene,
            args.width,
            args.height,
            args.tile_size,
            backward_mode,
            args.tiled_bwd_forward_alpha == "true",
            args.tiled_bwd_alpha_grad == "true",
        )
        metadata_times.append(metadata)
        backward_times.append(backward)
    return {
        "kind": "direct_tiled_backward",
        "metadata_s": stats(metadata_times),
        "backward_s": stats(backward_times),
        "meta": last_meta,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="supa:0")
    parser.add_argument("--scene", choices=("synthetic", "test-data", "npz"), default="synthetic")
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--gaussians", type=int, default=4096)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--cameras", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=("forward", "train", "metadata", "tiled-bwd"),
        default="forward",
    )
    parser.add_argument("--loss-mode", choices=("render-alpha", "render", "l1"), default="l1")
    parser.add_argument(
        "--tiled-bwd-mode",
        choices=_TILED_BWD_MODE_CHOICES,
        default="auto",
    )
    parser.add_argument(
        "--tiled-bwd-modes",
        default=None,
        help=(
            "Comma-separated tiled backward modes to run in one benchmark. "
            "Overrides --tiled-bwd-mode."
        ),
    )
    parser.add_argument(
        "--tiled-bwd-forward-alpha",
        choices=("false", "true"),
        default="false",
    )
    parser.add_argument(
        "--tiled-bwd-alpha-grad",
        choices=("false", "true"),
        default="true",
        help=(
            "Pass an explicit zero alpha gradient to tiled backward when true. "
            "Use false to emulate render-only training losses."
        ),
    )
    parser.add_argument(
        "--variants",
        default="false,auto,true",
        help="Comma-separated native_tiled values for forward/train: false, auto, true.",
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = _parse_args()
    device = torch.device(args.device)
    scene = make_scene(args, device)
    tiled_bwd_modes = _tiled_bwd_modes(args)
    config = {
        "scene": args.scene,
        "npz": str(args.npz) if args.npz else None,
        "device": str(device),
        "gaussians": int(scene["means"].shape[0]),
        "cameras": int(scene["viewmats"].shape[0]),
        "size": [args.width, args.height],
        "tile_size": args.tile_size,
        "mode": args.mode,
        "tiled_bwd_mode": args.tiled_bwd_mode,
        "tiled_bwd_modes": tiled_bwd_modes,
        "tiled_bwd_forward_alpha": args.tiled_bwd_forward_alpha,
        "tiled_bwd_alpha_grad": args.tiled_bwd_alpha_grad,
        "warmup": args.warmup,
        "iters": args.iters,
    }

    if args.mode == "metadata":
        runs = [_run_metadata(args, scene)]
    elif args.mode == "tiled-bwd":
        runs = [_run_tiled_bwd(args, scene, mode) for mode in tiled_bwd_modes]
    else:
        runs = [_run_public(args, scene, variant) for variant in parse_csv(args.variants)]

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "torch": torch.__version__,
        "config": config,
        "runs": runs,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    log_info(LOGGER, text)
    write_json(args.json_out, text)


if __name__ == "__main__":
    main()
