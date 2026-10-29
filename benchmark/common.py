# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for gsplat-biren benchmark scripts."""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ensure_project_root() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "min": None, "max": None, "p90": None}
    return {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "p90": quantile(values, 0.9),
    }


def synchronize(device: torch.device | str | None = None) -> None:
    from gsplat._device import synchronize as sync

    sync(device)


def time_sync(
    name: str,
    fn: Callable[[], Any],
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
        synchronize(device)
    samples: list[float] = []
    for _ in range(iters):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    return {"name": name, "time_s": stats(samples), "samples_s": samples}


def make_synthetic_scene(
    device: torch.device,
    n_gaussians: int,
    width: int,
    height: int,
    n_cameras: int,
    *,
    seed: int = 12345,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    means = torch.empty((n_gaussians, 3), dtype=torch.float32)
    means[:, 0] = (torch.rand(n_gaussians, generator=gen) - 0.5) * 1.2
    means[:, 1] = (torch.rand(n_gaussians, generator=gen) - 0.5) * 0.9
    means[:, 2] = 2.0 + torch.rand(n_gaussians, generator=gen) * 2.0

    quats = torch.randn((n_gaussians, 4), generator=gen, dtype=torch.float32)
    quats[:, 0] += 2.0
    scales = 0.012 + torch.rand((n_gaussians, 3), generator=gen) * 0.018
    opacities = 0.08 + torch.rand(n_gaussians, generator=gen) * 0.35
    colors = torch.rand((n_gaussians, 3), generator=gen, dtype=torch.float32)

    viewmats = torch.eye(4, dtype=torch.float32).repeat(n_cameras, 1, 1)
    if n_cameras > 1:
        viewmats[:, 0, 3] = torch.linspace(-0.08, 0.08, n_cameras)
    Ks = torch.tensor(
        [
            [float(width), 0.0, width * 0.5],
            [0.0, float(width), height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    ).repeat(n_cameras, 1, 1)
    target = torch.rand((n_cameras, height, width, 3), generator=gen, dtype=torch.float32)
    return {
        "means": means.to(device).contiguous(),
        "quats": quats.to(device).contiguous(),
        "scales": scales.to(device).contiguous(),
        "opacities": opacities.to(device).contiguous(),
        "colors": colors.to(device).contiguous(),
        "viewmats": viewmats.to(device).contiguous(),
        "Ks": Ks.to(device).contiguous(),
        "target": target.to(device).contiguous(),
    }


def make_test_data_scene(
    device: torch.device,
    n_gaussians: int,
    width: int,
    height: int,
    n_cameras: int,
) -> dict[str, torch.Tensor]:
    from gsplat._helper import load_test_data

    means, quats, scales, opacities, colors, viewmats, Ks, orig_width, orig_height = (
        load_test_data(device="cpu")
    )
    n = min(n_gaussians, means.shape[0])
    frame_count = min(n_cameras, viewmats.shape[0])
    scale_x = float(width) / float(orig_width)
    scale_y = float(height) / float(orig_height)
    viewmats = viewmats[:frame_count].clone()
    Ks = Ks[:frame_count].clone()
    Ks[:, 0, :] *= scale_x
    Ks[:, 1, :] *= scale_y
    target = torch.rand((frame_count, height, width, 3), dtype=torch.float32)
    return {
        "means": means[:n].contiguous().to(device),
        "quats": quats[:n].contiguous().to(device),
        "scales": scales[:n].contiguous().to(device),
        "opacities": opacities[:n].contiguous().to(device),
        "colors": colors[:n].contiguous().to(device),
        "viewmats": viewmats.contiguous().to(device),
        "Ks": Ks.contiguous().to(device),
        "target": target.to(device).contiguous(),
    }


def load_npz_scene(
    path: Path,
    device: torch.device,
    n_gaussians: int,
    width: int,
    height: int,
    n_cameras: int,
) -> dict[str, torch.Tensor]:
    import numpy as np

    data = np.load(path)
    images = torch.from_numpy(data["images"]).float()
    if images.max().item() > 1.5:
        images = images / 255.0
    if "viewmats" in data:
        viewmats = torch.from_numpy(data["viewmats"]).float()
    elif "camtoworlds" in data:
        viewmats = torch.linalg.inv(torch.from_numpy(data["camtoworlds"]).float())
    else:
        raise ValueError(f"{path} does not contain viewmats or camtoworlds")
    Ks = torch.from_numpy(data["Ks"]).float()

    if "points" in data:
        means = torch.from_numpy(data["points"]).float()
    elif "means" in data:
        means = torch.from_numpy(data["means"]).float()
    else:
        raise ValueError(f"{path} does not contain points or means")
    if "point_colors" in data:
        colors = torch.from_numpy(data["point_colors"][..., :3]).float()
    elif "colors" in data and data["colors"].shape[0] == means.shape[0]:
        colors = torch.from_numpy(data["colors"][..., :3]).float()
    else:
        colors = torch.full((means.shape[0], 3), 0.5, dtype=torch.float32)
    if colors.numel() > 0 and colors.max().item() > 1.5:
        colors = colors / 255.0

    n = min(n_gaussians, means.shape[0])
    frame_count = min(n_cameras, viewmats.shape[0])
    orig_height, orig_width = images.shape[1:3]
    scale_x = float(width) / float(orig_width)
    scale_y = float(height) / float(orig_height)
    Ks = Ks[:frame_count].clone()
    Ks[:, 0, :] *= scale_x
    Ks[:, 1, :] *= scale_y

    gen = torch.Generator(device="cpu")
    gen.manual_seed(54321)
    quats = torch.randn((n, 4), generator=gen, dtype=torch.float32)
    quats[:, 0] += 2.0
    scales = torch.full((n, 3), 0.01, dtype=torch.float32)
    opacities = torch.full((n,), 0.25, dtype=torch.float32)
    target = torch.nn.functional.interpolate(
        images[:frame_count].permute(0, 3, 1, 2),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    return {
        "means": means[:n].contiguous().to(device),
        "quats": quats.contiguous().to(device),
        "scales": scales.contiguous().to(device),
        "opacities": opacities.contiguous().to(device),
        "colors": colors[:n].contiguous().to(device),
        "viewmats": viewmats[:frame_count].contiguous().to(device),
        "Ks": Ks.contiguous().to(device),
        "target": target.contiguous().to(device),
    }


def make_scene(args: Any, device: torch.device) -> dict[str, torch.Tensor]:
    if args.scene == "synthetic":
        return make_synthetic_scene(
            device,
            args.gaussians,
            args.width,
            args.height,
            args.cameras,
        )
    if args.scene == "test-data":
        return make_test_data_scene(
            device,
            args.gaussians,
            args.width,
            args.height,
            args.cameras,
        )
    if args.scene == "npz":
        if args.npz is None:
            raise ValueError("--npz is required when --scene npz")
        return load_npz_scene(
            Path(args.npz),
            device,
            args.gaussians,
            args.width,
            args.height,
            args.cameras,
        )
    raise ValueError(f"unknown scene {args.scene!r}")


def set_requires_grad(scene: dict[str, torch.Tensor], value: bool) -> None:
    for key in ("means", "quats", "scales", "opacities", "colors"):
        scene[key].requires_grad_(value)


def clear_grads(scene: dict[str, torch.Tensor]) -> None:
    for key in ("means", "quats", "scales", "opacities", "colors"):
        scene[key].grad = None


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_json(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
