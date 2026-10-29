# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SUPA-safe multi-view 3DGS trainer.

This application targets the current Biren bring-up surface: basic pinhole
3DGS, unpacked rasterization, no distortion, opt-in single-node SUPA
distributed training, and trainer-local Gaussian refinement.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import shutil
import socket
import time
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from PIL import Image
from torch import Tensor

from gsplat._device import (
    accelerator_device_count,
    distributed_backend,
    ensure_supa_runtime,
    resolve_device,
    set_accelerator_device,
    synchronize,
)
from gsplat._logging import (
    configure_logging,
    format_key_value_table,
    get_logger,
    gsplat_biren_ascii_art,
    log_info,
    log_warning,
)
from gsplat.utils import normalized_quat_to_rotmat


LOGGER = get_logger(__name__)
DEFAULT_INIT_NUM_POINTS = -1
UPSTREAM_SCENE_SCALE_FACTOR = 1.1
KNN_SCALE_INIT_CHUNK_SIZE = 1024
SH_C0 = 0.28209479177387814
STANDARD_ALPHA_THRESHOLD = 1.0 / 255.0


@dataclass
class Frame:
    image: Tensor  # [H, W, 3], CPU float32 in [0, 1]
    viewmat: Tensor  # [4, 4], CPU float32 world-to-camera
    K: Tensor  # [3, 3], CPU float32
    name: str
    mask: Optional[Tensor] = None  # [H, W], CPU bool


@dataclass
class MultiviewData:
    train_frames: List[Frame]
    eval_frames: List[Frame]
    points: Optional[Tensor]  # [N, 3], CPU float32
    point_colors: Optional[Tensor]  # [N, 3], CPU float32 in [0, 1]
    scene_scale: float


@dataclass
class TrainConfig:
    data: Path
    result_dir: Path
    auto_output_root: Optional[Path] = None
    data_type: str = "auto"
    device: str = "supa:0"
    iterations: int = 100
    init_num_points: int = DEFAULT_INIT_NUM_POINTS
    init_opa: float = 0.1
    init_scale: float = 1.0
    data_factor: int = 4
    test_every: int = 8
    normalize_world_space: bool = True
    global_scale: float = 1.0
    lr_scale: float = 1.0
    save_every: int = 50
    seed: int = 42
    native: bool = False
    native_tiled: str = "false"
    native_tiled_metadata: bool = False
    native_tiled_train: bool = False
    native_tiled_backward_mode: str = "auto"
    native_alpha_threshold: Optional[float] = None
    distributed: bool = False
    distributed_world_size: int = 0
    distributed_master_addr: str = "127.0.0.1"
    distributed_master_port: int = 0
    save_ply: bool = True
    resume: Optional[Path] = None
    eval_max_frames: int = 8
    print_all_step_info: bool = False
    print_debug_info: bool = False
    train_crop_size: int = 0
    near_plane: float = 0.01
    far_plane: float = 1e10
    radius_clip: float = 0.0
    log_step_time: bool = False
    ssim_lambda: float = 0.2
    random_bkgd: bool = False
    shuffle_train: bool = True
    means_lr_final_factor: float = 0.01
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    refine_gaussians: bool = True
    refine_profile: str = "manual"
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000
    pause_refine_after_reset: int = 0
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    max_gaussians: int = 0
    max_gaussians_ramp_steps: int = 0
    refine_exposure: bool = False
    exposure_lr: float = 1e-3
    exposure_l2: float = 1e-4


SPLAT_KEYS = ("means", "scales", "quats", "opacities", "sh0")
OPTIONAL_SPLAT_KEYS = ("shN",)

TRAIN_PHASE_NAMES = (
    "schedule",
    "data_and_crop",
    "target_to_device",
    "rasterize",
    "postprocess",
    "loss",
    "zero_grad",
    "backward",
    "grad_sync",
    "refinement_state",
    "optimizer_step",
    "refinement",
    "synchronize",
    "metrics_cpu",
    "save_step",
)


@dataclass(frozen=True)
class CropWindow:
    x: int
    y: int
    width: int
    height: int


_AUTO_OUTPUT_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _add_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help: str,
) -> None:
    parser.add_argument(
        name,
        type=_parse_bool,
        default=default,
        metavar="true|false",
        help=f"{help} Defaults to {str(default).lower()}.",
    )


def _slug_part(value: object) -> str:
    text = _AUTO_OUTPUT_UNSAFE_CHARS.sub("-", str(value).strip())
    return text.strip("-._") or "run"


def _native_tiled_option(value: str) -> str:
    key = str(value).strip().lower()
    if key in {"0", "false"}:
        return "false"
    if key in {"1", "true"}:
        return "true"
    if key == "auto":
        return "auto"
    raise ValueError(
        "native_tiled must be one of false, true, or auto; "
        f"got {value!r}"
    )


def _native_tiled_backward_mode(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key in {"auto", "pixel", "tile_reduce", "parallel"}:
        return key
    raise ValueError(
        "native_tiled_backward_mode must be one of auto, pixel, "
        f"tile-reduce, or parallel; got {value!r}"
    )


def _supa_native_options(cfg: TrainConfig) -> dict:
    native_tiled = _native_tiled_option(cfg.native_tiled)
    backward_mode = _native_tiled_backward_mode(cfg.native_tiled_backward_mode)
    explicit_backward_mode = backward_mode != "auto"
    alpha_threshold = _resolve_native_alpha_threshold(
        cfg,
        native_tiled=native_tiled,
        explicit_backward_mode=explicit_backward_mode,
    )
    native_tiled_train = bool(
        cfg.native_tiled_train or cfg.native or alpha_threshold > 0.0 or explicit_backward_mode
    )
    if (bool(cfg.native) or alpha_threshold > 0.0 or explicit_backward_mode) and native_tiled == "false":
        native_tiled = "auto"
    native_tiled_metadata = bool(
        cfg.native_tiled_metadata
        or native_tiled_train
        or native_tiled != "false"
    )
    if (
        backward_mode == "auto"
        and alpha_threshold > 0.0
        and (native_tiled_train or native_tiled_metadata or native_tiled != "false")
    ):
        backward_mode = "parallel"
    return {
        "native": bool(
            cfg.native
            or native_tiled != "false"
            or native_tiled_metadata
            or native_tiled_train
            or alpha_threshold > 0.0
            or explicit_backward_mode
        ),
        "native_tiled": native_tiled,
        "native_tiled_metadata": native_tiled_metadata,
        "native_tiled_train": native_tiled_train,
        "native_tiled_backward_mode": backward_mode,
        "native_alpha_threshold": alpha_threshold,
    }


def _resolve_native_alpha_threshold(
    cfg: TrainConfig,
    *,
    native_tiled: Optional[str] = None,
    explicit_backward_mode: Optional[bool] = None,
) -> float:
    if cfg.native_alpha_threshold is not None:
        return float(cfg.native_alpha_threshold)
    tiled_mode = (
        _native_tiled_option(cfg.native_tiled)
        if native_tiled is None
        else native_tiled
    )
    explicit_mode = (
        _native_tiled_backward_mode(cfg.native_tiled_backward_mode) != "auto"
        if explicit_backward_mode is None
        else bool(explicit_backward_mode)
    )
    native_requested = bool(
        cfg.native
        or cfg.native_tiled_train
        or cfg.native_tiled_metadata
        or tiled_mode != "false"
        or explicit_mode
    )
    return STANDARD_ALPHA_THRESHOLD if native_requested else 0.0


def _format_config_value(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _device_detail_rows(cfg: TrainConfig) -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = [("requested", cfg.device)]
    try:
        device = resolve_device(cfg.device)
    except Exception as exc:  # pragma: no cover - startup diagnostics only
        rows.append(("status", f"unresolved: {exc}"))
        return rows

    rows.extend(
        [
            ("resolved", str(device)),
            ("type", device.type),
            ("torch", torch.__version__),
        ]
    )
    if device.type == "supa":
        try:
            import torch_br  # pylint: disable=import-outside-toplevel

            supa = ensure_supa_runtime()
            index = 0 if device.index is None else int(device.index)
            props = supa.get_device_properties(index)
            rows.extend(
                [
                    ("backend", "Biren SUPA / BRPyTorch"),
                    ("torch_br", getattr(torch_br, "__version__", "unknown")),
                    ("device_count", supa.device_count()),
                    ("selected_index", index),
                    ("selected_name", getattr(props, "name", props)),
                ]
            )
            total_memory = getattr(props, "total_memory", None)
            if total_memory is not None:
                rows.append(("total_memory_bytes", int(total_memory)))
        except Exception as exc:  # pragma: no cover - startup diagnostics only
            rows.append(("status", f"SUPA detail probe failed: {exc}"))
    else:
        rows.append(("backend", "CPU"))
    return rows


def _training_parameter_rows(cfg: TrainConfig) -> list[tuple[str, object]]:
    rows = [
        (field.name, _format_config_value(getattr(cfg, field.name)))
        for field in fields(TrainConfig)
    ]
    rows.append(("supa_native_options", json.dumps(_supa_native_options(cfg), sort_keys=True)))
    return rows


def _log_cli_startup(cfg: TrainConfig) -> None:
    log_info(LOGGER, gsplat_biren_ascii_art())
    log_info(
        LOGGER,
        format_key_value_table("Device Details", _device_detail_rows(cfg)),
    )
    log_info(
        LOGGER,
        format_key_value_table("Training Parameters", _training_parameter_rows(cfg)),
    )


def _debug_tensor_summary(value: Tensor) -> dict:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
    }


def _debug_value(value):
    if isinstance(value, Tensor):
        return _debug_tensor_summary(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, CropWindow):
        return _crop_metadata(value)
    if isinstance(value, dict):
        return {str(key): _debug_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_debug_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class _DebugClock:
    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.previous = self.start

    def mark(self) -> Tuple[float, float]:
        now = time.perf_counter()
        elapsed = now - self.start
        delta = now - self.previous
        self.previous = now
        return elapsed, delta


_DEBUG_CLOCK: Optional[_DebugClock] = None


def _reset_debug_clock() -> None:
    global _DEBUG_CLOCK  # pylint: disable=global-statement
    _DEBUG_CLOCK = _DebugClock()


def _debug_log(
    enabled: bool,
    event: str,
    *,
    prefix: str = "",
    **fields,
) -> None:
    if not enabled:
        return
    global _DEBUG_CLOCK  # pylint: disable=global-statement
    if _DEBUG_CLOCK is None:
        _DEBUG_CLOCK = _DebugClock()
    elapsed_s, delta_s = _DEBUG_CLOCK.mark()
    payload = " ".join(
        f"{key}={json.dumps(_debug_value(value), sort_keys=True)}"
        for key, value in fields.items()
    )
    wall = datetime.now(timezone.utc).astimezone().isoformat(timespec="microseconds")
    message = (
        f"[debug] wall={wall} elapsed_s={elapsed_s:.6f} "
        f"delta_s={delta_s:.6f} {prefix}{event}"
    )
    if payload:
        message = f"{message} {payload}"
    log_info(LOGGER, message)


def _debug_frame_summary(frame: Frame) -> dict:
    return {
        "name": frame.name,
        "image": _debug_tensor_summary(frame.image),
        "viewmat": _debug_tensor_summary(frame.viewmat),
        "K": _debug_tensor_summary(frame.K),
        "mask": None if frame.mask is None else _debug_tensor_summary(frame.mask),
    }


def _debug_data_summary(data: MultiviewData) -> dict:
    return {
        "train_frames": len(data.train_frames),
        "eval_frames": len(data.eval_frames),
        "points": None if data.points is None else _debug_tensor_summary(data.points),
        "point_colors": (
            None
            if data.point_colors is None
            else _debug_tensor_summary(data.point_colors)
        ),
        "scene_scale": float(data.scene_scale),
    }


def _debug_params_summary(params: torch.nn.ParameterDict) -> dict:
    return {name: _debug_tensor_summary(param) for name, param in params.items()}


def _debug_optimizer_summary(optimizer) -> dict:
    if isinstance(optimizer, dict):
        return {
            name: _debug_optimizer_summary(item)
            for name, item in optimizer.items()
        }
    return {
        "groups": [
            {
                "name": group.get("name"),
                "lr": float(group.get("lr", 0.0)),
                "params": len(group.get("params", [])),
                "foreach": group.get("foreach"),
            }
            for group in optimizer.param_groups
        ]
    }


def _debug_refinement_state_summary(state: dict) -> dict:
    return {
        name: _debug_tensor_summary(value)
        for name, value in state.items()
        if isinstance(value, Tensor)
    }


def _debug_meta_summary(meta: dict) -> dict:
    keys = (
        "backend",
        "projection_backend",
        "covariance_backend",
        "composite_backend",
        "composite_variant",
        "sh_backend",
        "extra_signals_sh_backend",
        "visible_gaussians",
        "culled_gaussians",
        "compact_backend",
        "tile_count_backend",
        "tiled_metadata_backend",
        "tiles_per_gauss",
        "distributed_backend",
        "world_rank",
        "world_size",
        "train_crop",
        "exposure_frame_id",
    )
    summary = {
        key: _debug_value(meta[key])
        for key in keys
        if key in meta
    }
    summary["keys"] = sorted(str(key) for key in meta.keys())
    return summary


def _debug_grad_summary(params: torch.nn.ParameterDict) -> dict:
    return {
        name: None if param.grad is None else _debug_tensor_summary(param.grad)
        for name, param in params.items()
    }


def _auto_result_dir(root: Path, args: argparse.Namespace) -> Path:
    data_path = Path(args.data)
    dataset_name = data_path.stem if data_path.suffix else data_path.name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [
        _slug_part(dataset_name),
        f"it{int(args.iterations)}",
        f"pts{int(args.init_num_points)}",
        f"sh{int(args.sh_degree)}",
        f"seed{int(args.seed)}",
        _slug_part(str(args.device).replace(":", "")),
    ]
    if bool(args.native):
        parts.append("native")
    native_tiled = _native_tiled_option(args.native_tiled)
    alpha_threshold = 0.0 if args.native_alpha_threshold is None else float(
        args.native_alpha_threshold
    )
    if (
        (bool(args.native) or alpha_threshold > 0.0)
        and native_tiled == "false"
    ):
        native_tiled = "auto"
    if native_tiled != "false":
        parts.append(f"tiled-{_slug_part(native_tiled)}")
    if int(args.train_crop_size) > 0:
        parts.append(f"crop{int(args.train_crop_size)}")
    if not bool(args.refine_gaussians):
        parts.append("refine-false")
    if bool(args.random_bkgd):
        parts.append("randbkgd")
    parts.append(timestamp)
    return root / "_".join(parts)


def _as_float_image(array: np.ndarray) -> Tensor:
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"images must have shape [M, H, W, 3] or [H, W, 3], got {array.shape}")
    image = torch.from_numpy(array[..., :3]).float()
    if image.max().item() > 1.5:
        image = image / 255.0
    return image.clamp(0.0, 1.0).contiguous()


def _as_float_colors(array: np.ndarray) -> Tensor:
    colors = torch.from_numpy(array[..., :3]).float()
    if colors.numel() > 0 and colors.max().item() > 1.5:
        colors = colors / 255.0
    return colors.clamp(0.0, 1.0).contiguous()


def _rgb_to_sh(rgb: Tensor) -> Tensor:
    return (rgb - 0.5) / SH_C0


def _sh_to_rgb(sh: Tensor) -> Tensor:
    return sh * SH_C0 + 0.5


def _sh_coeff_count(sh_degree: int) -> int:
    return max((int(sh_degree) + 1) ** 2 - 1, 0)


def _sh0_param_to_3d(sh0: Tensor) -> Tensor:
    if sh0.ndim == 3 and sh0.shape[1] == 1 and sh0.shape[-1] == 3:
        return sh0
    if sh0.ndim == 2 and sh0.shape[-1] == 3:
        return sh0.unsqueeze(1)
    raise ValueError(f"sh0 must have shape (N, 1, 3) or (N, 3), got {tuple(sh0.shape)}")


def _sh_param_to_3d(shN: Tensor) -> Tensor:
    if shN.ndim == 3 and shN.shape[-1] == 3:
        return shN
    if shN.ndim == 2 and shN.shape[1] % 3 == 0:
        return shN.reshape(shN.shape[0], shN.shape[1] // 3, 3)
    raise ValueError(f"shN must have shape (N, K, 3) or (N, K * 3), got {tuple(shN.shape)}")


def _sh_splat_to_param(shN: Tensor, sh_coeffs: int) -> Tensor:
    shN = _sh_param_to_3d(shN.detach().float()).contiguous()
    if sh_coeffs > shN.shape[1]:
        pad = torch.zeros(
            shN.shape[0],
            sh_coeffs - shN.shape[1],
            3,
            dtype=shN.dtype,
            device=shN.device,
        )
        shN = torch.cat((shN, pad), dim=1).contiguous()
    return shN[:, :sh_coeffs].contiguous()


def _params_splats_for_checkpoint(params: torch.nn.ParameterDict) -> Dict[str, Tensor]:
    splats = {name: value.detach().cpu() for name, value in params.items()}
    if "sh0" in splats:
        splats["sh0"] = _sh0_param_to_3d(splats["sh0"]).contiguous()
    if "shN" in splats:
        splats["shN"] = _sh_param_to_3d(splats["shN"]).contiguous()
    return splats


def _checkpoint_values_from_splats(
    splats: Dict[str, Tensor],
    sh_degree: int,
) -> Dict[str, Tensor]:
    missing = [name for name in ("means", "quats", "opacities") if name not in splats]
    if "scales" not in splats and "log_scales" not in splats:
        missing.append("scales")
    if "sh0" not in splats and "colors" not in splats:
        missing.append("sh0")
    if missing:
        raise ValueError(f"checkpoint splats are missing required keys: {missing}")

    means = splats["means"].detach().float().contiguous()
    n_gaussians = int(means.shape[0])
    scales_source = splats["scales"] if "scales" in splats else splats["log_scales"]
    if "sh0" in splats:
        sh0 = _sh0_param_to_3d(splats["sh0"].detach().float()).contiguous()
    else:
        # Older SUPA checkpoints stored trainable RGB logits under "colors".
        # Convert them into the upstream SH DC parameter for resume compatibility.
        rgb = torch.sigmoid(splats["colors"].detach().float()).clamp(0.0, 1.0)
        sh0 = _rgb_to_sh(rgb).unsqueeze(1).contiguous()

    values = {
        "means": means,
        "scales": scales_source.detach().float().contiguous(),
        "quats": splats["quats"].detach().float().contiguous(),
        "opacities": splats["opacities"].detach().float().contiguous(),
        "sh0": sh0,
    }
    expected_shapes = {
        "means": (n_gaussians, 3),
        "scales": (n_gaussians, 3),
        "quats": (n_gaussians, 4),
        "opacities": (n_gaussians,),
        "sh0": (n_gaussians, 1, 3),
    }
    for name, shape in expected_shapes.items():
        if tuple(values[name].shape) != shape:
            raise ValueError(
                f"checkpoint splat {name!r} has shape {tuple(values[name].shape)}, "
                f"expected {shape}"
            )

    sh_coeffs = _sh_coeff_count(sh_degree)
    if "shN" in splats:
        shN = _sh_splat_to_param(splats["shN"], sh_coeffs)
        if shN.shape[0] != n_gaussians:
            raise ValueError(
                f"checkpoint splat 'shN' has shape {tuple(shN.shape)}, "
                f"expected ({n_gaussians}, K, 3)"
            )
        values["shN"] = shN
    elif sh_coeffs > 0:
        values["shN"] = torch.zeros(n_gaussians, sh_coeffs, 3)
    return values


def _as_bool_masks(array: np.ndarray, num_images: int, height: int, width: int) -> List[Tensor]:
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.shape != (num_images, height, width):
        raise ValueError(
            f"masks must have shape [M, H, W] or [M, H, W, 1], got {array.shape}"
        )

    masks = []
    for index in range(num_images):
        mask_np = array[index]
        if np.issubdtype(mask_np.dtype, np.floating):
            mask = torch.from_numpy(mask_np > 0.5)
        else:
            mask = torch.from_numpy(mask_np != 0)
        mask = mask.bool().contiguous()
        if not bool(mask.any().item()):
            raise ValueError(f"mask for frame {index} has no valid pixels")
        masks.append(mask)
    return masks


def _split_indices(num_frames: int, test_every: int) -> Tuple[List[int], List[int]]:
    if num_frames <= 0:
        raise ValueError("dataset must contain at least one frame")
    if test_every <= 0 or num_frames == 1:
        return list(range(num_frames)), [0]
    train = [i for i in range(num_frames) if i % test_every != 0]
    eval_ = [i for i in range(num_frames) if i % test_every == 0]
    if not train:
        train = list(range(num_frames))
    if not eval_:
        eval_ = [train[0]]
    return train, eval_


def _npz_indices_or_default(
    data: np.lib.npyio.NpzFile,
    key: str,
    default: List[int],
    num_frames: int,
) -> List[int]:
    if key not in data:
        return default

    array = np.asarray(data[key])
    if array.ndim != 1:
        raise ValueError(f"{key} must be a 1D integer array, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{key} must be an integer array, got dtype {array.dtype}")
    if array.size == 0:
        raise ValueError(f"{key} must contain at least one frame index")

    indices = array.astype(np.int64, copy=False)
    invalid = indices[(indices < 0) | (indices >= num_frames)]
    if invalid.size > 0:
        bad = int(invalid[0])
        raise ValueError(
            f"{key} contains out-of-range frame index {bad}; valid range is [0, {num_frames})"
        )
    return indices.tolist()


def _load_npz_data(path: Path, test_every: int) -> MultiviewData:
    data = np.load(path)
    images_np = data["images"]
    if images_np.ndim != 4:
        raise ValueError(f"NPZ images must have shape [M, H, W, 3], got {images_np.shape}")
    images = [_as_float_image(images_np[i]) for i in range(images_np.shape[0])]

    if "viewmats" in data:
        viewmats = torch.from_numpy(data["viewmats"]).float()
    elif "camtoworlds" in data:
        camtoworlds = torch.from_numpy(data["camtoworlds"]).float()
        viewmats = torch.linalg.inv(camtoworlds)
    else:
        raise ValueError("NPZ data must contain either viewmats or camtoworlds")
    Ks = torch.from_numpy(data["Ks"]).float()
    if viewmats.shape != (len(images), 4, 4):
        raise ValueError(f"viewmats must have shape [M, 4, 4], got {tuple(viewmats.shape)}")
    if Ks.shape != (len(images), 3, 3):
        raise ValueError(f"Ks must have shape [M, 3, 3], got {tuple(Ks.shape)}")

    masks = None
    for key in ("masks", "mask"):
        if key in data:
            masks = _as_bool_masks(
                data[key],
                len(images),
                images[0].shape[0],
                images[0].shape[1],
            )
            break

    all_train, all_eval = _split_indices(len(images), test_every)
    train_indices = _npz_indices_or_default(data, "train_indices", all_train, len(images))
    eval_indices = _npz_indices_or_default(data, "eval_indices", all_eval, len(images))

    frames = [
        Frame(
            image=images[i],
            viewmat=viewmats[i].contiguous(),
            K=Ks[i].contiguous(),
            name=f"frame_{i:04d}",
            mask=masks[i] if masks is not None else None,
        )
        for i in range(len(images))
    ]

    points = None
    for key in ("points", "means", "means3d"):
        if key in data:
            points = torch.from_numpy(data[key]).float().contiguous()
            break

    point_colors = None
    for key in ("point_colors", "points_rgb", "colors"):
        if key in data:
            point_colors = _as_float_colors(data[key])
            break

    scene_scale = 1.0
    if "scene_scale" in data:
        scene_scale = float(np.asarray(data["scene_scale"]).item())
    elif points is not None and points.numel() > 0:
        scene_scale = float(points.std(dim=0).norm().clamp(min=1e-3).item())

    return MultiviewData(
        train_frames=[frames[i] for i in train_indices],
        eval_frames=[frames[i] for i in eval_indices],
        points=points,
        point_colors=point_colors,
        scene_scale=max(scene_scale, 1e-3),
    )


def _load_colmap_data(
    path: Path,
    factor: int,
    test_every: int,
    normalize_world_space: bool,
) -> MultiviewData:
    from applications.datasets.colmap import (  # pylint: disable=import-outside-toplevel
        Dataset,
        Parser,
    )

    parser = Parser(
        data_dir=str(path),
        factor=factor,
        normalize=normalize_world_space,
        test_every=test_every,
        load_exposure=False,
    )

    def load_split(split: str) -> List[Frame]:
        dataset = Dataset(parser, split=split)
        frames = []
        for i in range(len(dataset)):
            item = dataset[i]
            image = (item["image"].float() / 255.0).clamp(0.0, 1.0).contiguous()
            viewmat = torch.linalg.inv(item["camtoworld"].float()).contiguous()
            name = f"{split}_{i:04d}"
            mask = item.get("mask")
            frames.append(
                Frame(
                    image=image,
                    viewmat=viewmat,
                    K=item["K"].float().contiguous(),
                    name=name,
                    mask=mask.bool().contiguous() if mask is not None else None,
                )
            )
        return frames

    train_frames = load_split("train")
    eval_frames = load_split("val") or train_frames[:1]
    return MultiviewData(
        train_frames=train_frames,
        eval_frames=eval_frames,
        points=torch.from_numpy(parser.points).float().contiguous(),
        point_colors=torch.from_numpy(parser.points_rgb / 255.0).float().contiguous(),
        scene_scale=max(float(parser.scene_scale), 1e-3),
    )


def _apply_scene_scale_config(data: MultiviewData, cfg: TrainConfig) -> MultiviewData:
    # Upstream gsplat's trainer expands the parser scene scale by 1.1 and then
    # applies the user global scale. The value feeds the means LR and
    # densification thresholds, so keep it centralized.
    scene_scale = (
        max(float(data.scene_scale), 1e-3)
        * UPSTREAM_SCENE_SCALE_FACTOR
        * float(cfg.global_scale)
    )
    return replace(data, scene_scale=max(scene_scale, 1e-3))


def load_multiview_data(cfg: TrainConfig) -> MultiviewData:
    data_type = cfg.data_type
    if data_type == "auto":
        data_type = "npz" if cfg.data.suffix.lower() == ".npz" else "colmap"
    if data_type == "npz":
        return _apply_scene_scale_config(_load_npz_data(cfg.data, cfg.test_every), cfg)
    if data_type == "colmap":
        return _apply_scene_scale_config(
            _load_colmap_data(
                cfg.data,
                factor=cfg.data_factor,
                test_every=cfg.test_every,
                normalize_world_space=cfg.normalize_world_space,
            ),
            cfg,
        )
    raise ValueError(f"Unsupported data_type={cfg.data_type!r}; expected auto, npz, or colmap")


def _camera_random_points(frames: List[Frame], num_points: int, extent: float) -> Tensor:
    camtoworlds = torch.linalg.inv(torch.stack([frame.viewmat for frame in frames], dim=0))
    centers = camtoworlds[:, :3, 3]
    center = centers.mean(dim=0)
    scale = centers.sub(center).norm(dim=-1).max().clamp(min=1.0) * extent
    points = center + (torch.rand(num_points, 3) * 2.0 - 1.0) * scale
    points[:, 2] = points[:, 2].abs() + scale.clamp(min=1.0)
    return points.contiguous()


def _jitter_existing_points(
    points: Tensor,
    colors: Tensor,
    target: int,
    scene_scale: float,
    seed: int,
) -> Tuple[Tensor, Tensor]:
    if points.shape[0] >= target:
        return points, colors
    if points.shape[0] == 0:
        raise ValueError("cannot upsample an empty input point cloud")

    generator = torch.Generator(device="cpu").manual_seed(seed + 7919)
    missing = target - points.shape[0]
    base_ids = torch.randint(points.shape[0], (missing,), generator=generator)
    base_points = points[base_ids]
    base_colors = colors[base_ids]
    point_extent = float(points.float().std(dim=0).norm().clamp(min=1e-6).item())
    jitter_std = max(min(point_extent, scene_scale) * 0.0025, scene_scale * 1e-4, 1e-5)
    jitter = torch.randn(missing, 3, generator=generator, dtype=points.dtype) * jitter_std
    return (
        torch.cat((points, base_points + jitter), dim=0).contiguous(),
        torch.cat((colors, base_colors), dim=0).contiguous(),
    )


def _select_initial_points(data: MultiviewData, cfg: TrainConfig) -> Tuple[Tensor, Tensor]:
    if data.points is None or data.points.numel() == 0:
        if cfg.init_num_points < 0:
            raise ValueError(
                "init_num_points < 0 requires an input point cloud; "
                "set --init-num-points to a positive value for random initialization"
            )
        points = _camera_random_points(
            data.train_frames,
            cfg.init_num_points,
            extent=1.5,
        )
        colors = torch.rand(points.shape[0], 3)
        return points, colors

    points = data.points.float()
    colors = data.point_colors
    if colors is None or colors.shape[0] != points.shape[0]:
        colors = torch.full((points.shape[0], 3), 0.5)
    colors = colors.float().clamp(0.0, 1.0)

    target = (
        min(cfg.init_num_points, points.shape[0])
        if cfg.init_num_points > 0
        else points.shape[0]
    )
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    if points.shape[0] > target:
        ids = torch.randperm(points.shape[0], generator=generator)[:target]
        points = points[ids]
        colors = colors[ids]
    elif cfg.init_num_points > points.shape[0]:
        points, colors = _jitter_existing_points(
            points,
            colors,
            cfg.init_num_points,
            data.scene_scale,
            cfg.seed,
        )
    return points.contiguous(), colors.contiguous()


def _resolve_init_num_points(
    data: MultiviewData,
    cfg: TrainConfig,
    *,
    required: bool = True,
) -> TrainConfig:
    if int(cfg.init_num_points) >= 0:
        return cfg
    if data.points is not None and data.points.numel() > 0:
        return replace(cfg, init_num_points=int(data.points.shape[0]))
    if required:
        raise ValueError(
            "init_num_points < 0 requires an input point cloud; "
            "COLMAP datasets need points3D, and NPZ datasets need points/means/means3d"
        )
    return cfg


def _knn_rms_distances(
    points: Tensor,
    *,
    chunk_size: int = KNN_SCALE_INIT_CHUNK_SIZE,
) -> Tensor:
    n_points = int(points.shape[0])
    if n_points <= 1:
        return points.new_empty((n_points,))

    points_f = points.float()
    chunk = max(1, min(int(chunk_size), n_points))
    neighbor_count = min(4, n_points)
    distances = torch.empty(n_points, dtype=points_f.dtype, device=points_f.device)
    for start in range(0, n_points, chunk):
        end = min(start + chunk, n_points)
        block_dist = torch.cdist(points_f[start:end], points_f)
        nearest = torch.topk(block_dist, k=neighbor_count, largest=False).values[:, 1:]
        distances[start:end] = nearest.square().mean(dim=-1).sqrt()
    return distances


def _estimate_scales(points: Tensor, scene_scale: float, init_scale: float = 1.0) -> Tensor:
    n_points = points.shape[0]
    if n_points <= 1:
        base = torch.full((n_points,), max(scene_scale * 0.01, 1e-3))
        return base.log().unsqueeze(-1).repeat(1, 3)

    # Match upstream gsplat: initialize each Gaussian from the RMS distance to
    # its 3 nearest neighbors in the full input point cloud, excluding self.
    dist_avg = _knn_rms_distances(points).clamp_min(1e-7)
    return (dist_avg * init_scale).log().unsqueeze(-1).repeat(1, 3).contiguous()


def create_model(data: MultiviewData, cfg: TrainConfig, device: torch.device) -> torch.nn.ParameterDict:
    points, colors = _select_initial_points(data, cfg)
    scales = _estimate_scales(points, data.scene_scale, init_scale=float(cfg.init_scale))
    quats = torch.rand(points.shape[0], 4)
    opacities = torch.logit(torch.full((points.shape[0],), float(cfg.init_opa)))
    sh0 = _rgb_to_sh(colors.clamp(0.0, 1.0)).unsqueeze(1).contiguous()

    values = {
        "means": torch.nn.Parameter(points),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
    }
    sh_coeffs = _sh_coeff_count(cfg.sh_degree)
    if sh_coeffs > 0:
        values["shN"] = torch.nn.Parameter(torch.zeros(points.shape[0], sh_coeffs, 3))

    params = torch.nn.ParameterDict(values)
    return params.to(device)


def create_exposure_params(data: MultiviewData, device: torch.device) -> torch.nn.ParameterDict:
    frame_count = len(data.train_frames) + len(data.eval_frames)
    values = {
        "log_scale": torch.nn.Parameter(torch.zeros(frame_count, 3)),
        "bias": torch.nn.Parameter(torch.zeros(frame_count, 3)),
    }
    return torch.nn.ParameterDict(values).to(device)


def _exposure_frame_id(frame_index: int, *, is_eval: bool, data: MultiviewData) -> int:
    if is_eval:
        return len(data.train_frames) + int(frame_index)
    return int(frame_index)


def _apply_exposure(
    render: Tensor,
    exposure_params: Optional[torch.nn.ParameterDict],
    frame_id: int,
    *,
    clamp: bool = True,
) -> Tensor:
    if exposure_params is None:
        return render
    log_scale = exposure_params["log_scale"][frame_id].to(render.device)
    bias = exposure_params["bias"][frame_id].to(render.device)
    corrected = render * torch.exp(log_scale) + bias
    return corrected.clamp(0.0, 1.0) if clamp else corrected


def _exposure_regularization(
    exposure_params: Optional[torch.nn.ParameterDict],
    frame_id: int,
    weight: float,
) -> Optional[Tensor]:
    if exposure_params is None or weight <= 0.0:
        return None
    log_scale = exposure_params["log_scale"][frame_id]
    bias = exposure_params["bias"][frame_id]
    return (log_scale.square().mean() + bias.square().mean()) * float(weight)


def _exposure_state_for_checkpoint(
    exposure_params: Optional[torch.nn.ParameterDict],
) -> Optional[Dict[str, Tensor]]:
    if exposure_params is None:
        return None
    return {name: value.detach().cpu() for name, value in exposure_params.items()}


def _model_exposure_from_checkpoint(
    ckpt: dict,
    data: MultiviewData,
    device: torch.device,
    enabled: bool,
) -> Optional[torch.nn.ParameterDict]:
    if not enabled:
        return None
    exposure_params = create_exposure_params(data, device)
    exposure_state = ckpt.get("exposure")
    if not isinstance(exposure_state, dict):
        return exposure_params
    for name, param in exposure_params.items():
        if name not in exposure_state:
            continue
        value = exposure_state[name].detach().float()
        if tuple(value.shape) == tuple(param.shape):
            param.data.copy_(value.to(device))
        else:
            log_warning(
                LOGGER,
                f"skipping incompatible exposure state {name}: "
                f"checkpoint shape={tuple(value.shape)} expected={tuple(param.shape)}",
            )
    return exposure_params


def _model_from_splats(
    splats: Dict[str, Tensor],
    device: torch.device,
    sh_degree: int = 0,
) -> torch.nn.ParameterDict:
    values = {
        name: torch.nn.Parameter(value)
        for name, value in _checkpoint_values_from_splats(splats, sh_degree).items()
    }
    params = torch.nn.ParameterDict(values)
    return params.to(device)


class _SupaSafeAdam(torch.optim.Optimizer):
    """Small Adam implementation that avoids BRPyTorch's unsupported lerp path."""

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight_decay value: {weight_decay}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": False,
            "maximize": maximize,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "decoupled_weight_decay": False,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            maximize = bool(group.get("maximize", False))
            if group.get("amsgrad", False):
                raise RuntimeError("_SupaSafeAdam does not support amsgrad")
            if group.get("decoupled_weight_decay", False):
                raise RuntimeError("_SupaSafeAdam does not support decoupled weight decay")

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("_SupaSafeAdam does not support sparse gradients")
                grad = _sanitize_adam_update_tensor(grad)
                if maximize:
                    grad = -grad
                if weight_decay != 0.0:
                    grad = grad.add(param, alpha=weight_decay)
                    grad = _sanitize_adam_update_tensor(grad)

                state = self.state[param]
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(
                        param,
                        memory_format=torch.preserve_format,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        param,
                        memory_format=torch.preserve_format,
                    )

                step_state = state["step"]
                if isinstance(step_state, Tensor):
                    step_state.add_(1)
                    step = float(step_state.detach().cpu().item())
                else:
                    step = float(step_state) + 1.0
                    state["step"] = step

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                _sanitize_adam_state_(exp_avg)
                _sanitize_adam_state_(exp_avg_sq, nonnegative=True)
                exp_avg.mul_(beta1)
                exp_avg.add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2)
                exp_avg_sq.addcmul_(grad, grad, value=1.0 - beta2)
                _sanitize_adam_state_(exp_avg)
                _sanitize_adam_state_(exp_avg_sq, nonnegative=True)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                denom = exp_avg_sq.sqrt().add_(eps)
                update = _sanitize_adam_update_tensor(exp_avg / denom)
                param.add_(update, alpha=-step_size)

        return loss


def _adam_square_safe_bound(tensor: Tensor) -> Optional[float]:
    if not tensor.is_floating_point():
        return None
    finfo = torch.finfo(tensor.dtype)
    # Keep enough headroom for grad * grad and Adam's accumulation on float32
    # tensors. This guard is intentionally far above normal 3DGS gradients; it
    # only catches pathological native-backward spikes before they poison Adam
    # state with Inf and later produce NaN parameters.
    return math.sqrt(float(finfo.max)) * 0.25


def _sanitize_adam_update_tensor(tensor: Tensor) -> Tensor:
    bound = _adam_square_safe_bound(tensor)
    if bound is None:
        return tensor
    finite = torch.isfinite(tensor)
    zero = torch.zeros((), dtype=tensor.dtype, device=tensor.device)
    return torch.where(finite, tensor, zero).clamp(min=-bound, max=bound)


def _sanitize_adam_state_(tensor: Tensor, *, nonnegative: bool = False) -> None:
    bound = _adam_square_safe_bound(tensor)
    if bound is None:
        return
    finite = torch.isfinite(tensor)
    zero = torch.zeros((), dtype=tensor.dtype, device=tensor.device)
    tensor.copy_(torch.where(finite, tensor, zero))
    if nonnegative:
        tensor.clamp_(min=0.0, max=torch.finfo(tensor.dtype).max * 0.25)
    else:
        tensor.clamp_(min=-bound, max=bound)


def _finite_or_zero(tensor: Tensor) -> Tensor:
    if not tensor.is_floating_point():
        return tensor
    finite = torch.isfinite(tensor)
    zero = torch.zeros((), dtype=tensor.dtype, device=tensor.device)
    return torch.where(finite, tensor, zero)


def _create_adam(param_groups, *, eps: float = 1e-15) -> _SupaSafeAdam:
    return _SupaSafeAdam(param_groups, eps=eps)


def create_optimizers(
    params: torch.nn.ParameterDict,
    scene_scale: float,
    cfg: TrainConfig,
    exposure_params: Optional[torch.nn.ParameterDict] = None,
) -> Dict[str, torch.optim.Optimizer]:
    """Create one optimizer per parameter, matching upstream trainer structure."""
    lrs = _base_optimizer_lrs(scene_scale, cfg)
    optimizers: Dict[str, torch.optim.Optimizer] = {}
    for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        if name not in params:
            continue
        optimizers[name] = _create_adam(
            [{"params": [params[name]], "lr": lrs[name], "name": name}],
        )
    if exposure_params is not None:
        optimizers["exposure"] = _create_adam(
            [
                {
                    "params": list(exposure_params.parameters()),
                    "lr": float(cfg.exposure_lr) * float(cfg.lr_scale),
                    "name": "exposure",
                }
            ],
        )
    return optimizers


def _base_optimizer_lrs(scene_scale: float, cfg: TrainConfig) -> Dict[str, float]:
    lr_scale = float(cfg.lr_scale)
    return {
        "means": float(cfg.means_lr) * scene_scale * lr_scale,
        "scales": float(cfg.scales_lr) * lr_scale,
        "quats": float(cfg.quats_lr) * lr_scale,
        "opacities": float(cfg.opacities_lr) * lr_scale,
        "sh0": float(cfg.sh0_lr) * lr_scale,
        "shN": float(cfg.shN_lr) * lr_scale,
        "exposure": float(cfg.exposure_lr) * lr_scale,
    }


def _scheduled_means_lr(
    base_lr: float,
    final_factor: float,
    step: int,
    total_steps: int,
) -> float:
    if total_steps <= 1:
        return float(base_lr)
    # Upstream applies ExponentialLR(gamma=final_factor ** (1 / max_steps))
    # after each optimizer step. This helper returns the LR used before that
    # step, so the last training step uses exponent (max_steps - 1) / max_steps.
    progress = min(max(step - 1, 0), total_steps) / float(total_steps)
    return float(base_lr * (final_factor**progress))


def _set_optimizer_lrs(
    optimizer,
    scene_scale: float,
    cfg: TrainConfig,
    step: int,
) -> Dict[str, float]:
    base_lrs = _base_optimizer_lrs(scene_scale, cfg)
    ordered_names = (
        "means",
        "scales",
        "quats",
        "opacities",
        "sh0",
        "shN",
        "exposure",
    )
    current_lrs = {}
    group_items = []
    for name in ordered_names:
        opt = optimizer.get(name)
        if opt is None:
            continue
        for group in opt.param_groups:
            group_items.append((name, group))
    for name, group in group_items:
        base_lr = base_lrs[name]
        if name == "means":
            lr = _scheduled_means_lr(
                base_lr,
                float(cfg.means_lr_final_factor),
                step,
                cfg.iterations,
            )
        else:
            lr = base_lr
        group["lr"] = lr
        current_lrs[name] = lr
    return current_lrs


def _optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for item in optimizer.values():
        for state in item.state.values():
            for key, value in list(state.items()):
                if isinstance(value, Tensor):
                    state[key] = value.to(device)


def _optimizer_zero_grad(optimizer) -> None:
    for item in optimizer.values():
        item.zero_grad(set_to_none=True)


def _optimizer_step(optimizer) -> None:
    for item in optimizer.values():
        item.step()


def _optimizer_state_dict(optimizer) -> dict:
    return {
        "format": "optimizer_dict",
        "optimizers": {
            name: _state_to_cpu(item.state_dict())
            for name, item in optimizer.items()
        },
    }


def _try_load_optimizer_state(
    optimizer,
    state_dict: dict,
    device: torch.device,
) -> bool:
    if not isinstance(state_dict, dict) or state_dict.get("format") != "optimizer_dict":
        log_warning(
            LOGGER,
            "skipping incompatible optimizer state during resume: "
            "checkpoint uses legacy single-optimizer state",
        )
        return False
    saved = state_dict.get("optimizers", {})
    loaded_any = False
    for name, item in optimizer.items():
        if name not in saved:
            continue
        try:
            item.load_state_dict(saved[name])
        except ValueError as exc:
            log_warning(
                LOGGER,
                f"skipping incompatible optimizer state for {name} during resume: {exc}",
            )
            continue
        loaded_any = True
    _optimizer_state_to_device(optimizer, device)
    return loaded_any


def _optimizer_group_for_name(
    optimizer,
    name: str,
) -> Optional[dict]:
    item = optimizer.get(name)
    if item is None:
        return None
    for group in item.param_groups:
        if group.get("name") == name:
            return group
    return item.param_groups[0] if item.param_groups else None


def _clone_optimizer_state_value(value):
    if isinstance(value, Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


@torch.no_grad()
def _replace_param_with_optimizer_state(
    params: torch.nn.ParameterDict,
    optimizer,
    name: str,
    new_value: Tensor,
    resize_state,
) -> None:
    old_param = params[name]
    old_size = int(old_param.shape[0]) if old_param.ndim > 0 else 0
    new_param = torch.nn.Parameter(
        new_value.detach().contiguous(),
        requires_grad=old_param.requires_grad,
    )
    group = _optimizer_group_for_name(optimizer, name)
    state_owner = optimizer.get(name)
    if group is not None:
        old_state = state_owner.state.pop(old_param, {}) if state_owner is not None else {}
        if old_state:
            new_state = {}
            for key, value in old_state.items():
                if (
                    isinstance(value, Tensor)
                    and value.ndim > 0
                    and int(value.shape[0]) == old_size
                ):
                    new_state[key] = resize_state(value)
                else:
                    new_state[key] = _clone_optimizer_state_value(value)
            state_owner.state[new_param] = new_state
        group["params"] = [new_param]
    params[name] = new_param


def _zero_optimizer_state_for_param(
    optimizer,
    param: torch.nn.Parameter,
) -> None:
    for item in optimizer.values():
        state = item.state.get(param, {})
        for key, value in list(state.items()):
            if isinstance(value, Tensor) and key != "step":
                state[key] = torch.zeros_like(value)


def _all_reduce_exposure_grads(
    exposure_params: Optional[torch.nn.ParameterDict],
    world_size: int,
) -> None:
    if exposure_params is None or world_size <= 1:
        return
    for param in exposure_params.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad)
        param.grad.div_(float(world_size))


def _reset_refinement_state(state: dict, params: torch.nn.ParameterDict) -> None:
    n_gaussians = int(params["means"].shape[0])
    device = params["means"].device
    state["grad2d"] = torch.zeros(n_gaussians, device=device)
    state["count"] = torch.zeros(n_gaussians, device=device)
    state["radii"] = torch.zeros(n_gaussians, device=device)


def _new_refinement_state(params: torch.nn.ParameterDict) -> dict:
    state: dict = {}
    _reset_refinement_state(state, params)
    return state


def _retain_refinement_grad(meta: dict, cfg: TrainConfig, step: int) -> bool:
    if not cfg.refine_gaussians or step >= int(cfg.refine_stop_iter):
        return False
    means2d = meta.get("means2d")
    if not isinstance(means2d, Tensor) or not means2d.requires_grad:
        return False
    means2d.retain_grad()
    return True


def _update_refinement_state(state: dict, meta: dict) -> None:
    means2d = meta.get("means2d")
    radii = meta.get("radii")
    if not isinstance(means2d, Tensor) or not isinstance(radii, Tensor):
        return
    grads = means2d.grad
    if grads is None:
        return

    n_gaussians = int(means2d.shape[-2])
    if state["grad2d"].shape[0] != n_gaussians:
        state["grad2d"] = torch.zeros(n_gaussians, device=grads.device)
        state["count"] = torch.zeros(n_gaussians, device=grads.device)
        state["radii"] = torch.zeros(n_gaussians, device=grads.device)

    flat_grads = grads.reshape(-1, n_gaussians, 2).detach()
    flat_radii = radii.reshape(-1, n_gaussians, 2).detach()
    visible = (flat_radii > 0).all(dim=-1)
    if not bool(visible.any().detach().cpu().item()):
        return

    width = float(meta.get("width", 1))
    height = float(meta.get("height", 1))
    camera_count = max(int(flat_grads.shape[0]), 1)
    flat_grads = flat_grads.clone()
    flat_grads[..., 0] *= width * 0.5 * camera_count
    flat_grads[..., 1] *= height * 0.5 * camera_count
    grad_norm = flat_grads.norm(dim=-1) * visible.to(flat_grads.dtype)
    grad_norm = _finite_or_zero(grad_norm).clamp_min(0.0)

    state["grad2d"] = state["grad2d"].to(grad_norm.device)
    state["count"] = state["count"].to(grad_norm.device)
    state["radii"] = state["radii"].to(grad_norm.device)
    state["grad2d"] = _finite_or_zero(state["grad2d"]).clamp_min(0.0)
    state["count"] = _finite_or_zero(state["count"]).clamp_min(0.0)
    state["radii"] = _finite_or_zero(state["radii"]).clamp_min(0.0)
    state["grad2d"].add_(grad_norm.sum(dim=0))
    state["count"].add_(visible.to(torch.float32).sum(dim=0))

    max_radius = flat_radii.max(dim=-1).values / max(width, height, 1.0)
    max_radius = torch.where(visible, max_radius, torch.zeros_like(max_radius))
    max_radius = _finite_or_zero(max_radius).clamp_min(0.0)
    state["radii"] = torch.maximum(state["radii"], max_radius.max(dim=0).values)


@torch.no_grad()
def _duplicate_gaussians(
    params: torch.nn.ParameterDict,
    optimizer: torch.optim.Optimizer,
    mask: Tensor,
    state: Optional[dict] = None,
) -> int:
    sel = torch.where(mask)[0]
    if sel.numel() == 0:
        return 0

    for name in list(params.keys()):
        value = params[name].detach()
        new_value = torch.cat((value, value[sel]), dim=0)

        def resize_state(state_value: Tensor, sel=sel) -> Tensor:
            extra = torch.zeros((sel.numel(), *state_value.shape[1:]), device=state_value.device)
            return torch.cat((state_value, extra), dim=0)

        _replace_param_with_optimizer_state(
            params, optimizer, name, new_value, resize_state
        )
    if state is not None:
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                state[key] = torch.cat((value, value[sel]), dim=0)
    return int(sel.numel())


@torch.no_grad()
def _split_gaussians(
    params: torch.nn.ParameterDict,
    optimizer: torch.optim.Optimizer,
    mask: Tensor,
    state: Optional[dict] = None,
) -> int:
    sel = torch.where(mask)[0]
    if sel.numel() == 0:
        return 0
    rest = torch.where(~mask)[0]
    device = params["means"].device
    scales = torch.exp(params["scales"].detach()[sel])
    quats = F.normalize(params["quats"].detach()[sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)
    noise = torch.randn(2, sel.numel(), 3, device=device, dtype=scales.dtype)
    samples = torch.einsum("nij,nj,bnj->bni", rotmats, scales, noise)

    for name in list(params.keys()):
        value = params[name].detach()
        repeats = [2] + [1] * (value.dim() - 1)
        if name == "means":
            split_value = (value[sel].unsqueeze(0) + samples).reshape(-1, 3)
        elif name == "scales":
            split_value = torch.log(scales / 1.6).repeat(2, 1)
        else:
            split_value = value[sel].repeat(repeats)
        new_value = torch.cat((value[rest], split_value), dim=0)

        def resize_state(state_value: Tensor, rest=rest, sel_count=sel.numel()) -> Tensor:
            split_state = torch.zeros(
                (2 * sel_count, *state_value.shape[1:]),
                device=state_value.device,
            )
            return torch.cat((state_value[rest], split_state), dim=0)

        _replace_param_with_optimizer_state(
            params, optimizer, name, new_value, resize_state
        )
    if state is not None:
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                repeats = [2] + [1] * (value.dim() - 1)
                split_value = value[sel].repeat(repeats)
                state[key] = torch.cat((value[rest], split_value), dim=0)
    return int(sel.numel())


@torch.no_grad()
def _prune_gaussians(
    params: torch.nn.ParameterDict,
    optimizer: torch.optim.Optimizer,
    mask: Tensor,
    state: Optional[dict] = None,
) -> int:
    if mask.numel() == 0:
        return 0
    if bool(mask.all().detach().cpu().item()):
        keep = int(torch.argmax(torch.sigmoid(params["opacities"].detach())).item())
        mask = mask.clone()
        mask[keep] = False
    sel = torch.where(~mask)[0]
    if sel.numel() == mask.numel():
        return 0

    for name in list(params.keys()):
        value = params[name].detach()
        new_value = value[sel]

        def resize_state(state_value: Tensor, sel=sel) -> Tensor:
            return state_value[sel]

        _replace_param_with_optimizer_state(
            params, optimizer, name, new_value, resize_state
        )
    if state is not None:
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                state[key] = value[sel]
    return int(mask.sum().detach().cpu().item())


@torch.no_grad()
def _reset_opacities(
    params: torch.nn.ParameterDict,
    optimizer: torch.optim.Optimizer,
    value: float,
) -> bool:
    target = torch.logit(
        torch.tensor(
            float(value),
            device=params["opacities"].device,
            dtype=params["opacities"].dtype,
        )
    )
    with torch.no_grad():
        params["opacities"].data.copy_(torch.minimum(params["opacities"].data, target))
    _zero_optimizer_state_for_param(optimizer, params["opacities"])
    return True


def _limit_growth_masks(
    duplicate_mask: Tensor,
    split_mask: Tensor,
    scores: Tensor,
    capacity: int,
) -> Tuple[Tensor, Tensor, dict]:
    duplicate_mask = duplicate_mask & ~split_mask
    stats = {
        "growth_capacity": max(int(capacity), 0),
        "growth_selected_duplicate": 0,
        "growth_selected_split": 0,
        "growth_dropped_duplicate": _mask_count(duplicate_mask),
        "growth_dropped_split": _mask_count(split_mask),
        "growth_score_selected": None,
        "growth_score_dropped": None,
    }
    if capacity <= 0:
        return torch.zeros_like(duplicate_mask), torch.zeros_like(split_mask), stats

    selected_duplicate = torch.zeros_like(duplicate_mask)
    selected_split = torch.zeros_like(split_mask)
    remaining = int(capacity)

    split_count = _mask_count(split_mask)
    if split_count > 0 and remaining > 0:
        split_ids = torch.where(split_mask)[0]
        keep_count = min(split_count, remaining)
        keep = split_ids[torch.topk(scores[split_ids], k=keep_count).indices]
        selected_split[keep] = True
        remaining -= keep_count

    duplicate_count = _mask_count(duplicate_mask)
    if duplicate_count > 0 and remaining > 0:
        duplicate_ids = torch.where(duplicate_mask)[0]
        keep_count = min(duplicate_count, remaining)
        keep = duplicate_ids[torch.topk(scores[duplicate_ids], k=keep_count).indices]
        selected_duplicate[keep] = True

    selected = selected_duplicate | selected_split
    dropped = (duplicate_mask | split_mask) & ~selected
    stats.update(
        {
            "growth_selected_duplicate": _mask_count(selected_duplicate),
            "growth_selected_split": _mask_count(selected_split),
            "growth_dropped_duplicate": _mask_count(duplicate_mask & ~selected_duplicate),
            "growth_dropped_split": _mask_count(split_mask & ~selected_split),
            "growth_score_selected": _score_summary(scores, selected),
            "growth_score_dropped": _score_summary(scores, dropped),
        }
    )
    return selected_duplicate, selected_split, stats


def _mask_count(mask: Tensor) -> int:
    return int(mask.sum().detach().cpu().item())


def _mean_of_masked(values: Tensor, mask: Tensor) -> Optional[float]:
    if values.numel() == 0 or not bool(mask.any().detach().cpu().item()):
        return None
    return float(values[mask].mean().detach().cpu().item())


def _max_of_masked(values: Tensor, mask: Tensor) -> Optional[float]:
    if values.numel() == 0 or not bool(mask.any().detach().cpu().item()):
        return None
    return float(values[mask].max().detach().cpu().item())


@torch.no_grad()
def _apply_refinement(
    params: torch.nn.ParameterDict,
    optimizer: torch.optim.Optimizer,
    state: dict,
    cfg: TrainConfig,
    step: int,
    scene_scale: float,
) -> dict:
    stats = {
        "duplicated": 0,
        "split": 0,
        "pruned": 0,
        "opacity_reset": False,
        "num_gaussians": int(params["means"].shape[0]),
        "observed": 0,
        "grad2d_mean": None,
        "grad2d_max": None,
        "radii_max": None,
        "candidate_duplicate": 0,
        "candidate_split": 0,
        "candidate_prune": 0,
        "effective_refine_scale2d_stop_iter": _effective_refine_scale2d_stop_iter(cfg),
        "growth_capacity": None,
        "growth_selected_duplicate": 0,
        "growth_selected_split": 0,
        "growth_dropped_duplicate": 0,
        "growth_dropped_split": 0,
        "growth_score_candidate": None,
        "growth_score_selected": None,
        "growth_score_dropped": None,
    }
    if not cfg.refine_gaussians:
        return stats

    should_refine = (
        step > int(cfg.refine_start_iter)
        and step < int(cfg.refine_stop_iter)
        and step % max(int(cfg.refine_every), 1) == 0
        and step % max(int(cfg.reset_every), 1) >= int(cfg.pause_refine_after_reset)
    )
    if should_refine:
        counts = state["count"].clamp_min(1.0)
        grads = _finite_or_zero(state["grad2d"] / counts).clamp_min(0.0)
        radii = _finite_or_zero(state["radii"]).clamp_min(0.0)
        observed = state["count"] > 0
        stats["observed"] = _mask_count(observed)
        stats["grad2d_mean"] = _mean_of_masked(grads, observed)
        stats["grad2d_max"] = _max_of_masked(grads, observed)
        stats["radii_max"] = _max_of_masked(radii, observed)
        high_grad = grads > float(cfg.grow_grad2d)
        max_scale = _finite_or_zero(torch.exp(params["scales"].detach())).max(dim=-1).values
        opacity = _finite_or_zero(torch.sigmoid(params["opacities"].detach()))
        small = max_scale <= float(cfg.grow_scale3d) * float(scene_scale)
        duplicate_mask = high_grad & small
        split_mask = high_grad & ~small
        if step < int(stats["effective_refine_scale2d_stop_iter"]):
            split_mask |= radii > float(cfg.grow_scale2d)
        duplicate_mask = duplicate_mask & ~split_mask
        scores = _growth_scores(grads, radii, max_scale, opacity)
        stats["candidate_duplicate"] = _mask_count(duplicate_mask)
        stats["candidate_split"] = _mask_count(split_mask)
        stats["growth_score_candidate"] = _score_summary(
            scores,
            duplicate_mask | split_mask,
        )

        capacity = _effective_growth_capacity(cfg, step, int(params["means"].shape[0]))
        if capacity < 2**63 - 1:
            duplicate_mask, split_mask, growth_stats = _limit_growth_masks(
                duplicate_mask,
                split_mask,
                scores,
                capacity,
            )
            stats.update(growth_stats)
        else:
            stats.update(
                {
                    "growth_capacity": None,
                    "growth_selected_duplicate": _mask_count(duplicate_mask),
                    "growth_selected_split": _mask_count(split_mask),
                    "growth_dropped_duplicate": 0,
                    "growth_dropped_split": 0,
                    "growth_score_selected": _score_summary(
                        scores, duplicate_mask | split_mask
                    ),
                    "growth_score_dropped": _score_summary(
                        scores, torch.zeros_like(duplicate_mask)
                    ),
                }
            )

        stats["duplicated"] = _duplicate_gaussians(
            params, optimizer, duplicate_mask, state=state
        )
        if stats["duplicated"] > 0:
            split_mask = torch.cat(
                (
                    split_mask,
                    torch.zeros(
                        stats["duplicated"],
                        dtype=torch.bool,
                        device=split_mask.device,
                    ),
                )
            )
        stats["split"] = _split_gaussians(params, optimizer, split_mask, state=state)

        prune_mask = torch.sigmoid(params["opacities"].detach()) < float(cfg.prune_opa)
        if int(cfg.reset_every) > 0 and step > int(cfg.reset_every):
            too_big = (
                _finite_or_zero(torch.exp(params["scales"].detach())).max(dim=-1).values
                > float(cfg.prune_scale3d) * float(scene_scale)
            )
            if (
                step < int(stats["effective_refine_scale2d_stop_iter"])
                and radii.shape[0] == too_big.shape[0]
            ):
                too_big |= radii > float(cfg.prune_scale2d)
            prune_mask |= too_big
        stats["candidate_prune"] = _mask_count(prune_mask)
        stats["pruned"] = _prune_gaussians(params, optimizer, prune_mask, state=state)
        _reset_refinement_state(state, params)

    if int(cfg.reset_every) > 0 and step % int(cfg.reset_every) == 0 and step > 0:
        stats["opacity_reset"] = _reset_opacities(
            params,
            optimizer,
            float(cfg.prune_opa) * 2.0,
        )
    stats["num_gaussians"] = int(params["means"].shape[0])
    return stats


def _refinement_event(stats: dict) -> bool:
    return (
        int(stats.get("duplicated", 0)) > 0
        or int(stats.get("split", 0)) > 0
        or int(stats.get("pruned", 0)) > 0
        or bool(stats.get("opacity_reset", False))
    )


def _finite_tensor_summary(values: Tensor) -> dict:
    flat = values.detach().cpu().float().reshape(-1)
    total = int(flat.numel())
    finite = flat[torch.isfinite(flat)]
    summary = {
        "count": total,
        "finite": int(finite.numel()),
        "nonfinite": total - int(finite.numel()),
    }
    if finite.numel() == 0:
        return summary

    quantiles = torch.quantile(
        finite,
        torch.tensor([0.5, 0.9, 0.99], dtype=finite.dtype),
    )
    summary.update(
        {
            "min": float(finite.min().item()),
            "mean": float(finite.mean().item()),
            "std": float(finite.std(unbiased=False).item()),
            "p50": float(quantiles[0].item()),
            "p90": float(quantiles[1].item()),
            "p99": float(quantiles[2].item()),
            "max": float(finite.max().item()),
        }
    )
    return summary


def _fraction_below(values: Tensor, threshold: float) -> Optional[float]:
    flat = values.detach().cpu().float().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return None
    return float((finite < float(threshold)).to(torch.float32).mean().item())


def _fraction_above(values: Tensor, threshold: float) -> Optional[float]:
    flat = values.detach().cpu().float().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return None
    return float((finite > float(threshold)).to(torch.float32).mean().item())


@torch.no_grad()
def _training_diagnostics(
    params: torch.nn.ParameterDict,
    refinement_state: dict,
    cfg: TrainConfig,
    scene_scale: float,
    exposure_params: Optional[torch.nn.ParameterDict] = None,
) -> dict:
    scales = torch.exp(params["scales"].detach())
    max_scale = scales.max(dim=-1).values
    opacity = torch.sigmoid(params["opacities"].detach())
    scene_scale = max(float(scene_scale), 1e-6)
    diagnostics = {
        "params": {
            "num_gaussians": int(params["means"].shape[0]),
            "means_norm": _finite_tensor_summary(
                params["means"].detach().norm(dim=-1)
            ),
            "opacity": _finite_tensor_summary(opacity),
            "opacity_below_prune_fraction": _fraction_below(
                opacity,
                float(cfg.prune_opa),
            ),
            "scale": _finite_tensor_summary(scales),
            "max_scale": _finite_tensor_summary(max_scale),
            "max_scale_scene_fraction": _finite_tensor_summary(max_scale / scene_scale),
            "max_scale_above_grow3d_fraction": _fraction_above(
                max_scale,
                float(cfg.grow_scale3d) * scene_scale,
            ),
            "max_scale_above_prune3d_fraction": _fraction_above(
                max_scale,
                float(cfg.prune_scale3d) * scene_scale,
            ),
            "quat_norm": _finite_tensor_summary(
                params["quats"].detach().norm(dim=-1)
            ),
            "color": _finite_tensor_summary(
                _sh_to_rgb(_sh0_param_to_3d(params["sh0"]).detach()).clamp(0.0, 1.0)
            ),
        }
    }
    if "shN" in params:
        diagnostics["params"]["shN_abs"] = _finite_tensor_summary(params["shN"].detach().abs())
    if exposure_params is not None:
        diagnostics["exposure"] = {
            "log_scale": _finite_tensor_summary(exposure_params["log_scale"].detach()),
            "scale": _finite_tensor_summary(
                torch.exp(exposure_params["log_scale"].detach())
            ),
            "bias": _finite_tensor_summary(exposure_params["bias"].detach()),
        }
    else:
        diagnostics["exposure"] = None

    count = refinement_state.get("count")
    grad2d = refinement_state.get("grad2d")
    radii = refinement_state.get("radii")
    if isinstance(count, Tensor) and isinstance(grad2d, Tensor) and isinstance(radii, Tensor):
        count_cpu = count.detach().cpu().float()
        observed = count_cpu > 0
        observed_count = int(observed.sum().item())
        grad_avg = (grad2d.detach().cpu().float() / count_cpu.clamp_min(1.0))[observed]
        radii_observed = radii.detach().cpu().float()[observed]
        diagnostics["refinement_state"] = {
            "observed": observed_count,
            "observed_fraction": (
                float(observed_count / max(int(count_cpu.numel()), 1))
                if count_cpu.numel() > 0
                else None
            ),
            "count": _finite_tensor_summary(count_cpu),
            "grad2d_observed": _finite_tensor_summary(grad_avg),
            "radii_observed": _finite_tensor_summary(radii_observed),
        }
    else:
        diagnostics["refinement_state"] = None
    return diagnostics


def _state_to_cpu(value):
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(item) for item in value)
    return value


def _clone_state_value(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_state_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_value(item) for item in value)
    return copy.deepcopy(value)


def _jsonable_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    return value


def _config_metadata(cfg: TrainConfig) -> dict:
    return {field.name: _jsonable_value(getattr(cfg, field.name)) for field in fields(cfg)}


def _resolved_config_metadata(cfg: TrainConfig, records: List[dict]) -> dict:
    if records:
        config = records[0].get("config")
        if isinstance(config, dict):
            return config
    return _config_metadata(cfg)


def _meta_value(primary: dict, fallback: dict, key: str):
    return primary.get(key, fallback.get(key))


def _effective_refine_scale2d_stop_iter(cfg: TrainConfig) -> int:
    profile = str(cfg.refine_profile).replace("_", "-").lower()
    manual_value = int(cfg.refine_scale2d_stop_iter)
    if profile == "manual":
        return manual_value
    if profile == "full-stable":
        return 0
    if profile == "crop-balanced":
        if int(cfg.train_crop_size) <= 0:
            return 0
        return manual_value if manual_value > 0 else int(cfg.refine_stop_iter)
    raise ValueError(
        "refine_profile must be one of manual, full-stable, or crop-balanced; "
        f"got {cfg.refine_profile!r}"
    )


def _effective_growth_capacity(cfg: TrainConfig, step: int, current_gaussians: int) -> int:
    max_gaussians = int(cfg.max_gaussians)
    if max_gaussians <= 0:
        return 2**63 - 1
    target = max_gaussians
    ramp_steps = int(cfg.max_gaussians_ramp_steps)
    if ramp_steps > 0:
        progress = min(max(step, 0), ramp_steps) / float(ramp_steps)
        target = int(round(current_gaussians + (max_gaussians - current_gaussians) * progress))
        target = min(max(target, current_gaussians), max_gaussians)
    return max(target - int(current_gaussians), 0)


def _growth_scores(grads: Tensor, radii: Tensor, max_scale: Tensor, opacity: Tensor) -> Tensor:
    scale_score = max_scale / max_scale.detach().amax().clamp_min(1e-12)
    opacity_score = opacity.clamp_min(1e-6).sqrt()
    return grads.clamp_min(0.0) * (1.0 + radii.clamp_min(0.0)) * (1.0 + scale_score) * opacity_score


def _score_summary(values: Tensor, mask: Tensor) -> dict:
    if values.numel() == 0 or not bool(mask.any().detach().cpu().item()):
        return _finite_tensor_summary(values.new_empty((0,)))
    return _finite_tensor_summary(values[mask])


def _rank_range(total: int, world_rank: int, world_size: int) -> Tuple[int, int]:
    base, remainder = divmod(total, world_size)
    start = world_rank * base + min(world_rank, remainder)
    stop = start + base + (1 if world_rank < remainder else 0)
    return start, stop


def _checkpoint_rank_range(
    ckpt: dict,
    world_rank: int,
    world_size: int,
    total: int,
) -> Tuple[int, int]:
    rank_shards = ckpt.get("rank_shards")
    if rank_shards is None:
        return _rank_range(total, world_rank, world_size)
    for item in rank_shards:
        if int(item["world_rank"]) == int(world_rank):
            return int(item["start"]), int(item["stop"])
    raise ValueError(
        f"distributed checkpoint is missing shard metadata for rank {world_rank}"
    )


def create_model_shard(
    data: MultiviewData,
    cfg: TrainConfig,
    device: torch.device,
    world_rank: int,
    world_size: int,
) -> Tuple[torch.nn.ParameterDict, int]:
    full_params = create_model(data, cfg, torch.device("cpu"))
    n_gaussians = int(full_params["means"].shape[0])
    start, stop = _rank_range(n_gaussians, world_rank, world_size)
    params = torch.nn.ParameterDict(
        {
            name: torch.nn.Parameter(
                full_params[name].detach()[start:stop].clone().contiguous()
            )
            for name in full_params.keys()
        }
    )
    return params.to(device), n_gaussians


def _splat_keys(splats: Dict[str, Tensor]) -> List[str]:
    return [name for name in (*SPLAT_KEYS, *OPTIONAL_SPLAT_KEYS) if name in splats]


def _shard_splats(
    splats: Dict[str, Tensor],
    device: torch.device,
    start: int,
    stop: int,
    sh_degree: int = 0,
) -> torch.nn.ParameterDict:
    values = _checkpoint_values_from_splats(splats, sh_degree)
    values = {
        name: torch.nn.Parameter(
            value.detach().float()[start:stop].clone().contiguous()
        )
        for name, value in values.items()
    }
    params = torch.nn.ParameterDict(values)
    return params.to(device)


def _load_checkpoint(
    path: Path,
    device: torch.device,
    sh_degree: int = 0,
) -> Tuple[torch.nn.ParameterDict, int, dict]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "splats" not in ckpt:
        raise ValueError(f"checkpoint {path} does not contain a splats dictionary")
    params = _model_from_splats(ckpt["splats"], device, sh_degree=sh_degree)
    step = int(ckpt.get("step", 0))
    return params, step, ckpt


def _load_distributed_resume_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "splats" not in ckpt:
        raise ValueError(f"checkpoint {path} does not contain a splats dictionary")

    config = ckpt.get("config", {})
    config_is_distributed = isinstance(config, dict) and bool(config.get("distributed", False))
    checkpoint_is_distributed_root = bool(ckpt.get("distributed", False))
    if checkpoint_is_distributed_root:
        missing = [
            key for key in ("world_size", "rank_shards") if key not in ckpt
        ]
        if missing:
            raise ValueError(
                f"distributed checkpoint {path} is missing root metadata {missing}; "
                "resume from the root distributed checkpoint, not a rank shard"
            )
    elif config_is_distributed:
        raise ValueError(
            f"checkpoint {path} looks like a per-rank distributed shard; "
            "resume distributed training from the root checkpoint under ckpts/, "
            "or from a single-card checkpoint"
        )
    return ckpt


def _optimizer_state_param_names(optimizer_state: dict) -> Dict[int, str]:
    names = {}
    for group in optimizer_state.get("param_groups", []):
        name = group.get("name")
        if not name:
            continue
        for param_id in group.get("params", []):
            names[int(param_id)] = str(name)
    return names


def _is_optimizer_dict_state(optimizer_state: dict) -> bool:
    return isinstance(optimizer_state, dict) and optimizer_state.get("format") == "optimizer_dict"


def _merge_optimizer_states(
    optimizer_states: List[dict],
    shard_sizes: List[int],
) -> dict:
    if not optimizer_states:
        return {}
    if not all(_is_optimizer_dict_state(state) for state in optimizer_states):
        return {}
    names = sorted(
        set().union(
            *[
                set(state.get("optimizers", {}).keys())
                for state in optimizer_states
            ]
        )
    )
    return {
        "format": "optimizer_dict",
        "optimizers": {
            name: _merge_single_optimizer_states(
                [
                    state["optimizers"][name]
                    for state in optimizer_states
                    if name in state.get("optimizers", {})
                ],
                shard_sizes,
            )
            for name in names
        },
    }


def _merge_single_optimizer_states(
    optimizer_states: List[dict],
    shard_sizes: List[int],
) -> dict:
    merged = {
        "state": {},
        "param_groups": copy.deepcopy(optimizer_states[0].get("param_groups", [])),
    }
    first_state = optimizer_states[0].get("state", {})
    param_names = _optimizer_state_param_names(optimizer_states[0])
    for param_id, state in first_state.items():
        merged_state = {}
        for state_key in state:
            values = [
                optimizer_state.get("state", {}).get(param_id, {}).get(state_key)
                for optimizer_state in optimizer_states
            ]
            if all(isinstance(value, Tensor) for value in values):
                tensors = [value.detach().cpu() for value in values]
                if param_names.get(int(param_id)) == "exposure":
                    merged_state[state_key] = tensors[0].clone()
                elif all(
                    tensor.ndim > 0 and int(tensor.shape[0]) == int(shard_size)
                    for tensor, shard_size in zip(tensors, shard_sizes)
                ):
                    merged_state[state_key] = torch.cat(tensors, dim=0).contiguous()
                elif all(tensor.ndim == 0 for tensor in tensors):
                    merged_state[state_key] = torch.stack(
                        [tensor.reshape(()) for tensor in tensors], dim=0
                    ).max()
                else:
                    merged_state[state_key] = tensors[0].clone()
            else:
                merged_state[state_key] = _clone_state_value(values[0])
        merged["state"][param_id] = merged_state
    return merged


def _slice_optimizer_state(
    optimizer_state: dict,
    total_size: int,
    start: int,
    stop: int,
) -> dict:
    if not _is_optimizer_dict_state(optimizer_state):
        return {}
    return {
        "format": "optimizer_dict",
        "optimizers": {
            name: _slice_single_optimizer_state(state, total_size, start, stop)
            for name, state in optimizer_state.get("optimizers", {}).items()
        },
    }


def _slice_single_optimizer_state(
    optimizer_state: dict,
    total_size: int,
    start: int,
    stop: int,
) -> dict:
    sliced = {
        "state": {},
        "param_groups": copy.deepcopy(optimizer_state.get("param_groups", [])),
    }
    param_names = _optimizer_state_param_names(optimizer_state)
    for param_id, state in optimizer_state.get("state", {}).items():
        sliced_state = {}
        for state_key, value in state.items():
            if isinstance(value, Tensor):
                tensor = value.detach().cpu()
                if (
                    param_names.get(int(param_id)) != "exposure"
                    and tensor.ndim > 0
                    and int(tensor.shape[0]) == int(total_size)
                ):
                    sliced_state[state_key] = tensor[start:stop].clone().contiguous()
                else:
                    sliced_state[state_key] = tensor.clone()
            else:
                sliced_state[state_key] = _clone_state_value(value)
        sliced["state"][param_id] = sliced_state
    return sliced


def _sample_train_crop(
    frame: Frame,
    crop_size: int,
    *,
    seed: Optional[int] = None,
    step: Optional[int] = None,
    stream_rank: int = 0,
    stream_world_size: int = 1,
) -> Optional[CropWindow]:
    if crop_size <= 0:
        return None

    height, width = int(frame.image.shape[0]), int(frame.image.shape[1])
    crop_width = min(int(crop_size), width)
    crop_height = min(int(crop_size), height)
    if crop_width == width and crop_height == height:
        return None

    max_x = width - crop_width
    max_y = height - crop_height
    if seed is None or step is None:
        rng = random
    else:
        position = (int(step) - 1) * max(int(stream_world_size), 1) + int(stream_rank)
        rng = random.Random(int(seed) + 1_000_003 * position + 0xC0FFEE)

    def random_window() -> CropWindow:
        x = rng.randint(0, max_x) if max_x > 0 else 0
        y = rng.randint(0, max_y) if max_y > 0 else 0
        return CropWindow(x=x, y=y, width=crop_width, height=crop_height)

    if frame.mask is None:
        return random_window()

    # Masked datasets should not train on crops with no supervised pixels.
    # Try cheap random candidates first, then fall back to anchoring on a valid pixel.
    for _ in range(16):
        crop = random_window()
        y0, y1 = crop.y, crop.y + crop.height
        x0, x1 = crop.x, crop.x + crop.width
        if bool(frame.mask[y0:y1, x0:x1].any().item()):
            return crop

    valid_yx = torch.nonzero(frame.mask, as_tuple=False)
    if valid_yx.numel() == 0:
        return random_window()
    pixel_y, pixel_x = valid_yx[rng.randrange(int(valid_yx.shape[0]))].tolist()
    min_x = max(0, int(pixel_x) - crop_width + 1)
    max_anchor_x = min(int(pixel_x), max_x)
    min_y = max(0, int(pixel_y) - crop_height + 1)
    max_anchor_y = min(int(pixel_y), max_y)
    x = rng.randint(min_x, max_anchor_x) if max_anchor_x > min_x else min_x
    y = rng.randint(min_y, max_anchor_y) if max_anchor_y > min_y else min_y
    return CropWindow(x=x, y=y, width=crop_width, height=crop_height)


def _crop_image_and_mask(
    frame: Frame,
    crop: Optional[CropWindow],
) -> Tuple[Tensor, Optional[Tensor]]:
    if crop is None:
        return frame.image, frame.mask

    y0, y1 = crop.y, crop.y + crop.height
    x0, x1 = crop.x, crop.x + crop.width
    image = frame.image[y0:y1, x0:x1].contiguous()
    mask = frame.mask[y0:y1, x0:x1].contiguous() if frame.mask is not None else None
    return image, mask


def _crop_K(K: Tensor, crop: Optional[CropWindow]) -> Tensor:
    if crop is None:
        return K

    cropped = K.clone()
    cropped[0, 2] -= float(crop.x)
    cropped[1, 2] -= float(crop.y)
    return cropped.contiguous()


def _crop_metadata(crop: Optional[CropWindow]) -> Optional[dict]:
    if crop is None:
        return None
    return {
        "x": crop.x,
        "y": crop.y,
        "width": crop.width,
        "height": crop.height,
    }


def _apply_random_background(render: Tensor, alpha: Tensor) -> Tensor:
    background = torch.rand(
        (1, render.shape[-1]),
        device=render.device,
        dtype=render.dtype,
    )
    return render + background * (1.0 - alpha)


def _rasterize_basic(
    params: torch.nn.ParameterDict,
    frame: Frame,
    device: torch.device,
    cfg: TrainConfig,
    distributed: bool = False,
    *,
    crop: Optional[CropWindow] = None,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    sh_degree: int = 0,
    debug_prefix: str = "",
) -> Tuple[Tensor, Tensor, dict]:
    use_public_rasterization = distributed or device.type == "supa"
    if use_public_rasterization:
        from gsplat import rasterization  # pylint: disable=import-outside-toplevel
    else:
        from gsplat._torch_fallback import rasterization  # pylint: disable=import-outside-toplevel

    image_cpu, _ = _crop_image_and_mask(frame, crop)
    K = _crop_K(frame.K, crop)
    image = image_cpu.to(device)
    colors: Tensor = _sh0_param_to_3d(params["sh0"])
    active_sh_degree: Optional[int] = 0
    if sh_degree > 0 and "shN" in params:
        colors = torch.cat((colors, _sh_param_to_3d(params["shN"])), dim=1)
        active_sh_degree = int(sh_degree)
    kwargs = {
        "means": params["means"],
        "quats": params["quats"],
        "scales": torch.exp(params["scales"]),
        "opacities": torch.sigmoid(params["opacities"]),
        "colors": colors,
        "viewmats": frame.viewmat.to(device).unsqueeze(0),
        "Ks": K.to(device).unsqueeze(0),
        "width": image.shape[1],
        "height": image.shape[0],
        "near_plane": float(near_plane),
        "far_plane": float(far_plane),
        "radius_clip": float(radius_clip),
        "eps2d": 0.3,
        "sh_degree": active_sh_degree,
        "tile_size": 16,
        "backgrounds": None,
        "render_mode": "RGB",
        "rasterize_mode": "classic",
        "camera_model": "pinhole",
        "covars": None,
        "extra_signals": None,
        "extra_signals_sh_degree": None,
        "supa_native_options": _supa_native_options(cfg),
    }
    if use_public_rasterization:
        kwargs.update(
            {
                "packed": False,
                "distributed": bool(distributed),
                "channel_chunk": 32,
                "supa_use_torch_fallback": device.type == "supa",
            }
        )
    else:
        kwargs["chunk_size"] = 32
    _debug_log(
        cfg.print_debug_info,
        "rasterize.begin",
        prefix=debug_prefix,
        frame=frame.name,
        use_public_rasterization=use_public_rasterization,
        distributed=distributed,
        crop=crop,
        image=image,
        means=kwargs["means"],
        quats=kwargs["quats"],
        scales=kwargs["scales"],
        opacities=kwargs["opacities"],
        colors=kwargs["colors"],
        viewmats=kwargs["viewmats"],
        Ks=kwargs["Ks"],
        width=kwargs["width"],
        height=kwargs["height"],
        sh_degree=active_sh_degree,
        native_options=kwargs["supa_native_options"],
    )
    render, alpha, meta = rasterization(**kwargs)
    crop_meta = _crop_metadata(crop)
    if crop_meta is not None:
        meta["train_crop"] = crop_meta
    _debug_log(
        cfg.print_debug_info,
        "rasterize.done",
        prefix=debug_prefix,
        render=render,
        alpha=alpha,
        meta=_debug_meta_summary(meta),
    )
    return render[0], alpha[0], meta


def _save_image(image: Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(array).save(path)


def _pad_image_width(image: Tensor, width: int) -> Tensor:
    if image.shape[1] == width:
        return image
    padded = torch.zeros((image.shape[0], width, image.shape[2]), dtype=image.dtype)
    padded[:, : image.shape[1]] = image
    return padded


def _save_eval_grid(rows: List[Tuple[Tensor, Tensor]], path: Path) -> Optional[str]:
    if not rows:
        return None

    row_images = []
    max_width = 0
    for target, render in rows:
        target = target.detach().cpu().clamp(0.0, 1.0)
        render = render.detach().cpu().clamp(0.0, 1.0)
        error = (render - target).abs().mul(4.0).clamp(0.0, 1.0)
        row = torch.cat((target, render, error), dim=1)
        row_images.append(row)
        max_width = max(max_width, int(row.shape[1]))

    grid = torch.cat([_pad_image_width(row, max_width) for row in row_images], dim=0)
    _save_image(grid, path)
    return str(path)


def _load_saved_image(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).contiguous()


def _save_image_stack(images: List[Tensor], path: Path) -> Optional[str]:
    if not images:
        return None

    rows = []
    max_width = 0
    for image in images:
        image = image.detach().cpu().clamp(0.0, 1.0)
        rows.append(image)
        max_width = max(max_width, int(image.shape[1]))

    grid = torch.cat([_pad_image_width(row, max_width) for row in rows], dim=0)
    _save_image(grid, path)
    return str(path)


def _psnr_from_mse(mse: float) -> Optional[float]:
    if mse <= 0.0 or not math.isfinite(mse):
        return None
    return -10.0 * math.log10(mse)


def _render_psnr(
    render: Tensor,
    target: Tensor,
    mask_cpu: Optional[Tensor],
) -> Optional[float]:
    with torch.no_grad():
        render = render.detach()
        target = target.detach()
        if mask_cpu is not None and bool(mask_cpu.any().item()):
            mask = mask_cpu.to(render.device)
            diff = render[mask] - target[mask]
        else:
            diff = render - target
        if diff.numel() == 0:
            return None
        mse = float(diff.square().mean().detach().cpu().item())
    return _psnr_from_mse(mse)


def _image_metrics(render: Tensor, target: Tensor, mask: Optional[Tensor]) -> Tuple[float, float, int]:
    if mask is not None:
        mask = mask.to(render.device)
        pixel_count = int(mask.sum().detach().cpu().item())
        render = render[mask]
        target = target[mask]
    else:
        pixel_count = int(render.shape[0] * render.shape[1])
    diff = render - target
    l1 = float(diff.abs().mean().detach().cpu().item())
    mse = float(diff.square().mean().detach().cpu().item())
    return l1, mse, pixel_count


def _evaluate_frames(
    params: torch.nn.ParameterDict,
    frames: List[Frame],
    device: torch.device,
    cfg: TrainConfig,
    result_dir: Path,
    max_frames: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    sh_degree: int,
    exposure_params: Optional[torch.nn.ParameterDict] = None,
    data: Optional[MultiviewData] = None,
    eval_frame_offset: int = 0,
    frames_are_eval: bool = True,
) -> Tuple[Tensor, dict, dict]:
    if not frames:
        raise ValueError("at least one frame is required for evaluation")

    metric_limit = max(0, min(len(frames), max_frames))
    render_limit = max(1, metric_limit)
    frame_metrics = []
    grid_rows = []
    final_render = None
    final_meta = {}
    for index, frame in enumerate(frames[:render_limit]):
        render, _, meta = _rasterize_basic(
            params,
            frame,
            device,
            cfg,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            sh_degree=sh_degree,
            debug_prefix="eval.",
        )
        if exposure_params is not None and data is not None:
            frame_id = _exposure_frame_id(
                eval_frame_offset + index,
                is_eval=frames_are_eval,
                data=data,
            )
            render = _apply_exposure(render, exposure_params, frame_id)
            meta["exposure_frame_id"] = frame_id
        if final_render is None:
            final_render = render
            final_meta = meta
        if index >= metric_limit:
            continue

        target = frame.image.to(device)
        l1, mse, pixel_count = _image_metrics(render, target, frame.mask)
        grid_rows.append((target, render))
        frame_metrics.append(
            {
                "name": frame.name,
                "mask_pixels": pixel_count,
                "l1": l1,
                "mse": mse,
                "psnr": _psnr_from_mse(mse),
                "projection_backend": meta.get("projection_backend"),
                "composite_backend": meta.get("composite_backend"),
            }
        )
        _save_image(render, result_dir / "renders" / f"eval_{index:04d}_{frame.name}.png")

    if frame_metrics:
        l1 = float(sum(item["l1"] for item in frame_metrics) / len(frame_metrics))
        mse = float(sum(item["mse"] for item in frame_metrics) / len(frame_metrics))
    else:
        l1 = None
        mse = None
    grid_path = _save_eval_grid(grid_rows, result_dir / "renders" / "eval_grid.png")
    metrics = {
        "num_frames": len(frame_metrics),
        "l1": l1,
        "mse": mse,
        "psnr": _psnr_from_mse(mse) if mse is not None else None,
        "grid": grid_path,
        "frames": frame_metrics,
    }
    return final_render, final_meta, metrics


def _save_checkpoint(
    params: torch.nn.ParameterDict,
    optimizer,
    cfg: TrainConfig,
    step: int,
    path: Path,
    exposure_params: Optional[torch.nn.ParameterDict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "config": _config_metadata(cfg),
        "splats": _params_splats_for_checkpoint(params),
        "optimizer": _optimizer_state_dict(optimizer),
    }
    exposure_state = _exposure_state_for_checkpoint(exposure_params)
    if exposure_state is not None:
        payload["exposure"] = exposure_state
    torch.save(payload, path)


def _rank_checkpoint_path(result_dir: Path, world_rank: int, name: str) -> Path:
    return result_dir / f"rank_{world_rank:04d}" / "ckpts" / name


def _merge_splats_from_checkpoints(checkpoints: List[dict]) -> Dict[str, Tensor]:
    keys = _splat_keys(checkpoints[0]["splats"])
    return {
        name: torch.cat([ckpt["splats"][name].detach().cpu() for ckpt in checkpoints], dim=0)
        for name in keys
    }


def _save_global_distributed_checkpoint(
    cfg: TrainConfig,
    step: int,
    world_size: int,
    checkpoint_name: str,
    output_path: Path,
) -> Dict[str, Tensor]:
    checkpoints = [
        torch.load(_rank_checkpoint_path(cfg.result_dir, rank, checkpoint_name), map_location="cpu")
        for rank in range(world_size)
    ]
    config = _resolved_config_metadata(cfg, checkpoints)
    splats = _merge_splats_from_checkpoints(checkpoints)
    shard_sizes = [int(ckpt["splats"]["means"].shape[0]) for ckpt in checkpoints]
    optimizer = _merge_optimizer_states(
        [ckpt["optimizer"] for ckpt in checkpoints if "optimizer" in ckpt],
        shard_sizes,
    )

    shards = []
    start = 0
    for rank, size in enumerate(shard_sizes):
        stop = start + size
        shards.append({"world_rank": rank, "start": start, "stop": stop})
        start = stop

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "config": config,
        "distributed": True,
        "world_size": world_size,
        "splats": splats,
        "optimizer": optimizer,
        "rank_checkpoints": [
            str(_rank_checkpoint_path(cfg.result_dir, rank, checkpoint_name))
            for rank in range(world_size)
        ],
        "rank_shards": shards,
    }
    if "exposure" in checkpoints[0]:
        payload["exposure"] = _clone_state_value(checkpoints[0]["exposure"])
    torch.save(payload, output_path)
    return splats


def _save_global_distributed_ply(
    splats: Dict[str, Tensor],
    path: Path,
    sh_degree: int,
) -> None:
    params = _model_from_splats(splats, torch.device("cpu"), sh_degree=sh_degree)
    _save_ply(params, path)


def _copy_if_exists(source: Path, target: Path) -> Optional[str]:
    if not source.is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return str(target)


def _save_distributed_train_step_render(
    cfg: TrainConfig,
    step: int,
    world_size: int,
) -> Optional[str]:
    images = []
    for rank in range(world_size):
        source = (
            cfg.result_dir
            / f"rank_{rank:04d}"
            / "renders"
            / f"train_step_{step:04d}.png"
        )
        if not source.is_file():
            raise FileNotFoundError(f"missing distributed train render: {source}")
        images.append(_load_saved_image(source))

    return _save_image_stack(
        images,
        cfg.result_dir / "renders" / f"train_step_{step:04d}.png",
    )


def _save_ply(params: torch.nn.ParameterDict, path: Path) -> None:
    """Save Gaussians in the common binary 3DGS PLY layout used by viewers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    means = params["means"].detach().cpu().float()
    f_dc = _sh0_param_to_3d(params["sh0"].detach().cpu().float()).squeeze(1)
    if "shN" in params:
        f_rest = _sh_param_to_3d(params["shN"]).detach().cpu().float().permute(0, 2, 1).reshape(
            means.shape[0], -1
        )
        if f_rest.shape[1] < 45:
            padding = torch.zeros(means.shape[0], 45 - f_rest.shape[1])
            f_rest = torch.cat((f_rest, padding), dim=1)
        elif f_rest.shape[1] > 45:
            f_rest = f_rest[:, :45]
    else:
        f_rest = torch.zeros(means.shape[0], 45, dtype=torch.float32)
    opacities = params["opacities"].detach().cpu().float()
    scales = params["scales"].detach().cpu().float()
    rotations = F.normalize(params["quats"].detach().cpu().float(), dim=-1)
    zeros = torch.zeros(means.shape[0], 3, dtype=torch.float32)

    invalid = (
        ~torch.isfinite(means).all(dim=1)
        | ~torch.isfinite(f_dc).all(dim=1)
        | ~torch.isfinite(opacities)
        | ~torch.isfinite(scales).all(dim=1)
        | ~torch.isfinite(rotations).all(dim=1)
    )
    valid = ~invalid
    means = means[valid]
    zeros = zeros[valid]
    f_dc = f_dc[valid]
    f_rest = f_rest[valid]
    opacities = opacities[valid]
    scales = scales[valid]
    rotations = rotations[valid]

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment gsplat-biren minimal multiview trainer 3DGS output",
        f"element vertex {means.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    header_lines.extend(f"property float f_rest_{i}" for i in range(f_rest.shape[1]))
    header_lines.extend(
        [
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
        ]
    )
    body = torch.cat(
        [
            means,
            zeros,
            f_dc,
            f_rest,
            opacities.unsqueeze(1),
            scales,
            rotations,
        ],
        dim=1,
    )
    array = body.numpy().astype(np.dtype(np.float32).newbyteorder("<"), copy=False)

    with path.open("wb") as file:
        file.write(("\n".join(header_lines) + "\n").encode("ascii"))
        file.write(array.tobytes())


def _save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
        file.write("\n")


def _should_save_step(step: int, total_steps: int, save_every: int) -> bool:
    interval = max(save_every, 1)
    return step == 1 or step == total_steps or step % interval == 0


def _validate_training_config(cfg: TrainConfig) -> None:
    _effective_refine_scale2d_stop_iter(cfg)
    if not 0.0 <= float(cfg.ssim_lambda) <= 1.0:
        raise ValueError(f"ssim_lambda must be in [0, 1], got {cfg.ssim_lambda}")
    if not 0.0 < float(cfg.means_lr_final_factor) <= 1.0:
        raise ValueError(
            "means_lr_final_factor must be in (0, 1], "
            f"got {cfg.means_lr_final_factor}"
        )
    if float(cfg.global_scale) <= 0.0:
        raise ValueError(f"global_scale must be positive, got {cfg.global_scale}")
    if cfg.native_alpha_threshold is not None:
        if not 0.0 <= float(cfg.native_alpha_threshold) < 1.0:
            raise ValueError(
                "native_alpha_threshold must be in [0, 1), "
                f"got {cfg.native_alpha_threshold}"
            )
    _native_tiled_backward_mode(cfg.native_tiled_backward_mode)
    if float(cfg.near_plane) <= 0.0:
        raise ValueError(f"near_plane must be positive, got {cfg.near_plane}")
    if float(cfg.far_plane) <= float(cfg.near_plane):
        raise ValueError(
            "far_plane must be greater than near_plane, "
            f"got far_plane={cfg.far_plane} near_plane={cfg.near_plane}"
        )
    if not 0.0 < float(cfg.init_opa) < 1.0:
        raise ValueError(f"init_opa must be in (0, 1), got {cfg.init_opa}")
    if float(cfg.init_scale) <= 0.0:
        raise ValueError(f"init_scale must be positive, got {cfg.init_scale}")
    for name in (
        "means_lr",
        "scales_lr",
        "opacities_lr",
        "quats_lr",
        "sh0_lr",
        "shN_lr",
        "lr_scale",
    ):
        if float(getattr(cfg, name)) <= 0.0:
            raise ValueError(f"{name} must be positive, got {getattr(cfg, name)}")
    if int(cfg.sh_degree) < 0 or int(cfg.sh_degree) > 3:
        raise ValueError(f"sh_degree must be in [0, 3], got {cfg.sh_degree}")
    if int(cfg.sh_degree_interval) <= 0:
        raise ValueError(
            f"sh_degree_interval must be positive, got {cfg.sh_degree_interval}"
        )
    for name in (
        "refine_start_iter",
        "refine_stop_iter",
        "refine_every",
        "reset_every",
        "pause_refine_after_reset",
        "refine_scale2d_stop_iter",
        "max_gaussians",
        "max_gaussians_ramp_steps",
    ):
        if int(getattr(cfg, name)) < 0:
            raise ValueError(f"{name} must be non-negative, got {getattr(cfg, name)}")
    if int(cfg.refine_every) == 0:
        raise ValueError("refine_every must be positive")
    if not 0.0 < float(cfg.prune_opa) < 1.0:
        raise ValueError(f"prune_opa must be in (0, 1), got {cfg.prune_opa}")
    for name in ("grow_scale3d", "grow_scale2d", "prune_scale3d", "prune_scale2d"):
        if float(getattr(cfg, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative, got {getattr(cfg, name)}")
    if float(cfg.exposure_lr) < 0.0:
        raise ValueError(f"exposure_lr must be non-negative, got {cfg.exposure_lr}")
    if float(cfg.exposure_l2) < 0.0:
        raise ValueError(f"exposure_l2 must be non-negative, got {cfg.exposure_l2}")


def _train_frame_index(
    step: int,
    num_frames: int,
    *,
    seed: int,
    shuffle: bool,
    stream_rank: int = 0,
    stream_world_size: int = 1,
) -> int:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    position = (step - 1) * stream_world_size + stream_rank
    if not shuffle:
        return position % num_frames

    epoch, offset = divmod(position, num_frames)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = None
    for _ in range(int(epoch) + 1):
        order = torch.randperm(num_frames, generator=generator)
    assert order is not None
    return int(order[offset].item())


def _ssim_loss(render: Tensor, target: Tensor, mask: Optional[Tensor]) -> Tensor:
    height, width = int(render.shape[0]), int(render.shape[1])
    window_size = max(1, min(11, height, width))
    if window_size > 1 and window_size % 2 == 0:
        window_size -= 1
    padding = window_size // 2

    render_nchw = render.permute(2, 0, 1).unsqueeze(0)
    target_nchw = target.permute(2, 0, 1).unsqueeze(0)
    box_kernels: Dict[int, Tuple[Tensor, Tensor]] = {}

    def pool(value: Tensor) -> Tensor:
        if value.device.type != "supa" or window_size <= 1:
            return F.avg_pool2d(value, window_size, stride=1, padding=padding)
        # BRPyTorch avg_pool2d backward is a major SUPA training bottleneck.
        # A box filter is separable, and zero-padded grouped conv matches
        # avg_pool2d's default count_include_pad=True behavior.
        channels = int(value.shape[1])
        kernels = box_kernels.get(channels)
        if kernels is None:
            scale = 1.0 / float(window_size)
            weight_x = value.new_full((channels, 1, 1, window_size), scale)
            weight_y = value.new_full((channels, 1, window_size, 1), scale)
            kernels = (weight_x.contiguous(), weight_y.contiguous())
            box_kernels[channels] = kernels
        weight_x, weight_y = kernels
        horizontal = F.conv2d(
            value,
            weight_x,
            stride=1,
            padding=(0, padding),
            groups=channels,
        )
        return F.conv2d(
            horizontal,
            weight_y,
            stride=1,
            padding=(padding, 0),
            groups=channels,
        )

    if mask is not None:
        mask_nchw = mask.to(render.device, dtype=render.dtype).unsqueeze(0).unsqueeze(0)
        local_weight = pool(mask_nchw)
        local_weight = local_weight.clamp_min(1e-6)

        def masked_pool(value: Tensor) -> Tensor:
            return pool(value * mask_nchw) / local_weight

        mu_render = masked_pool(render_nchw)
        mu_target = masked_pool(target_nchw)
        render_sq_mean = masked_pool(render_nchw * render_nchw)
        target_sq_mean = masked_pool(target_nchw * target_nchw)
        render_target_mean = masked_pool(render_nchw * target_nchw)
    else:
        mu_render = pool(render_nchw)
        mu_target = pool(target_nchw)
        render_sq_mean = pool(render_nchw * render_nchw)
        target_sq_mean = pool(target_nchw * target_nchw)
        render_target_mean = pool(render_nchw * target_nchw)

    mu_render_sq = mu_render.square()
    mu_target_sq = mu_target.square()
    mu_render_target = mu_render * mu_target

    sigma_render_sq = render_sq_mean - mu_render_sq
    sigma_target_sq = target_sq_mean - mu_target_sq
    sigma_render_target = render_target_mean - mu_render_target

    one = render_nchw.new_tensor(1.0)
    two = render_nchw.new_tensor(2.0)
    c1 = render_nchw.new_tensor(0.01**2)
    c2 = render_nchw.new_tensor(0.03**2)
    numerator = (two * mu_render_target + c1) * (two * sigma_render_target + c2)
    denominator = (mu_render_sq + mu_target_sq + c1) * (
        sigma_render_sq + sigma_target_sq + c2
    )
    ssim = numerator / denominator.clamp_min(1e-12)
    ssim = ssim.clamp(-1.0, 1.0)
    if mask is not None:
        weights = mask_nchw.expand_as(ssim)
        return one - (ssim * weights).sum() / weights.sum().clamp_min(1.0)
    return one - ssim.mean()


def _render_loss(
    render: Tensor,
    target: Tensor,
    mask_cpu: Optional[Tensor],
    ssim_lambda: float,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    mask = None
    if mask_cpu is not None and bool(mask_cpu.any().item()):
        mask = mask_cpu.to(render.device)

    if mask is not None:
        l1_loss = F.l1_loss(render[mask], target[mask])
    else:
        l1_loss = F.l1_loss(render, target)

    if ssim_lambda <= 0.0:
        return l1_loss, l1_loss, None

    if mask is not None:
        mask_float = mask.to(render.device, dtype=render.dtype).unsqueeze(-1)
        ssim_loss = _ssim_loss(render * mask_float, target * mask_float, None)
    else:
        ssim_loss = _ssim_loss(render, target, None)
    loss = l1_loss * (1.0 - ssim_lambda) + ssim_loss * ssim_lambda
    return loss, l1_loss, ssim_loss


def _active_sh_degree(step: int, cfg: TrainConfig) -> int:
    if cfg.sh_degree <= 0:
        return 0
    interval = max(int(cfg.sh_degree_interval), 1)
    # Upstream uses zero-based training steps. This trainer logs one-based
    # steps, so map step 1 to upstream step 0 before applying the schedule.
    return min(max(step - 1, 0) // interval, int(cfg.sh_degree))


def _format_train_step_message(
    *,
    prefix: str,
    step: int,
    total_steps: int,
    loss_value: float,
    l1_value: float,
    ssim_value: Optional[float],
    psnr_value: Optional[float],
    current_lrs: Dict[str, float],
    sh_degree: int,
    refine_stats: dict,
    meta: dict,
    log_step_time: bool,
    step_time: float,
) -> str:
    message = (
        f"{prefix}step {step:04d}/{total_steps:04d} "
        f"loss={loss_value:.6f} "
        f"projection={meta.get('projection_backend')} "
        f"composite={meta.get('composite_backend')} "
        f"variant={meta.get('composite_variant')}"
    )
    if ssim_value is not None:
        message += f" l1={l1_value:.6f} ssim={ssim_value:.6f}"
    if psnr_value is not None:
        message += f" psnr={psnr_value:.3f}"
    message += f" means_lr={current_lrs['means']:.3e}"
    message += f" sh_degree={sh_degree}"
    if _refinement_event(refine_stats):
        message += (
            f" gs={refine_stats['num_gaussians']}"
            f" dup={refine_stats['duplicated']}"
            f" split={refine_stats['split']}"
            f" prune={refine_stats['pruned']}"
            f" reset={int(refine_stats['opacity_reset'])}"
        )
    if log_step_time:
        message += f" time={step_time:.3f}s"
    crop_meta = meta.get("train_crop")
    if crop_meta is not None:
        message += (
            f" crop={crop_meta['width']}x{crop_meta['height']}"
            f"+{crop_meta['x']}+{crop_meta['y']}"
        )
    return message


def _train_multiview_single(cfg: TrainConfig) -> dict:
    if cfg.print_debug_info:
        _reset_debug_clock()
    _debug_log(
        cfg.print_debug_info,
        "init.begin",
        config=_config_metadata(cfg),
    )
    _validate_training_config(cfg)
    _debug_log(cfg.print_debug_info, "init.validate_config.done")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    _debug_log(cfg.print_debug_info, "init.seed.done", seed=cfg.seed)

    device = resolve_device(cfg.device)
    if device.type == "supa":
        set_accelerator_device(device)
    _debug_log(cfg.print_debug_info, "init.device.done", device=device)
    data = load_multiview_data(cfg)
    _debug_log(
        cfg.print_debug_info,
        "init.load_data.done",
        data=_debug_data_summary(data),
    )
    if not data.train_frames:
        raise ValueError("no training frames were loaded")
    cfg = _resolve_init_num_points(data, cfg, required=cfg.resume is None)
    _debug_log(
        cfg.print_debug_info,
        "init.resolve_points.done",
        init_num_points=cfg.init_num_points,
    )

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    resume_step = 0
    resume_ckpt = {}
    if cfg.resume is not None:
        params, resume_step, resume_ckpt = _load_checkpoint(
            cfg.resume, device, sh_degree=cfg.sh_degree
        )
    else:
        params = create_model(data, cfg, device)
    _debug_log(
        cfg.print_debug_info,
        "init.model.done",
        resume=cfg.resume,
        resume_step=resume_step,
        params=_debug_params_summary(params),
    )
    exposure_params = _model_exposure_from_checkpoint(
        resume_ckpt,
        data,
        device,
        bool(cfg.refine_exposure),
    )
    _debug_log(
        cfg.print_debug_info,
        "init.exposure.done",
        exposure=None
        if exposure_params is None
        else _debug_params_summary(exposure_params),
    )
    optimizer = create_optimizers(
        params,
        data.scene_scale,
        cfg,
        exposure_params=exposure_params,
    )
    _debug_log(
        cfg.print_debug_info,
        "init.optimizer.done",
        optimizer=_debug_optimizer_summary(optimizer),
    )
    if "optimizer" in resume_ckpt:
        _try_load_optimizer_state(optimizer, resume_ckpt["optimizer"], device)
        _debug_log(cfg.print_debug_info, "init.optimizer_state.done")
    refinement_state = _new_refinement_state(params)
    _debug_log(
        cfg.print_debug_info,
        "init.refinement_state.done",
        state=_debug_refinement_state_summary(refinement_state),
    )
    if cfg.resume is not None:
        log_info(
            LOGGER,
            f"resumed step {resume_step} from {cfg.resume} "
            f"num_gaussians={params['means'].shape[0]}"
        )

    losses = []
    l1_losses = []
    ssim_losses = []
    exposure_reg_losses = []
    means_lrs = []
    sh_degrees = []
    gaussian_counts = []
    refinement_events = []
    step_times = []
    phase_times = _new_phase_times()
    last_meta = {}
    last_refinement = {}
    for step in range(resume_step + 1, cfg.iterations + 1):
        step_start = time.perf_counter()
        phase_step = _new_phase_step()
        phase_start = step_start
        _debug_log(cfg.print_debug_info, "train.step.begin", step=step)
        current_lrs = _set_optimizer_lrs(optimizer, data.scene_scale, cfg, step)
        _debug_log(
            cfg.print_debug_info,
            "train.set_lrs.done",
            step=step,
            lrs=current_lrs,
        )
        sh_degree_to_use = _active_sh_degree(step, cfg)
        _debug_log(
            cfg.print_debug_info,
            "train.active_sh_degree.done",
            step=step,
            sh_degree=sh_degree_to_use,
        )
        phase_start = _mark_phase(phase_step, "schedule", phase_start)
        frame_index = _train_frame_index(
            step,
            len(data.train_frames),
            seed=cfg.seed,
            shuffle=cfg.shuffle_train,
        )
        frame = data.train_frames[frame_index]
        _debug_log(
            cfg.print_debug_info,
            "train.select_frame.done",
            step=step,
            frame_index=frame_index,
            frame=_debug_frame_summary(frame),
        )
        crop = _sample_train_crop(
            frame,
            cfg.train_crop_size,
            seed=cfg.seed,
            step=step,
        )
        _debug_log(
            cfg.print_debug_info,
            "train.sample_crop.done",
            step=step,
            crop=crop,
        )
        target_cpu, mask_cpu = _crop_image_and_mask(frame, crop)
        _debug_log(
            cfg.print_debug_info,
            "train.crop_target.done",
            step=step,
            target_cpu=target_cpu,
            mask_cpu=mask_cpu,
        )
        phase_start = _mark_phase(phase_step, "data_and_crop", phase_start)
        target = target_cpu.to(device)
        _debug_log(
            cfg.print_debug_info,
            "train.move_target.done",
            step=step,
            target=target,
        )
        phase_start = _mark_phase(phase_step, "target_to_device", phase_start)
        render, alpha, meta = _rasterize_basic(
            params,
            frame,
            device,
            cfg,
            crop=crop,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            radius_clip=cfg.radius_clip,
            sh_degree=sh_degree_to_use,
            debug_prefix="train.",
        )
        phase_start = _mark_phase(phase_step, "rasterize", phase_start)
        if cfg.random_bkgd:
            _debug_log(
                cfg.print_debug_info,
                "train.random_background.begin",
                step=step,
                render=render,
                alpha=alpha,
            )
            render = _apply_random_background(render, alpha)
            _debug_log(
                cfg.print_debug_info,
                "train.random_background.done",
                step=step,
                render=render,
            )
        del alpha
        _debug_log(cfg.print_debug_info, "train.release_alpha.done", step=step)
        frame_id = _exposure_frame_id(frame_index, is_eval=False, data=data)
        _debug_log(
            cfg.print_debug_info,
            "train.exposure_frame.done",
            step=step,
            frame_id=frame_id,
        )
        render = _apply_exposure(render, exposure_params, frame_id, clamp=False)
        _debug_log(
            cfg.print_debug_info,
            "train.apply_exposure.done",
            step=step,
            render=render,
        )
        if exposure_params is not None:
            meta["exposure_frame_id"] = frame_id
        refinement_tracked = _retain_refinement_grad(meta, cfg, step)
        _debug_log(
            cfg.print_debug_info,
            "train.retain_refinement_grad.done",
            step=step,
            tracked=refinement_tracked,
        )
        phase_start = _mark_phase(phase_step, "postprocess", phase_start)
        loss, l1_loss, ssim_loss = _render_loss(
            render, target, mask_cpu, float(cfg.ssim_lambda)
        )
        _debug_log(
            cfg.print_debug_info,
            "train.render_loss.done",
            step=step,
            loss=loss,
            l1_loss=l1_loss,
            ssim_loss=ssim_loss,
        )
        exposure_reg = _exposure_regularization(
            exposure_params,
            frame_id,
            float(cfg.exposure_l2),
        )
        _debug_log(
            cfg.print_debug_info,
            "train.exposure_regularization.done",
            step=step,
            exposure_reg=exposure_reg,
        )
        if exposure_reg is not None:
            loss = loss + exposure_reg
            _debug_log(
                cfg.print_debug_info,
                "train.add_exposure_regularization.done",
                step=step,
                loss=loss,
            )
        phase_start = _mark_phase(phase_step, "loss", phase_start)

        _optimizer_zero_grad(optimizer)
        _debug_log(cfg.print_debug_info, "train.optimizer_zero_grad.done", step=step)
        phase_start = _mark_phase(phase_step, "zero_grad", phase_start)
        loss.backward()
        _debug_log(
            cfg.print_debug_info,
            "train.backward.done",
            step=step,
            grads=_debug_grad_summary(params),
        )
        phase_start = _mark_phase(phase_step, "backward", phase_start)
        phase_step["grad_sync"] = 0.0
        if refinement_tracked:
            _update_refinement_state(refinement_state, meta)
            _debug_log(
                cfg.print_debug_info,
                "train.update_refinement_state.done",
                step=step,
                state=_debug_refinement_state_summary(refinement_state),
            )
        phase_start = _mark_phase(phase_step, "refinement_state", phase_start)
        _optimizer_step(optimizer)
        _debug_log(
            cfg.print_debug_info,
            "train.optimizer_step.done",
            step=step,
            optimizer=_debug_optimizer_summary(optimizer),
        )
        phase_start = _mark_phase(phase_step, "optimizer_step", phase_start)
        refine_stats = _apply_refinement(
            params,
            optimizer,
            refinement_state,
            cfg,
            step,
            data.scene_scale,
        )
        _debug_log(
            cfg.print_debug_info,
            "train.apply_refinement.done",
            step=step,
            refine_stats=refine_stats,
            params=_debug_params_summary(params),
        )
        phase_start = _mark_phase(phase_step, "refinement", phase_start)
        synchronize(device)
        _debug_log(cfg.print_debug_info, "train.synchronize.done", step=step)
        phase_start = _mark_phase(phase_step, "synchronize", phase_start)
        step_time = time.perf_counter() - step_start

        loss_value = float(loss.detach().cpu().item())
        l1_value = float(l1_loss.detach().cpu().item())
        ssim_value = (
            None if ssim_loss is None else float(ssim_loss.detach().cpu().item())
        )
        exposure_reg_value = (
            None if exposure_reg is None else float(exposure_reg.detach().cpu().item())
        )
        losses.append(loss_value)
        l1_losses.append(l1_value)
        ssim_losses.append(ssim_value)
        exposure_reg_losses.append(exposure_reg_value)
        means_lrs.append(float(current_lrs["means"]))
        sh_degrees.append(int(sh_degree_to_use))
        gaussian_counts.append(int(params["means"].shape[0]))
        if _refinement_event(refine_stats):
            event = {"step": step, **refine_stats}
            refinement_events.append(event)
            last_refinement = event
        else:
            last_refinement = refine_stats
        step_times.append(step_time)
        last_meta = meta
        should_save_step = _should_save_step(step, cfg.iterations, cfg.save_every)
        should_log_step = cfg.print_all_step_info or should_save_step
        should_compute_psnr = should_log_step or cfg.print_debug_info
        psnr_value = (
            _render_psnr(render, target, mask_cpu) if should_compute_psnr else None
        )
        phase_start = _mark_phase(phase_step, "metrics_cpu", phase_start)
        _debug_log(
            cfg.print_debug_info,
            "train.step_metrics.done",
            step=step,
            loss=loss_value,
            l1=l1_value,
            ssim=ssim_value,
            psnr=psnr_value,
            exposure_reg=exposure_reg_value,
            step_time=step_time,
            should_save_step=should_save_step,
            meta=_debug_meta_summary(meta),
        )
        if should_log_step:
            log_info(
                LOGGER,
                _format_train_step_message(
                    prefix="",
                    step=step,
                    total_steps=cfg.iterations,
                    loss_value=loss_value,
                    l1_value=l1_value,
                    ssim_value=ssim_value,
                    psnr_value=psnr_value,
                    current_lrs=current_lrs,
                    sh_degree=sh_degree_to_use,
                    refine_stats=refine_stats,
                    meta=meta,
                    log_step_time=cfg.log_step_time,
                    step_time=step_time,
                ),
            )
        if should_save_step:
            save_start = time.perf_counter()
            _debug_log(cfg.print_debug_info, "train.save_step.begin", step=step)
            _save_image(render, cfg.result_dir / "renders" / f"train_step_{step:04d}.png")
            _save_checkpoint(
                params,
                optimizer,
                cfg,
                step,
                cfg.result_dir / "ckpts" / f"step_{step:04d}.pt",
                exposure_params=exposure_params,
            )
            if cfg.save_ply:
                _save_ply(
                    params,
                    cfg.result_dir / "ply" / f"step_{step:04d}_points.ply",
                )
            _debug_log(cfg.print_debug_info, "train.save_step.done", step=step)
            phase_step["save_step"] = time.perf_counter() - save_start
        _append_phase_step(phase_times, phase_step)

    final_step = max(resume_step, cfg.iterations)
    eval_frames_are_eval = bool(data.eval_frames)
    eval_frames = data.eval_frames if data.eval_frames else data.train_frames
    _debug_log(
        cfg.print_debug_info,
        "eval.begin",
        final_step=final_step,
        eval_frames=len(eval_frames),
        frames_are_eval=eval_frames_are_eval,
    )
    with torch.no_grad():
        final_render, final_meta, eval_metrics = _evaluate_frames(
            params,
            eval_frames,
            device,
            cfg,
            cfg.result_dir,
            cfg.eval_max_frames,
            cfg.near_plane,
            cfg.far_plane,
            cfg.radius_clip,
            cfg.sh_degree,
            exposure_params=exposure_params,
            data=data,
            frames_are_eval=eval_frames_are_eval,
        )
    synchronize(device)
    _debug_log(
        cfg.print_debug_info,
        "eval.done",
        final_render=final_render,
        final_meta=_debug_meta_summary(final_meta),
        eval_metrics=eval_metrics,
    )
    _save_image(final_render, cfg.result_dir / "renders" / "final.png")
    _debug_log(cfg.print_debug_info, "final.save_render.done")
    _save_checkpoint(
        params,
        optimizer,
        cfg,
        final_step,
        cfg.result_dir / "ckpts" / "final.pt",
        exposure_params=exposure_params,
    )
    _debug_log(cfg.print_debug_info, "final.save_checkpoint.done", step=final_step)
    ply_path = cfg.result_dir / "ply" / "final_points.ply"
    if cfg.save_ply:
        _save_ply(params, ply_path)
        _debug_log(cfg.print_debug_info, "final.save_ply.done", path=ply_path)
    diagnostics = _training_diagnostics(
        params,
        refinement_state,
        cfg,
        data.scene_scale,
        exposure_params=exposure_params,
    )
    _debug_log(cfg.print_debug_info, "final.diagnostics.done", diagnostics=diagnostics)
    metrics = {
        "step": final_step,
        "start_step": resume_step + 1,
        "num_gaussians": int(params["means"].shape[0]),
        "config": _config_metadata(cfg),
        "train": {
            "num_steps": len(losses),
            "losses": losses,
            "last_loss": losses[-1] if losses else None,
            "l1_losses": l1_losses,
            "last_l1_loss": l1_losses[-1] if l1_losses else None,
            "ssim_losses": ssim_losses,
            "last_ssim_loss": ssim_losses[-1] if ssim_losses else None,
            "exposure_reg_losses": exposure_reg_losses,
            "last_exposure_reg_loss": (
                exposure_reg_losses[-1] if exposure_reg_losses else None
            ),
            "means_lrs": means_lrs,
            "last_means_lr": means_lrs[-1] if means_lrs else None,
            "sh_degrees": sh_degrees,
            "last_sh_degree": sh_degrees[-1] if sh_degrees else None,
            "num_gaussians": gaussian_counts,
            "last_num_gaussians": gaussian_counts[-1] if gaussian_counts else None,
            "refinement_events": refinement_events,
            "last_refinement": last_refinement,
            "step_times_s": step_times,
            "last_step_time_s": step_times[-1] if step_times else None,
            "step_time_summary_s": _float_series_summary(step_times),
            "phase_names": list(TRAIN_PHASE_NAMES),
            "phase_times_s": phase_times,
            "phase_time_summaries_s": _phase_time_summaries(phase_times),
            "last_projection_backend": last_meta.get("projection_backend"),
            "last_composite_backend": last_meta.get("composite_backend"),
            "last_composite_variant": last_meta.get("composite_variant"),
            "last_tile_count_backend": last_meta.get("tile_count_backend"),
            "last_tiled_metadata_backend": last_meta.get("tiled_metadata_backend"),
        },
        "eval": eval_metrics,
        "diagnostics": diagnostics,
        "projection_backend": _meta_value(final_meta, last_meta, "projection_backend"),
        "composite_backend": _meta_value(final_meta, last_meta, "composite_backend"),
        "composite_variant": _meta_value(final_meta, last_meta, "composite_variant"),
        "tile_count_backend": _meta_value(final_meta, last_meta, "tile_count_backend"),
        "tiled_metadata_backend": _meta_value(
            final_meta, last_meta, "tiled_metadata_backend"
        ),
    }
    metrics_path = cfg.result_dir / "metrics.json"
    _save_metrics(metrics, metrics_path)

    return {
        "losses": losses,
        "step_times_s": step_times,
        "step_time_summary_s": metrics["train"]["step_time_summary_s"],
        "phase_time_summaries_s": metrics["train"]["phase_time_summaries_s"],
        "num_gaussians": int(params["means"].shape[0]),
        "start_step": resume_step + 1,
        "step": final_step,
        "final_render": str(cfg.result_dir / "renders" / "final.png"),
        "checkpoint": str(cfg.result_dir / "ckpts" / "final.pt"),
        "ply": str(ply_path) if cfg.save_ply else None,
        "metrics": str(metrics_path),
        "projection_backend": metrics["projection_backend"],
        "composite_backend": metrics["composite_backend"],
        "composite_variant": metrics["composite_variant"],
        "tile_count_backend": metrics["tile_count_backend"],
        "tiled_metadata_backend": metrics["tiled_metadata_backend"],
        "eval_l1": eval_metrics["l1"],
        "eval_mse": eval_metrics["mse"],
        "eval_psnr": eval_metrics["psnr"],
        "diagnostics": diagnostics,
    }


def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _evaluate_frames_distributed(
    params: torch.nn.ParameterDict,
    frames: List[Frame],
    device: torch.device,
    cfg: TrainConfig,
    result_dir: Path,
    max_frames: int,
    world_rank: int,
    world_size: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    sh_degree: int,
    exposure_params: Optional[torch.nn.ParameterDict] = None,
    data: Optional[MultiviewData] = None,
    frames_are_eval: bool = True,
) -> Tuple[Tensor, dict, dict]:
    if not frames:
        raise ValueError("at least one frame is required for evaluation")

    metric_limit = max(0, min(len(frames), max_frames))
    render_limit = max(1, math.ceil(max(metric_limit, 1) / world_size))
    frame_metrics = []
    grid_rows = []
    final_render = None
    final_meta = {}
    for local_index in range(render_limit):
        global_index = local_index * world_size + world_rank
        frame = frames[global_index % len(frames)]
        render, _, meta = _rasterize_basic(
            params,
            frame,
            device,
            cfg,
            distributed=True,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            sh_degree=sh_degree,
            debug_prefix=f"rank {world_rank} eval.",
        )
        if exposure_params is not None and data is not None:
            frame_id = _exposure_frame_id(
                global_index % len(frames),
                is_eval=frames_are_eval,
                data=data,
            )
            render = _apply_exposure(render, exposure_params, frame_id, clamp=False)
            meta["exposure_frame_id"] = frame_id
        if final_render is None:
            final_render = render
            final_meta = meta
        if global_index >= metric_limit:
            continue

        target = frame.image.to(device)
        l1, mse, pixel_count = _image_metrics(render, target, frame.mask)
        grid_rows.append((target, render))
        frame_metrics.append(
            {
                "name": frame.name,
                "global_eval_index": global_index,
                "mask_pixels": pixel_count,
                "l1": l1,
                "mse": mse,
                "psnr": _psnr_from_mse(mse),
                "projection_backend": meta.get("projection_backend"),
                "composite_backend": meta.get("composite_backend"),
            }
        )
        _save_image(
            render,
            result_dir / "renders" / f"eval_{global_index:04d}_{frame.name}.png",
        )

    if frame_metrics:
        l1 = float(sum(item["l1"] for item in frame_metrics) / len(frame_metrics))
        mse = float(sum(item["mse"] for item in frame_metrics) / len(frame_metrics))
    else:
        l1 = None
        mse = None
    grid_path = _save_eval_grid(grid_rows, result_dir / "renders" / "eval_grid.png")
    metrics = {
        "num_frames": len(frame_metrics),
        "l1": l1,
        "mse": mse,
        "psnr": _psnr_from_mse(mse) if mse is not None else None,
        "grid": grid_path,
        "frames": frame_metrics,
    }
    assert final_render is not None
    return final_render, final_meta, metrics


def _train_multiview_distributed_worker(
    world_rank: int,
    world_size: int,
    cfg: TrainConfig,
    master_addr: str,
    master_port: int,
) -> None:
    if cfg.print_debug_info:
        _reset_debug_clock()
    debug_prefix = f"rank {world_rank} "
    _debug_log(
        cfg.print_debug_info,
        "init.begin",
        prefix=debug_prefix,
        config=_config_metadata(cfg),
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
    )
    _validate_training_config(cfg)
    _debug_log(cfg.print_debug_info, "init.validate_config.done", prefix=debug_prefix)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    local_rank = world_rank
    set_accelerator_device(f"supa:{local_rank}")
    device = resolve_device(f"supa:{local_rank}")
    _debug_log(
        cfg.print_debug_info,
        "init.device.done",
        prefix=debug_prefix,
        device=device,
        local_rank=local_rank,
    )
    dist.init_process_group(
        distributed_backend(device),
        world_size=world_size,
        rank=world_rank,
    )
    _debug_log(
        cfg.print_debug_info,
        "init.process_group.done",
        prefix=debug_prefix,
        backend=distributed_backend(device),
    )

    try:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        _debug_log(cfg.print_debug_info, "init.seed.done", prefix=debug_prefix, seed=cfg.seed)

        data = load_multiview_data(cfg)
        _debug_log(
            cfg.print_debug_info,
            "init.load_data.done",
            prefix=debug_prefix,
            data=_debug_data_summary(data),
        )
        if not data.train_frames:
            raise ValueError("no training frames were loaded")
        cfg = _resolve_init_num_points(data, cfg, required=cfg.resume is None)
        _debug_log(
            cfg.print_debug_info,
            "init.resolve_points.done",
            prefix=debug_prefix,
            init_num_points=cfg.init_num_points,
        )

        resume_step = 0
        resume_ckpt = {}
        if cfg.resume is not None:
            resume_ckpt = _load_distributed_resume_checkpoint(cfg.resume)
            global_num_gaussians = int(resume_ckpt["splats"]["means"].shape[0])
            shard_start, shard_stop = _checkpoint_rank_range(
                resume_ckpt,
                world_rank,
                world_size,
                global_num_gaussians,
            )
            params = _shard_splats(
                resume_ckpt["splats"],
                device,
                shard_start,
                shard_stop,
                sh_degree=cfg.sh_degree,
            )
            resume_step = int(resume_ckpt.get("step", 0))
        else:
            params, global_num_gaussians = create_model_shard(
                data, cfg, device, world_rank, world_size
            )
            shard_start, shard_stop = _rank_range(
                global_num_gaussians, world_rank, world_size
            )
        _debug_log(
            cfg.print_debug_info,
            "init.model.done",
            prefix=debug_prefix,
            resume=cfg.resume,
            resume_step=resume_step,
            global_num_gaussians=global_num_gaussians,
            shard_start=shard_start,
            shard_stop=shard_stop,
            params=_debug_params_summary(params),
        )
        random.seed(cfg.seed + world_rank)
        np.random.seed(cfg.seed + world_rank)
        torch.manual_seed(cfg.seed + world_rank)
        _debug_log(
            cfg.print_debug_info,
            "init.rank_seed.done",
            prefix=debug_prefix,
            seed=cfg.seed + world_rank,
        )

        rank_dir = cfg.result_dir / f"rank_{world_rank:04d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        exposure_params = _model_exposure_from_checkpoint(
            resume_ckpt,
            data,
            device,
            bool(cfg.refine_exposure),
        )
        _debug_log(
            cfg.print_debug_info,
            "init.exposure.done",
            prefix=debug_prefix,
            exposure=None
            if exposure_params is None
            else _debug_params_summary(exposure_params),
        )
        optimizer = create_optimizers(
            params,
            data.scene_scale,
            cfg,
            exposure_params=exposure_params,
        )
        _debug_log(
            cfg.print_debug_info,
            "init.optimizer.done",
            prefix=debug_prefix,
            optimizer=_debug_optimizer_summary(optimizer),
        )
        if "optimizer" in resume_ckpt:
            _try_load_optimizer_state(
                optimizer,
                _slice_optimizer_state(
                    resume_ckpt["optimizer"],
                    global_num_gaussians,
                    shard_start,
                    shard_stop,
                ),
                device,
            )
            _debug_log(
                cfg.print_debug_info,
                "init.optimizer_state.done",
                prefix=debug_prefix,
            )
        refinement_state = _new_refinement_state(params)
        _debug_log(
            cfg.print_debug_info,
            "init.refinement_state.done",
            prefix=debug_prefix,
            state=_debug_refinement_state_summary(refinement_state),
        )
        if cfg.resume is not None:
            log_info(
                LOGGER,
                f"rank {world_rank} resumed step {resume_step} from {cfg.resume} "
                f"local_num_gaussians={params['means'].shape[0]} "
                f"global_num_gaussians={global_num_gaussians}"
            )

        losses = []
        l1_losses = []
        ssim_losses = []
        exposure_reg_losses = []
        means_lrs = []
        sh_degrees = []
        gaussian_counts = []
        refinement_events = []
        step_times = []
        phase_times = _new_phase_times()
        last_meta = {}
        last_refinement = {}
        for step in range(resume_step + 1, cfg.iterations + 1):
            step_start = time.perf_counter()
            phase_step = _new_phase_step()
            phase_start = step_start
            _debug_log(
                cfg.print_debug_info,
                "train.step.begin",
                prefix=debug_prefix,
                step=step,
            )
            current_lrs = _set_optimizer_lrs(optimizer, data.scene_scale, cfg, step)
            _debug_log(
                cfg.print_debug_info,
                "train.set_lrs.done",
                prefix=debug_prefix,
                step=step,
                lrs=current_lrs,
            )
            sh_degree_to_use = _active_sh_degree(step, cfg)
            _debug_log(
                cfg.print_debug_info,
                "train.active_sh_degree.done",
                prefix=debug_prefix,
                step=step,
                sh_degree=sh_degree_to_use,
            )
            phase_start = _mark_phase(phase_step, "schedule", phase_start)
            frame_index = _train_frame_index(
                step,
                len(data.train_frames),
                seed=cfg.seed,
                shuffle=cfg.shuffle_train,
                stream_rank=world_rank,
                stream_world_size=world_size,
            )
            frame = data.train_frames[frame_index]
            _debug_log(
                cfg.print_debug_info,
                "train.select_frame.done",
                prefix=debug_prefix,
                step=step,
                frame_index=frame_index,
                frame=_debug_frame_summary(frame),
            )
            crop = _sample_train_crop(
                frame,
                cfg.train_crop_size,
                seed=cfg.seed,
                step=step,
                stream_rank=world_rank,
                stream_world_size=world_size,
            )
            _debug_log(
                cfg.print_debug_info,
                "train.sample_crop.done",
                prefix=debug_prefix,
                step=step,
                crop=crop,
            )
            target_cpu, mask_cpu = _crop_image_and_mask(frame, crop)
            _debug_log(
                cfg.print_debug_info,
                "train.crop_target.done",
                prefix=debug_prefix,
                step=step,
                target_cpu=target_cpu,
                mask_cpu=mask_cpu,
            )
            phase_start = _mark_phase(phase_step, "data_and_crop", phase_start)
            target = target_cpu.to(device)
            _debug_log(
                cfg.print_debug_info,
                "train.move_target.done",
                prefix=debug_prefix,
                step=step,
                target=target,
            )
            phase_start = _mark_phase(phase_step, "target_to_device", phase_start)
            render, alpha, meta = _rasterize_basic(
                params,
                frame,
                device,
                cfg,
                distributed=True,
                crop=crop,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                radius_clip=cfg.radius_clip,
                sh_degree=sh_degree_to_use,
                debug_prefix=f"{debug_prefix}train.",
            )
            phase_start = _mark_phase(phase_step, "rasterize", phase_start)
            if cfg.random_bkgd:
                _debug_log(
                    cfg.print_debug_info,
                    "train.random_background.begin",
                    prefix=debug_prefix,
                    step=step,
                    render=render,
                    alpha=alpha,
                )
                render = _apply_random_background(render, alpha)
                _debug_log(
                    cfg.print_debug_info,
                    "train.random_background.done",
                    prefix=debug_prefix,
                    step=step,
                    render=render,
                )
            del alpha
            _debug_log(
                cfg.print_debug_info,
                "train.release_alpha.done",
                prefix=debug_prefix,
                step=step,
            )
            frame_id = _exposure_frame_id(frame_index, is_eval=False, data=data)
            _debug_log(
                cfg.print_debug_info,
                "train.exposure_frame.done",
                prefix=debug_prefix,
                step=step,
                frame_id=frame_id,
            )
            render = _apply_exposure(render, exposure_params, frame_id)
            _debug_log(
                cfg.print_debug_info,
                "train.apply_exposure.done",
                prefix=debug_prefix,
                step=step,
                render=render,
            )
            if exposure_params is not None:
                meta["exposure_frame_id"] = frame_id
            refinement_tracked = _retain_refinement_grad(meta, cfg, step)
            _debug_log(
                cfg.print_debug_info,
                "train.retain_refinement_grad.done",
                prefix=debug_prefix,
                step=step,
                tracked=refinement_tracked,
            )
            phase_start = _mark_phase(phase_step, "postprocess", phase_start)
            local_loss, l1_loss, ssim_loss = _render_loss(
                render, target, mask_cpu, float(cfg.ssim_lambda)
            )
            _debug_log(
                cfg.print_debug_info,
                "train.render_loss.done",
                prefix=debug_prefix,
                step=step,
                local_loss=local_loss,
                l1_loss=l1_loss,
                ssim_loss=ssim_loss,
            )
            exposure_reg = _exposure_regularization(
                exposure_params,
                frame_id,
                float(cfg.exposure_l2),
            )
            _debug_log(
                cfg.print_debug_info,
                "train.exposure_regularization.done",
                prefix=debug_prefix,
                step=step,
                exposure_reg=exposure_reg,
            )
            if exposure_reg is not None:
                local_loss = local_loss + exposure_reg
                _debug_log(
                    cfg.print_debug_info,
                    "train.add_exposure_regularization.done",
                    prefix=debug_prefix,
                    step=step,
                    local_loss=local_loss,
                )
            loss = local_loss / world_size
            _debug_log(
                cfg.print_debug_info,
                "train.scale_distributed_loss.done",
                prefix=debug_prefix,
                step=step,
                loss=loss,
            )
            phase_start = _mark_phase(phase_step, "loss", phase_start)

            _optimizer_zero_grad(optimizer)
            _debug_log(
                cfg.print_debug_info,
                "train.optimizer_zero_grad.done",
                prefix=debug_prefix,
                step=step,
            )
            phase_start = _mark_phase(phase_step, "zero_grad", phase_start)
            loss.backward()
            _debug_log(
                cfg.print_debug_info,
                "train.backward.done",
                prefix=debug_prefix,
                step=step,
                grads=_debug_grad_summary(params),
            )
            phase_start = _mark_phase(phase_step, "backward", phase_start)
            _all_reduce_exposure_grads(exposure_params, world_size)
            _debug_log(
                cfg.print_debug_info,
                "train.all_reduce_exposure_grads.done",
                prefix=debug_prefix,
                step=step,
            )
            phase_start = _mark_phase(phase_step, "grad_sync", phase_start)
            if refinement_tracked:
                _update_refinement_state(refinement_state, meta)
                _debug_log(
                    cfg.print_debug_info,
                    "train.update_refinement_state.done",
                    prefix=debug_prefix,
                    step=step,
                    state=_debug_refinement_state_summary(refinement_state),
                )
            phase_start = _mark_phase(phase_step, "refinement_state", phase_start)
            _optimizer_step(optimizer)
            _debug_log(
                cfg.print_debug_info,
                "train.optimizer_step.done",
                prefix=debug_prefix,
                step=step,
                optimizer=_debug_optimizer_summary(optimizer),
            )
            phase_start = _mark_phase(phase_step, "optimizer_step", phase_start)
            refine_stats = _apply_refinement(
                params,
                optimizer,
                refinement_state,
                cfg,
                step,
                data.scene_scale,
            )
            _debug_log(
                cfg.print_debug_info,
                "train.apply_refinement.done",
                prefix=debug_prefix,
                step=step,
                refine_stats=refine_stats,
                params=_debug_params_summary(params),
            )
            phase_start = _mark_phase(phase_step, "refinement", phase_start)
            synchronize(device)
            _debug_log(
                cfg.print_debug_info,
                "train.synchronize.done",
                prefix=debug_prefix,
                step=step,
            )
            phase_start = _mark_phase(phase_step, "synchronize", phase_start)
            step_time = time.perf_counter() - step_start

            loss_value = float(local_loss.detach().cpu().item())
            l1_value = float(l1_loss.detach().cpu().item())
            ssim_value = (
                None if ssim_loss is None else float(ssim_loss.detach().cpu().item())
            )
            exposure_reg_value = (
                None if exposure_reg is None else float(exposure_reg.detach().cpu().item())
            )
            losses.append(loss_value)
            l1_losses.append(l1_value)
            ssim_losses.append(ssim_value)
            exposure_reg_losses.append(exposure_reg_value)
            means_lrs.append(float(current_lrs["means"]))
            sh_degrees.append(int(sh_degree_to_use))
            gaussian_counts.append(int(params["means"].shape[0]))
            if _refinement_event(refine_stats):
                event = {"step": step, **refine_stats}
                refinement_events.append(event)
                last_refinement = event
            else:
                last_refinement = refine_stats
            step_times.append(step_time)
            last_meta = meta
            should_save_step = _should_save_step(step, cfg.iterations, cfg.save_every)
            should_log_step = cfg.print_all_step_info or should_save_step
            should_compute_psnr = should_log_step or cfg.print_debug_info
            psnr_value = (
                _render_psnr(render, target, mask_cpu) if should_compute_psnr else None
            )
            phase_start = _mark_phase(phase_step, "metrics_cpu", phase_start)
            _debug_log(
                cfg.print_debug_info,
                "train.step_metrics.done",
                prefix=debug_prefix,
                step=step,
                loss=loss_value,
                l1=l1_value,
                ssim=ssim_value,
                psnr=psnr_value,
                exposure_reg=exposure_reg_value,
                step_time=step_time,
                should_save_step=should_save_step,
                meta=_debug_meta_summary(meta),
            )
            if should_log_step:
                log_info(
                    LOGGER,
                    _format_train_step_message(
                        prefix=f"rank {world_rank} ",
                        step=step,
                        total_steps=cfg.iterations,
                        loss_value=loss_value,
                        l1_value=l1_value,
                        ssim_value=ssim_value,
                        psnr_value=psnr_value,
                        current_lrs=current_lrs,
                        sh_degree=sh_degree_to_use,
                        refine_stats=refine_stats,
                        meta=meta,
                        log_step_time=cfg.log_step_time,
                        step_time=step_time,
                    ),
                )
            if should_save_step:
                save_start = time.perf_counter()
                _debug_log(
                    cfg.print_debug_info,
                    "train.save_step.begin",
                    prefix=debug_prefix,
                    step=step,
                )
                _save_image(render, rank_dir / "renders" / f"train_step_{step:04d}.png")
                _save_checkpoint(
                    params,
                    optimizer,
                    cfg,
                    step,
                    rank_dir / "ckpts" / f"step_{step:04d}.pt",
                    exposure_params=exposure_params,
                )
                if cfg.save_ply:
                    _save_ply(
                        params,
                        rank_dir / "ply" / f"step_{step:04d}_points.ply",
                    )
                dist.barrier()
                _debug_log(
                    cfg.print_debug_info,
                    "train.save_step.rank_barrier.done",
                    prefix=debug_prefix,
                    step=step,
                )
                if world_rank == 0:
                    checkpoint_name = f"step_{step:04d}.pt"
                    splats = _save_global_distributed_checkpoint(
                        cfg,
                        step,
                        world_size,
                        checkpoint_name,
                        cfg.result_dir / "ckpts" / checkpoint_name,
                    )
                    if cfg.save_ply:
                        _save_global_distributed_ply(
                            splats,
                            cfg.result_dir / "ply" / f"step_{step:04d}_points.ply",
                            cfg.sh_degree,
                        )
                    _save_distributed_train_step_render(cfg, step, world_size)
                    _debug_log(
                        cfg.print_debug_info,
                        "train.save_step.global_done",
                        prefix=debug_prefix,
                        step=step,
                        checkpoint=checkpoint_name,
                    )
                dist.barrier()
                _debug_log(
                    cfg.print_debug_info,
                    "train.save_step.done",
                    prefix=debug_prefix,
                    step=step,
                )
                phase_step["save_step"] = time.perf_counter() - save_start
            _append_phase_step(phase_times, phase_step)

        final_step = max(resume_step, cfg.iterations)
        eval_frames_are_eval = bool(data.eval_frames)
        eval_frames = data.eval_frames if data.eval_frames else data.train_frames
        _debug_log(
            cfg.print_debug_info,
            "eval.begin",
            prefix=debug_prefix,
            final_step=final_step,
            eval_frames=len(eval_frames),
            frames_are_eval=eval_frames_are_eval,
        )
        with torch.no_grad():
            final_render, final_meta, eval_metrics = _evaluate_frames_distributed(
                params,
                eval_frames,
                device,
                cfg,
                rank_dir,
                cfg.eval_max_frames,
                world_rank,
                world_size,
                cfg.near_plane,
                cfg.far_plane,
                cfg.radius_clip,
                cfg.sh_degree,
                exposure_params=exposure_params,
                data=data,
                frames_are_eval=eval_frames_are_eval,
            )
        synchronize(device)
        _debug_log(
            cfg.print_debug_info,
            "eval.done",
            prefix=debug_prefix,
            final_render=final_render,
            final_meta=_debug_meta_summary(final_meta),
            eval_metrics=eval_metrics,
        )
        _save_image(final_render, rank_dir / "renders" / "final.png")
        _debug_log(
            cfg.print_debug_info,
            "final.save_render.done",
            prefix=debug_prefix,
        )
        _save_checkpoint(
            params,
            optimizer,
            cfg,
            final_step,
            rank_dir / "ckpts" / "final.pt",
            exposure_params=exposure_params,
        )
        _debug_log(
            cfg.print_debug_info,
            "final.save_checkpoint.done",
            prefix=debug_prefix,
            step=final_step,
        )
        ply_path = rank_dir / "ply" / "final_points.ply"
        if cfg.save_ply:
            _save_ply(params, ply_path)
            _debug_log(
                cfg.print_debug_info,
                "final.save_ply.done",
                prefix=debug_prefix,
                path=ply_path,
            )
        diagnostics = _training_diagnostics(
            params,
            refinement_state,
            cfg,
            data.scene_scale,
            exposure_params=exposure_params,
        )
        _debug_log(
            cfg.print_debug_info,
            "final.diagnostics.done",
            prefix=debug_prefix,
            diagnostics=diagnostics,
        )
        dist.barrier()
        _debug_log(
            cfg.print_debug_info,
            "final.rank_barrier.done",
            prefix=debug_prefix,
        )
        if world_rank == 0:
            splats = _save_global_distributed_checkpoint(
                cfg,
                final_step,
                world_size,
                "final.pt",
                cfg.result_dir / "ckpts" / "final.pt",
            )
            if cfg.save_ply:
                _save_global_distributed_ply(
                    splats,
                    cfg.result_dir / "ply" / "final_points.ply",
                    cfg.sh_degree,
                )
            _copy_if_exists(
                rank_dir / "renders" / "final.png",
                cfg.result_dir / "renders" / "final.png",
            )
            _copy_if_exists(
                rank_dir / "renders" / "eval_grid.png",
                cfg.result_dir / "renders" / "eval_grid.png",
            )
            _debug_log(
                cfg.print_debug_info,
                "final.global_outputs.done",
                prefix=debug_prefix,
                checkpoint="final.pt",
            )
        dist.barrier()
        _debug_log(
            cfg.print_debug_info,
            "final.done",
            prefix=debug_prefix,
            step=final_step,
        )

        metrics = {
            "step": final_step,
            "start_step": resume_step + 1,
            "distributed": True,
            "distributed_backend": final_meta.get(
                "distributed_backend", last_meta.get("distributed_backend")
            ),
            "world_rank": world_rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "num_gaussians": global_num_gaussians,
            "local_num_gaussians": int(params["means"].shape[0]),
            "shard": {"start": shard_start, "stop": shard_stop},
            "config": _config_metadata(cfg),
            "train": {
                "num_steps": len(losses),
                "losses": losses,
                "last_loss": losses[-1] if losses else None,
                "l1_losses": l1_losses,
                "last_l1_loss": l1_losses[-1] if l1_losses else None,
                "ssim_losses": ssim_losses,
                "last_ssim_loss": ssim_losses[-1] if ssim_losses else None,
                "exposure_reg_losses": exposure_reg_losses,
                "last_exposure_reg_loss": (
                    exposure_reg_losses[-1] if exposure_reg_losses else None
                ),
                "means_lrs": means_lrs,
                "last_means_lr": means_lrs[-1] if means_lrs else None,
                "sh_degrees": sh_degrees,
                "last_sh_degree": sh_degrees[-1] if sh_degrees else None,
                "num_gaussians": gaussian_counts,
                "last_num_gaussians": gaussian_counts[-1] if gaussian_counts else None,
                "refinement_events": refinement_events,
                "last_refinement": last_refinement,
                "step_times_s": step_times,
                "last_step_time_s": step_times[-1] if step_times else None,
                "step_time_summary_s": _float_series_summary(step_times),
                "phase_names": list(TRAIN_PHASE_NAMES),
                "phase_times_s": phase_times,
                "phase_time_summaries_s": _phase_time_summaries(phase_times),
                "last_projection_backend": last_meta.get("projection_backend"),
                "last_composite_backend": last_meta.get("composite_backend"),
                "last_composite_variant": last_meta.get("composite_variant"),
                "last_tile_count_backend": last_meta.get("tile_count_backend"),
                "last_tiled_metadata_backend": last_meta.get("tiled_metadata_backend"),
            },
            "eval": eval_metrics,
            "diagnostics": diagnostics,
            "projection_backend": _meta_value(final_meta, last_meta, "projection_backend"),
            "composite_backend": _meta_value(final_meta, last_meta, "composite_backend"),
            "composite_variant": _meta_value(final_meta, last_meta, "composite_variant"),
            "tile_count_backend": _meta_value(final_meta, last_meta, "tile_count_backend"),
            "tiled_metadata_backend": _meta_value(
                final_meta, last_meta, "tiled_metadata_backend"
            ),
            "final_render": str(rank_dir / "renders" / "final.png"),
            "checkpoint": str(rank_dir / "ckpts" / "final.pt"),
            "ply": str(ply_path) if cfg.save_ply else None,
        }
        _save_metrics(metrics, rank_dir / "metrics.json")
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _float_series_summary(values: List[Optional[float]]) -> dict:
    finite = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            finite.append(number)
    summary = {
        "count": len(finite),
        "min": None,
        "mean": None,
        "median": None,
        "max": None,
    }
    if not finite:
        return summary

    ordered = sorted(finite)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = 0.5 * (ordered[mid - 1] + ordered[mid])
    summary.update(
        {
            "min": float(ordered[0]),
            "mean": float(sum(ordered) / len(ordered)),
            "median": float(median),
            "max": float(ordered[-1]),
        }
    )
    return summary


def _new_phase_step() -> Dict[str, Optional[float]]:
    return {name: None for name in TRAIN_PHASE_NAMES}


def _new_phase_times() -> Dict[str, List[Optional[float]]]:
    return {name: [] for name in TRAIN_PHASE_NAMES}


def _mark_phase(
    phase_step: Dict[str, Optional[float]],
    name: str,
    started_at: float,
) -> float:
    now = time.perf_counter()
    phase_step[name] = now - started_at
    return now


def _append_phase_step(
    phase_times: Dict[str, List[Optional[float]]],
    phase_step: Dict[str, Optional[float]],
) -> None:
    for name in TRAIN_PHASE_NAMES:
        phase_times.setdefault(name, []).append(phase_step.get(name))


def _phase_time_summaries(
    phase_times: Dict[str, List[Optional[float]]],
) -> Dict[str, dict]:
    return {
        name: _float_series_summary(phase_times.get(name, []))
        for name in TRAIN_PHASE_NAMES
    }


def _aggregate_float_series(
    rank_metrics: List[dict],
    key: str,
    max_steps: int,
    *,
    skip_none: bool = False,
) -> List[Optional[float]]:
    values = []
    for index in range(max_steps):
        step_values = []
        for item in rank_metrics:
            series = item["train"].get(key, [])
            if index >= len(series):
                continue
            value = series[index]
            if value is None and skip_none:
                continue
            if value is not None:
                step_values.append(float(value))
        values.append(float(sum(step_values) / len(step_values)) if step_values else None)
    return values


def _aggregate_phase_times(
    rank_metrics: List[dict],
    max_steps: int,
) -> Dict[str, List[Optional[float]]]:
    phase_times: Dict[str, List[Optional[float]]] = {}
    for name in TRAIN_PHASE_NAMES:
        values: List[Optional[float]] = []
        for index in range(max_steps):
            step_values = []
            for item in rank_metrics:
                series = item["train"].get("phase_times_s", {}).get(name, [])
                if index < len(series) and series[index] is not None:
                    step_values.append(float(series[index]))
            values.append(
                float(sum(step_values) / len(step_values)) if step_values else None
            )
        phase_times[name] = values
    return phase_times


def _aggregate_distributed_metrics(cfg: TrainConfig, rank_metrics: List[dict]) -> dict:
    rank_metrics = sorted(rank_metrics, key=lambda item: int(item["world_rank"]))
    final_step = max((int(item["step"]) for item in rank_metrics), default=cfg.iterations)
    start_step = min((int(item["start_step"]) for item in rank_metrics), default=1)
    max_steps = max((len(item["train"]["losses"]) for item in rank_metrics), default=0)
    losses = _aggregate_float_series(rank_metrics, "losses", max_steps)
    l1_losses = _aggregate_float_series(rank_metrics, "l1_losses", max_steps)
    ssim_losses = _aggregate_float_series(
        rank_metrics, "ssim_losses", max_steps, skip_none=True
    )
    step_times = _aggregate_float_series(rank_metrics, "step_times_s", max_steps)
    phase_times = _aggregate_phase_times(rank_metrics, max_steps)
    means_lrs = _aggregate_float_series(rank_metrics, "means_lrs", max_steps)
    sh_degrees = [
        int(round(value)) if value is not None else None
        for value in _aggregate_float_series(rank_metrics, "sh_degrees", max_steps)
    ]
    gaussian_counts = []
    for index in range(max_steps):
        total = 0
        found = False
        for item in rank_metrics:
            series = item["train"].get("num_gaussians", [])
            if index < len(series):
                total += int(series[index])
                found = True
        gaussian_counts.append(total if found else None)
    refinement_events = []
    for item in rank_metrics:
        for event in item["train"].get("refinement_events", []):
            refinement_events.append(
                {
                    **event,
                    "world_rank": int(item["world_rank"]),
                }
            )
    refinement_events.sort(
        key=lambda item: (int(item["step"]), int(item["world_rank"]))
    )

    frame_metrics = []
    for item in rank_metrics:
        for frame_metric in item["eval"]["frames"]:
            frame_metrics.append(
                {
                    **frame_metric,
                    "world_rank": int(item["world_rank"]),
                }
            )
    frame_metrics.sort(key=lambda item: int(item["global_eval_index"]))
    l1 = _mean_or_none([float(item["l1"]) for item in frame_metrics])
    mse = _mean_or_none([float(item["mse"]) for item in frame_metrics])
    rank_summaries = []
    shard_start = 0
    for item in rank_metrics:
        local_count = int(item["local_num_gaussians"])
        shard_stop = shard_start + local_count
        rank_summaries.append(
            {
                "world_rank": int(item["world_rank"]),
                "local_rank": int(item["local_rank"]),
                "local_num_gaussians": local_count,
                "shard": {"start": shard_start, "stop": shard_stop},
                "metrics": str(
                    cfg.result_dir
                    / f"rank_{int(item['world_rank']):04d}"
                    / "metrics.json"
                ),
                "final_render": item["final_render"],
                "checkpoint": item["checkpoint"],
                "ply": item["ply"],
                "composite_variant": item.get("composite_variant"),
                "tile_count_backend": item.get("tile_count_backend"),
                "tiled_metadata_backend": item.get("tiled_metadata_backend"),
                "diagnostics": item.get("diagnostics"),
            }
        )
        shard_start = shard_stop
    projection_backend = next(
        (item.get("projection_backend") for item in rank_metrics if item.get("projection_backend")),
        None,
    )
    composite_backend = next(
        (item.get("composite_backend") for item in rank_metrics if item.get("composite_backend")),
        None,
    )
    composite_variant = next(
        (
            item.get("composite_variant")
            for item in rank_metrics
            if item.get("composite_variant")
        ),
        None,
    )
    tile_count_backend = next(
        (
            item.get("tile_count_backend")
            for item in rank_metrics
            if item.get("tile_count_backend")
        ),
        None,
    )
    tiled_metadata_backend = next(
        (
            item.get("tiled_metadata_backend")
            for item in rank_metrics
            if item.get("tiled_metadata_backend")
        ),
        None,
    )
    distributed_backend_name = next(
        (
            item.get("distributed_backend")
            for item in rank_metrics
            if item.get("distributed_backend")
        ),
        None,
    )
    return {
        "step": final_step,
        "start_step": start_step,
        "distributed": True,
        "distributed_backend": distributed_backend_name,
        "world_size": len(rank_metrics),
        "num_gaussians": sum(item["local_num_gaussians"] for item in rank_summaries),
        "config": _resolved_config_metadata(cfg, rank_metrics),
        "train": {
            "num_steps": max_steps,
            "losses": losses,
            "last_loss": losses[-1] if losses else None,
            "l1_losses": l1_losses,
            "last_l1_loss": l1_losses[-1] if l1_losses else None,
            "ssim_losses": ssim_losses,
            "last_ssim_loss": ssim_losses[-1] if ssim_losses else None,
            "means_lrs": means_lrs,
            "last_means_lr": means_lrs[-1] if means_lrs else None,
            "sh_degrees": sh_degrees,
            "last_sh_degree": sh_degrees[-1] if sh_degrees else None,
            "num_gaussians": gaussian_counts,
            "last_num_gaussians": gaussian_counts[-1] if gaussian_counts else None,
            "refinement_events": refinement_events,
            "step_times_s": step_times,
            "last_step_time_s": step_times[-1] if step_times else None,
            "step_time_summary_s": _float_series_summary(step_times),
            "phase_names": list(TRAIN_PHASE_NAMES),
            "phase_times_s": phase_times,
            "phase_time_summaries_s": _phase_time_summaries(phase_times),
            "rank_losses": [item["train"]["losses"] for item in rank_metrics],
            "rank_l1_losses": [
                item["train"].get("l1_losses", []) for item in rank_metrics
            ],
            "rank_ssim_losses": [
                item["train"].get("ssim_losses", []) for item in rank_metrics
            ],
            "rank_means_lrs": [
                item["train"].get("means_lrs", []) for item in rank_metrics
            ],
            "rank_sh_degrees": [
                item["train"].get("sh_degrees", []) for item in rank_metrics
            ],
            "rank_phase_time_summaries_s": [
                item["train"].get("phase_time_summaries_s", {})
                for item in rank_metrics
            ],
            "rank_last_composite_variants": [
                item["train"].get("last_composite_variant") for item in rank_metrics
            ],
            "rank_last_tiled_metadata_backends": [
                item["train"].get("last_tiled_metadata_backend") for item in rank_metrics
            ],
        },
        "eval": {
            "num_frames": len(frame_metrics),
            "l1": l1,
            "mse": mse,
            "psnr": _psnr_from_mse(mse) if mse is not None else None,
            "grid": str(cfg.result_dir / "renders" / "eval_grid.png")
            if (cfg.result_dir / "renders" / "eval_grid.png").is_file()
            else (rank_metrics[0]["eval"]["grid"] if rank_metrics else None),
            "frames": frame_metrics,
        },
        "diagnostics": {
            "ranks": [
                {
                    "world_rank": int(item["world_rank"]),
                    "diagnostics": item.get("diagnostics"),
                }
                for item in rank_metrics
            ]
        },
        "ranks": rank_summaries,
        "final_render": str(cfg.result_dir / "renders" / "final.png"),
        "checkpoint": str(cfg.result_dir / "ckpts" / "final.pt"),
        "ply": str(cfg.result_dir / "ply" / "final_points.ply") if cfg.save_ply else None,
        "projection_backend": projection_backend,
        "composite_backend": composite_backend,
        "composite_variant": composite_variant,
        "tile_count_backend": tile_count_backend,
        "tiled_metadata_backend": tiled_metadata_backend,
    }


def _save_distributed_eval_artifacts(cfg: TrainConfig, rank_metrics: List[dict]) -> Optional[str]:
    frame_records = []
    for item in rank_metrics:
        world_rank = int(item["world_rank"])
        for frame_metric in item["eval"]["frames"]:
            frame_records.append(
                {
                    "world_rank": world_rank,
                    "global_eval_index": int(frame_metric["global_eval_index"]),
                    "name": str(frame_metric["name"]),
                }
            )
    if not frame_records:
        return None

    data = load_multiview_data(cfg)
    eval_frames = data.eval_frames if data.eval_frames else data.train_frames
    frame_records.sort(key=lambda item: item["global_eval_index"])

    rows = []
    for record in frame_records:
        source = (
            cfg.result_dir
            / f"rank_{record['world_rank']:04d}"
            / "renders"
            / f"eval_{record['global_eval_index']:04d}_{record['name']}.png"
        )
        target = (
            cfg.result_dir
            / "renders"
            / f"eval_{record['global_eval_index']:04d}_{record['name']}.png"
        )
        _copy_if_exists(source, target)
        if not target.is_file():
            raise FileNotFoundError(f"missing distributed eval render: {source}")

        frame = eval_frames[record["global_eval_index"] % len(eval_frames)]
        rows.append((frame.image, _load_saved_image(target)))

    return _save_eval_grid(rows, cfg.result_dir / "renders" / "eval_grid.png")


def _train_multiview_distributed(cfg: TrainConfig) -> dict:
    device = resolve_device(cfg.device)
    if device.type != "supa":
        raise NotImplementedError("distributed training currently requires a SUPA device")

    available = accelerator_device_count("supa")
    world_size = cfg.distributed_world_size if cfg.distributed_world_size > 0 else available
    if world_size < 2:
        raise ValueError("distributed training requires at least two SUPA devices")
    if world_size > available:
        raise ValueError(f"requested {world_size} SUPA devices but only {available} are visible")

    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    master_addr = cfg.distributed_master_addr
    master_port = cfg.distributed_master_port or _find_free_port()
    launch_cfg = replace(
        cfg,
        distributed=True,
        distributed_world_size=world_size,
        distributed_master_addr=master_addr,
        distributed_master_port=master_port,
    )
    if launch_cfg.resume is not None:
        _load_distributed_resume_checkpoint(launch_cfg.resume)
    mp.spawn(
        _train_multiview_distributed_worker,
        args=(world_size, launch_cfg, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )

    rank_metrics = []
    for rank in range(world_size):
        metrics_path = cfg.result_dir / f"rank_{rank:04d}" / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as file:
            rank_metrics.append(json.load(file))
    _save_distributed_eval_artifacts(launch_cfg, rank_metrics)
    metrics = _aggregate_distributed_metrics(launch_cfg, rank_metrics)
    metrics_path = cfg.result_dir / "metrics.json"
    _save_metrics(metrics, metrics_path)

    return {
        "losses": metrics["train"]["losses"],
        "step_times_s": metrics["train"].get("step_times_s", []),
        "step_time_summary_s": metrics["train"].get("step_time_summary_s"),
        "phase_time_summaries_s": metrics["train"].get("phase_time_summaries_s"),
        "num_gaussians": metrics["num_gaussians"],
        "start_step": metrics["start_step"],
        "step": metrics["step"],
        "final_render": metrics["final_render"],
        "checkpoint": metrics["checkpoint"],
        "ply": metrics["ply"],
        "metrics": str(metrics_path),
        "projection_backend": metrics["projection_backend"],
        "composite_backend": metrics["composite_backend"],
        "composite_variant": metrics["composite_variant"],
        "tile_count_backend": metrics["tile_count_backend"],
        "tiled_metadata_backend": metrics["tiled_metadata_backend"],
        "eval_l1": metrics["eval"]["l1"],
        "eval_mse": metrics["eval"]["mse"],
        "eval_psnr": metrics["eval"]["psnr"],
        "rank_checkpoints": [item["checkpoint"] for item in metrics["ranks"]],
        "rank_plys": [item["ply"] for item in metrics["ranks"]],
    }


def train_multiview(cfg: TrainConfig) -> dict:
    if cfg.distributed:
        return _train_multiview_distributed(cfg)
    return _train_multiview_single(cfg)


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="COLMAP directory or multiview NPZ")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Exact output directory. Defaults to .agents/output/supa_multiview.",
    )
    output_group.add_argument(
        "--auto-output-root",
        type=Path,
        default=None,
        help=(
            "Parent directory for an automatically named run directory. "
            "The child name includes dataset, key training parameters, and a timestamp."
        ),
    )
    parser.add_argument("--data-type", choices=("auto", "npz", "colmap"), default="auto")
    parser.add_argument("--device", type=str, default="supa:0")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--init-num-points",
        type=int,
        default=DEFAULT_INIT_NUM_POINTS,
        help=(
            "Initial Gaussian count. Use a negative value or 0 for all input "
            "points; requests above the input point count are filled by "
            "jittering existing points."
        ),
    )
    parser.add_argument("--init-opa", type=float, default=0.1, help="Initial Gaussian opacity.")
    parser.add_argument("--init-scale", type=float, default=1.0, help="Initial Gaussian scale multiplier.")
    parser.add_argument("--data-factor", type=int, default=4)
    parser.add_argument("--test-every", type=int, default=8)
    _add_bool_arg(
        parser,
        "--normalize-world-space",
        default=True,
        help="Normalize COLMAP world coordinates before training.",
    )
    parser.add_argument(
        "--global-scale",
        type=float,
        default=1.0,
        help="Scale scene-size-dependent learning rates and densification thresholds.",
    )
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    _add_bool_arg(
        parser,
        "--native",
        default=False,
        help="Enable native SUPA kernels where supported.",
    )
    parser.add_argument(
        "--native-tiled",
        choices=("false", "true", "auto"),
        default="false",
        help="Enable no-grad native tiled rendering: false, true, or auto.",
    )
    _add_bool_arg(
        parser,
        "--native-tiled-train",
        default=False,
        help="Enable the experimental native tiled training composite path.",
    )
    _add_bool_arg(
        parser,
        "--native-tiled-metadata",
        default=False,
        help="Enable the experimental native int32-lane tiled metadata path.",
    )
    parser.add_argument(
        "--native-tiled-backward-mode",
        choices=("auto", "pixel", "tile-reduce", "parallel"),
        default="auto",
        help=(
            "Native tiled backward kernel selection. "
            "parallel is an explicit opt-in for the tile-parallel reduced kernel."
        ),
    )
    parser.add_argument(
        "--native-alpha-threshold",
        type=float,
        default=None,
        help=(
            "Alpha cutoff for native tiled metadata. By default, native training "
            "uses the standard 1/255 cutoff; pass 0 for exact current SUPA metadata."
        ),
    )
    _add_bool_arg(
        parser,
        "--distributed",
        default=False,
        help="Enable single-node distributed SUPA training.",
    )
    parser.add_argument(
        "--distributed-world-size",
        type=int,
        default=0,
        help="Number of local SUPA ranks to launch; 0 uses all visible SUPA devices.",
    )
    parser.add_argument("--distributed-master-addr", type=str, default="127.0.0.1")
    parser.add_argument(
        "--distributed-master-port",
        type=int,
        default=0,
        help="Process-group port for distributed mode; 0 selects a free local port.",
    )
    _add_bool_arg(
        parser,
        "--save-ply",
        default=True,
        help="Save PLY point clouds at checkpoint steps.",
    )
    _add_bool_arg(
        parser,
        "--print-all-step-info",
        default=False,
        help="Print training information every step without changing save frequency.",
    )
    _add_bool_arg(
        parser,
        "--print-debug-info",
        default=False,
        help=(
            "Print verbose initialization and per-operation training debug logs. "
            "This is intentionally much noisier than --print-all-step-info."
        ),
    )
    parser.add_argument(
        "--train-crop-size",
        type=int,
        default=0,
        help="Random square crop size for training steps; 0 renders full training images.",
    )
    parser.add_argument(
        "--near-plane",
        type=float,
        default=0.01,
        help="Near clipping plane passed to rasterization.",
    )
    parser.add_argument(
        "--far-plane",
        type=float,
        default=1e10,
        help="Far clipping plane passed to rasterization.",
    )
    parser.add_argument(
        "--radius-clip",
        type=float,
        default=0.0,
        help="Skip Gaussians with projected radius at or below this pixel value.",
    )
    _add_bool_arg(
        parser,
        "--log-step-time",
        default=False,
        help="Print synchronized wall-clock step time in training logs.",
    )
    parser.add_argument(
        "--ssim-lambda",
        type=float,
        default=0.2,
        help="Weight for SSIM loss mixed with L1; use 0 for pure L1.",
    )
    _add_bool_arg(
        parser,
        "--random-bkgd",
        default=False,
        help="Use a random RGB background during training.",
    )
    _add_bool_arg(
        parser,
        "--shuffle-train",
        default=True,
        help="Shuffle training frames deterministically per epoch.",
    )
    parser.add_argument(
        "--means-lr-final-factor",
        type=float,
        default=0.01,
        help="Exponential means LR decay factor reached after the target iteration; use 1 to disable decay.",
    )
    parser.add_argument("--means-lr", type=float, default=1.6e-4)
    parser.add_argument("--scales-lr", type=float, default=5e-3)
    parser.add_argument("--opacities-lr", type=float, default=5e-2)
    parser.add_argument("--quats-lr", type=float, default=1e-3)
    parser.add_argument("--sh0-lr", type=float, default=2.5e-3)
    parser.add_argument("--shN-lr", type=float, default=2.5e-3 / 20.0)
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        help="Maximum spherical-harmonics degree for view-dependent color; use 0 for fixed RGB.",
    )
    parser.add_argument(
        "--sh-degree-interval",
        type=int,
        default=1000,
        help="Training interval for increasing active SH degree.",
    )
    _add_bool_arg(
        parser,
        "--refine-gaussians",
        default=True,
        help="Enable 3DGS densification, pruning, and opacity reset.",
    )
    parser.add_argument(
        "--refine-profile",
        choices=("manual", "full-stable", "crop-balanced"),
        default="manual",
        help="Preset policy for coupling crop training and screen-size refinement.",
    )
    parser.add_argument("--refine-start-iter", type=int, default=500)
    parser.add_argument("--refine-stop-iter", type=int, default=15_000)
    parser.add_argument("--refine-every", type=int, default=100)
    parser.add_argument("--reset-every", type=int, default=3000)
    parser.add_argument("--pause-refine-after-reset", type=int, default=0)
    parser.add_argument("--prune-opa", type=float, default=0.005)
    parser.add_argument("--grow-grad2d", type=float, default=0.0002)
    parser.add_argument("--grow-scale3d", type=float, default=0.01)
    parser.add_argument("--grow-scale2d", type=float, default=0.05)
    parser.add_argument("--prune-scale3d", type=float, default=0.1)
    parser.add_argument("--prune-scale2d", type=float, default=0.15)
    parser.add_argument("--refine-scale2d-stop-iter", type=int, default=0)
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=0,
        help="Maximum Gaussian count after densification; 0 leaves growth uncapped.",
    )
    parser.add_argument(
        "--max-gaussians-ramp-steps",
        type=int,
        default=0,
        help="Ramp the active Gaussian cap over this many steps; 0 uses the cap immediately.",
    )
    _add_bool_arg(
        parser,
        "--refine-exposure",
        default=False,
        help="Optimize per-frame RGB exposure scale and bias.",
    )
    parser.add_argument("--exposure-lr", type=float, default=1e-3)
    parser.add_argument("--exposure-l2", type=float, default=1e-4)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a trainer checkpoint; --iterations is the target total step.",
    )
    parser.add_argument(
        "--eval-max-frames",
        type=int,
        default=8,
        help="Maximum evaluation frames for metrics.json; use 0 to skip multi-frame metrics.",
    )
    args = parser.parse_args()
    if args.auto_output_root is not None:
        args.result_dir = _auto_result_dir(args.auto_output_root, args)
    elif args.result_dir is None:
        args.result_dir = Path(".agents/output/supa_multiview")
    return TrainConfig(**vars(args))


def main() -> None:
    configure_logging()
    cfg = _parse_args()
    _log_cli_startup(cfg)
    stats = train_multiview(cfg)
    log_info(
        LOGGER,
        format_key_value_table(
            "Training Result",
            [
                ("num_gaussians", stats["num_gaussians"]),
                ("eval_psnr", stats["eval_psnr"]),
                ("projection", stats["projection_backend"]),
                ("composite", stats["composite_backend"]),
                ("render", stats["final_render"]),
                ("checkpoint", stats["checkpoint"]),
                ("ply", stats["ply"]),
                ("metrics", stats["metrics"]),
            ],
        ),
    )


if __name__ == "__main__":
    main()
