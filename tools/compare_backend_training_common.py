# SPDX-License-Identifier: Apache-2.0

"""Shared COLMAP training diagnostics for SUPA backend probes.

This module intentionally keeps the training loop small and self-contained so
the SUPA fallback and native entry scripts can share data loading,
initialization, loss, logging, and metric code. Backend-specific scripts are
responsible only for selecting the SUPA rasterization options.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from gsplat._logging import get_logger, log_info


LOGGER = get_logger(__name__)
SH_C0 = 0.28209479177387814
_WINDOWS_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_AUTO_OUTPUT_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_WINDOWS_RESERVED_FILENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def add_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help: str,
) -> None:
    parser.add_argument(
        name,
        type=parse_bool,
        default=default,
        metavar="true|false",
        help=f"{help} Defaults to {str(default).lower()}.",
    )


def _slug_part(value: object) -> str:
    text = _AUTO_OUTPUT_UNSAFE_CHARS.sub("-", str(value).strip())
    return text.strip("-._") or "run"


def _auto_result_dir(root: Path, args: argparse.Namespace, backend_name: str) -> Path:
    data_path = Path(args.data)
    dataset_name = data_path.stem if data_path.suffix else data_path.name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [
        _slug_part(dataset_name),
        _slug_part(backend_name),
        f"it{int(args.iterations)}",
        f"pts{int(args.init_num_points)}",
        f"sh{int(args.sh_degree)}",
        f"seed{int(args.seed)}",
        _slug_part(str(args.device).replace(":", "")),
    ]
    if int(args.train_crop_size) > 0:
        parts.append(f"crop{int(args.train_crop_size)}")
    return root / "_".join(parts + [timestamp])


@dataclass
class CameraInfo:
    camera_id: int
    model_name: str
    width: int
    height: int
    calibration: np.ndarray
    distortion_params: np.ndarray


@dataclass
class FrameInfo:
    dataset_index: int
    name: str
    image_path: str
    camera_id: int
    viewmat: Tensor
    K: Tensor
    image: Optional[Tensor]
    height: int
    width: int
    resized_in_memory: bool
    resize_factor: int


@dataclass
class SceneData:
    train_frames: List[FrameInfo]
    eval_frames: List[FrameInfo]
    points: Tensor
    point_colors: Tensor
    scene_scale: float
    transform: np.ndarray
    dataset_info: Dict[str, Any]


@dataclass(frozen=True)
class CropWindow:
    x: int
    y: int
    width: int
    height: int


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_data: str,
    default_result_dir: str,
    default_device: str,
) -> None:
    parser.add_argument("--data", type=Path, default=Path(default_data))
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help=f"Exact output directory. Defaults to {default_result_dir}.",
    )
    output_group.add_argument(
        "--auto-output-root",
        type=Path,
        default=None,
        help=(
            "Parent directory for an automatically named run directory. "
            "The child name includes dataset, backend, key parameters, and a timestamp."
        ),
    )
    parser.set_defaults(_default_result_dir=Path(default_result_dir))
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--init-num-points",
        type=int,
        default=-1,
        help="Use <=0 for all COLMAP points, or a positive fixed Gaussian count.",
    )
    parser.add_argument("--data-factor", type=int, default=4)
    parser.add_argument("--test-every", type=int, default=8)
    add_bool_arg(
        parser,
        "--normalize-world-space",
        default=True,
        help="Normalize COLMAP world coordinates before training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--means-lr-final-factor", type=float, default=0.01)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--sh-degree-interval", type=int, default=250)
    parser.add_argument("--ssim-lambda", type=float, default=0.2)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--train-crop-size", type=int, default=0)
    add_bool_arg(
        parser,
        "--shuffle-train",
        default=True,
        help="Shuffle training frames deterministically per epoch.",
    )
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-max-frames", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--diagnostic-every", type=int, default=1)
    parser.add_argument("--image-dir", type=Path, default=None)
    add_bool_arg(
        parser,
        "--preload-images",
        default=True,
        help="Preload training/evaluation images into memory.",
    )
    add_bool_arg(
        parser,
        "--allow-distorted-cameras",
        default=False,
        help="Allow COLMAP cameras with distortion metadata.",
    )
    parser.add_argument("--init-state", type=Path, default=None)
    parser.add_argument("--init-state-out", type=Path, default=None)
    add_bool_arg(
        parser,
        "--save-init-state",
        default=True,
        help="Save the shared initialization state.",
    )
    add_bool_arg(
        parser,
        "--save-checkpoint",
        default=True,
        help="Save a final training checkpoint.",
    )
    add_bool_arg(
        parser,
        "--overwrite",
        default=False,
        help="Reuse a non-empty result directory.",
    )
    add_bool_arg(
        parser,
        "--stop-on-nonfinite",
        default=True,
        help="Stop training when a non-finite loss is observed.",
    )


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_as_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_as_jsonable(payload), sort_keys=True) + "\n")


def _get_map_items(map_like: Any) -> Iterable[Tuple[int, Any]]:
    return [(int(k), v) for k, v in map_like.items()]


def _get_camera(reconstruction: Any, camera_id: int) -> Any:
    cameras = getattr(reconstruction, "cameras")
    try:
        return cameras[camera_id]
    except TypeError:
        return dict(_get_map_items(cameras))[camera_id]


def _get_image(reconstruction: Any, image_id: int) -> Any:
    image_fn = getattr(reconstruction, "image", None)
    if callable(image_fn):
        return image_fn(image_id)
    return dict(_get_map_items(reconstruction.images))[image_id]


def _camera_model_name(camera: Any) -> str:
    model_name = getattr(camera, "model_name", "")
    if model_name:
        return str(model_name).replace("CameraModelId.", "")
    model = getattr(camera, "model", "")
    return getattr(model, "name", str(model)).replace("CameraModelId.", "")


def _camera_calibration(camera: Any) -> np.ndarray:
    calibration = getattr(camera, "calibration_matrix", None)
    if callable(calibration):
        return np.asarray(calibration(), dtype=np.float64)

    model_name = _camera_model_name(camera)
    params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    if model_name in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        fx, fy, cx, cy = params[:4]
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _camera_distortion(camera: Any) -> np.ndarray:
    model_name = _camera_model_name(camera)
    params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    if model_name in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return np.empty(0, dtype=np.float32)
    if model_name == "SIMPLE_RADIAL":
        return np.asarray([params[3], 0.0, 0.0, 0.0], dtype=np.float32)
    if model_name == "RADIAL":
        return np.asarray([params[3], params[4], 0.0, 0.0], dtype=np.float32)
    if model_name in {"OPENCV", "OPENCV_FISHEYE"}:
        return np.asarray(params[4:8], dtype=np.float32)
    raise ValueError(f"Unsupported COLMAP camera model: {model_name}")


def _image_w2c(image: Any) -> np.ndarray:
    cam_from_world = getattr(image, "cam_from_world", None)
    if callable(cam_from_world):
        cam_from_world = cam_from_world()
    if cam_from_world is not None and hasattr(cam_from_world, "matrix"):
        matrix = np.asarray(cam_from_world.matrix(), dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :4] = matrix[:3, :4]
        return w2c

    qvec = np.asarray(image.qvec, dtype=np.float64)
    tvec = np.asarray(image.tvec, dtype=np.float64).reshape(3, 1)
    qvec = qvec / max(np.linalg.norm(qvec), 1e-12)
    q0, q1, q2, q3 = qvec
    rot = np.asarray(
        [
            [1.0 - 2.0 * q2 * q2 - 2.0 * q3 * q3, 2.0 * q1 * q2 - 2.0 * q0 * q3, 2.0 * q3 * q1 + 2.0 * q0 * q2],
            [2.0 * q1 * q2 + 2.0 * q0 * q3, 1.0 - 2.0 * q1 * q1 - 2.0 * q3 * q3, 2.0 * q2 * q3 - 2.0 * q0 * q1],
            [2.0 * q3 * q1 - 2.0 * q0 * q2, 2.0 * q2 * q3 + 2.0 * q0 * q1, 1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2],
        ],
        dtype=np.float64,
    )
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rot
    w2c[:3, 3:4] = tvec
    return w2c


def _similarity_from_cameras(c2w: np.ndarray, strict_scaling: bool = False) -> np.ndarray:
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]
    ups = np.sum(R * np.asarray([0.0, -1.0, 0.0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= max(np.linalg.norm(world_up), 1e-12)

    up_camspace = np.asarray([0.0, -1.0, 0.0])
    c = float((up_camspace * world_up).sum())
    cross = np.cross(world_up, up_camspace)
    skew = np.asarray(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * (1.0 / max(1.0 + c, 1e-12))
    else:
        R_align = np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    R = R_align @ R
    fwds = np.sum(R * np.asarray([0.0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
    translate = -np.median(nearest, axis=0)

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / max(float(scale_fn(np.linalg.norm(t + translate, axis=-1))), 1e-12)
    transform[:3, :] *= scale
    return transform


def _align_principal_axes(point_cloud: np.ndarray) -> np.ndarray:
    centroid = np.median(point_cloud, axis=0)
    translated = point_cloud - centroid
    covariance = np.cov(translated, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    rotation = eigenvectors.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ centroid
    return transform


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _transform_cameras(matrix: np.ndarray, camtoworlds: np.ndarray) -> np.ndarray:
    camtoworlds = np.einsum("nij,ki->nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


def _rel_paths(path_dir: Path) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(path_dir):
        for filename in files:
            rel_path = os.path.relpath(Path(root) / filename, path_dir)
            out.append(Path(rel_path).as_posix())
    return sorted(out)


def _normalize_colmap_image_name(name: str) -> str:
    return str(name).replace("\\", "/")


def _pil_bicubic_filter() -> Any:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return resampling.BICUBIC
    return Image.BICUBIC


def _safe_artifact_name(name: str, max_length: int = 160) -> str:
    sanitized = _WINDOWS_UNSAFE_FILENAME_CHARS.sub("_", str(name)).strip(" .")
    if not sanitized:
        sanitized = "frame"
    stem_for_reserved = os.path.splitext(sanitized)[0].upper()
    if stem_for_reserved in _WINDOWS_RESERVED_FILENAMES:
        sanitized = f"_{sanitized}"
    if len(sanitized) <= max_length:
        return sanitized
    digest = hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:12]
    stem, suffix = os.path.splitext(sanitized)
    suffix = suffix[:24]
    stem_limit = max(max_length - len(digest) - len(suffix) - 1, 1)
    return f"{stem[:stem_limit]}_{digest}{suffix}"


def _select_image_dir(data_dir: Path, factor: int, explicit: Optional[Path]) -> Tuple[Path, bool, str]:
    if explicit is not None:
        return explicit, False, "explicit"
    if factor > 1:
        for name in (f"images_{factor}_png", f"images_{factor}"):
            candidate = data_dir / name
            if candidate.exists():
                return candidate, False, name
    images = data_dir / "images"
    return images, factor > 1, "images_resized_in_memory" if factor > 1 else "images"


def _load_image(path: Path, *, resize_in_memory: bool, factor: int) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if resize_in_memory and factor > 1:
            width = max(int(round(image.width / factor)), 1)
            height = max(int(round(image.height / factor)), 1)
            image = image.resize((width, height), _pil_bicubic_filter())
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).contiguous()


def _image_shape(path: Path, *, resize_in_memory: bool, factor: int) -> Tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    if resize_in_memory and factor > 1:
        width = max(int(round(width / factor)), 1)
        height = max(int(round(height / factor)), 1)
    return height, width


def _resolve_image_paths(data_dir: Path, image_dir: Path, image_names: List[str]) -> Dict[str, Path]:
    colmap_files = _rel_paths(data_dir / "images")
    image_files = _rel_paths(image_dir)
    direct = {name: image_dir / name for name in image_names if (image_dir / name).exists()}
    if len(direct) == len(image_names):
        return direct
    if len(colmap_files) != len(image_files):
        raise ValueError(
            f"Cannot map COLMAP image names to {image_dir}: "
            f"{len(colmap_files)} source images vs {len(image_files)} selected images."
        )
    mapped = dict(zip(colmap_files, image_files))
    missing = [name for name in image_names if name not in mapped]
    if missing:
        raise ValueError(f"Selected image directory cannot map COLMAP image {missing[0]!r}.")
    return {name: image_dir / mapped[name] for name in image_names}


def load_colmap_scene(args: argparse.Namespace) -> SceneData:
    data_dir = Path(args.data)
    sparse_dir = data_dir / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = data_dir / "sparse"
    if not sparse_dir.exists():
        raise FileNotFoundError(f"COLMAP sparse directory does not exist: {sparse_dir}")

    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    reg_image_ids = [int(image_id) for image_id in reconstruction.reg_image_ids()]
    if not reg_image_ids:
        raise ValueError(f"No registered COLMAP images found in {sparse_dir}")

    cameras: Dict[int, CameraInfo] = {}
    for camera_id, camera in _get_map_items(reconstruction.cameras):
        cameras[camera_id] = CameraInfo(
            camera_id=camera_id,
            model_name=_camera_model_name(camera),
            width=int(camera.width),
            height=int(camera.height),
            calibration=_camera_calibration(camera),
            distortion_params=_camera_distortion(camera),
        )

    image_rows = []
    for image_id in reg_image_ids:
        image = _get_image(reconstruction, image_id)
        w2c = _image_w2c(image)
        image_rows.append(
            {
                "image_id": image_id,
                "name": _normalize_colmap_image_name(str(image.name)),
                "camera_id": int(image.camera_id),
                "w2c": w2c,
            }
        )
    image_rows.sort(key=lambda item: item["name"])
    image_names = [item["name"] for item in image_rows]
    camera_ids = [int(item["camera_id"]) for item in image_rows]
    w2c_mats = np.stack([item["w2c"] for item in image_rows], axis=0)
    camtoworlds = np.linalg.inv(w2c_mats)

    points3D = dict(_get_map_items(reconstruction.points3D))
    point_ids = sorted(points3D)
    if not point_ids:
        raise ValueError(f"No COLMAP points3D found in {sparse_dir}")
    points = np.asarray([points3D[point_id].xyz for point_id in point_ids], dtype=np.float32).reshape(-1, 3)
    point_colors = np.asarray([points3D[point_id].color for point_id in point_ids], dtype=np.float32).reshape(-1, 3) / 255.0

    transform = np.eye(4, dtype=np.float64)
    if args.normalize_world_space:
        T1 = _similarity_from_cameras(camtoworlds)
        camtoworlds = _transform_cameras(T1, camtoworlds)
        points = _transform_points(T1, points)
        T2 = _align_principal_axes(points)
        camtoworlds = _transform_cameras(T2, camtoworlds)
        points = _transform_points(T2, points)
        transform = T2 @ T1
        if np.median(points[:, 2]) > np.mean(points[:, 2]):
            T3 = np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            camtoworlds = _transform_cameras(T3, camtoworlds)
            points = _transform_points(T3, points)
            transform = T3 @ transform

    image_dir, resize_in_memory, image_dir_mode = _select_image_dir(data_dir, int(args.data_factor), args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    image_paths = _resolve_image_paths(data_dir, image_dir, image_names)

    distorted = {
        str(camera_id): {
            "model": camera.model_name,
            "distortion_params": camera.distortion_params.tolist(),
        }
        for camera_id, camera in cameras.items()
        if camera.distortion_params.size > 0 and bool(np.any(np.abs(camera.distortion_params) > 0.0))
    }
    if distorted and not args.allow_distorted_cameras:
        raise ValueError(
            "The comparison scripts do not undistort images. Re-run with "
            "--allow-distorted-cameras to continue and record this as a quality risk."
        )

    frames: List[FrameInfo] = []
    for dataset_index, (row, camtoworld) in enumerate(zip(image_rows, camtoworlds)):
        camera = cameras[int(row["camera_id"])]
        path = image_paths[str(row["name"])]
        height, width = _image_shape(path, resize_in_memory=resize_in_memory, factor=int(args.data_factor))
        factor = max(int(args.data_factor), 1)
        expected_width = max(camera.width // factor, 1)
        expected_height = max(camera.height // factor, 1)
        K = camera.calibration.copy()
        K[:2, :] /= factor
        K[0, :] *= float(width) / float(expected_width)
        K[1, :] *= float(height) / float(expected_height)
        image = (
            _load_image(path, resize_in_memory=resize_in_memory, factor=int(args.data_factor))
            if args.preload_images
            else None
        )
        frames.append(
            FrameInfo(
                dataset_index=dataset_index,
                name=str(row["name"]),
                image_path=str(path),
                camera_id=int(row["camera_id"]),
                viewmat=torch.from_numpy(np.linalg.inv(camtoworld).astype(np.float32)).contiguous(),
                K=torch.from_numpy(K.astype(np.float32)).contiguous(),
                image=image,
                height=height,
                width=width,
                resized_in_memory=resize_in_memory,
                resize_factor=factor,
            )
        )

    indices = np.arange(len(frames))
    test_every = int(args.test_every)
    train_indices = indices[indices % test_every != 0] if test_every > 0 else indices
    eval_indices = indices[indices % test_every == 0] if test_every > 0 else indices[:1]
    if train_indices.size == 0:
        train_indices = indices
    if eval_indices.size == 0:
        eval_indices = train_indices[:1]

    camera_locations = camtoworlds[:, :3, 3]
    scene_center = np.mean(camera_locations, axis=0)
    dists = np.linalg.norm(camera_locations - scene_center, axis=1)
    scene_scale = max(float(np.max(dists)), 1e-3)

    return SceneData(
        train_frames=[frames[int(i)] for i in train_indices],
        eval_frames=[frames[int(i)] for i in eval_indices],
        points=torch.from_numpy(points.astype(np.float32)).contiguous(),
        point_colors=torch.from_numpy(point_colors.astype(np.float32)).contiguous(),
        scene_scale=scene_scale,
        transform=transform,
        dataset_info={
            "data_dir": str(data_dir),
            "sparse_dir": str(sparse_dir),
            "image_dir": str(image_dir),
            "image_dir_mode": image_dir_mode,
            "resized_in_memory": resize_in_memory,
            "num_images": len(frames),
            "num_train_frames": int(len(train_indices)),
            "num_eval_frames": int(len(eval_indices)),
            "num_points3D": int(points.shape[0]),
            "camera_models": {str(k): v.model_name for k, v in cameras.items()},
            "distorted_cameras": distorted,
            "first_train_frame": frames[int(train_indices[0])].name,
            "first_eval_frame": frames[int(eval_indices[0])].name,
            "scene_scale": scene_scale,
            "normalize_world_space": bool(args.normalize_world_space),
        },
    )


def _rgb_to_sh(rgb: Tensor) -> Tensor:
    return (rgb - 0.5) / SH_C0


def _sh_coeff_count(sh_degree: int) -> int:
    return max((int(sh_degree) + 1) ** 2 - 1, 0)


def _logit_np(values: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    values = np.clip(values, eps, 1.0 - eps)
    return np.log(values / (1.0 - values))


def _estimate_log_scales_np(points: np.ndarray, scene_scale: float, seed: int) -> np.ndarray:
    n_points = int(points.shape[0])
    if n_points <= 1:
        base = np.full((n_points, 1), max(scene_scale * 0.01, 1e-3), dtype=np.float32)
        return np.log(base).repeat(3, axis=1).astype(np.float32)

    rng = np.random.default_rng(seed + 1234)
    max_ref = min(n_points, 2048)
    ref_ids = rng.permutation(n_points)[:max_ref]
    ref = points[ref_ids].astype(np.float32)
    same_ref = max_ref == n_points
    k = 4 if same_ref else min(3, max_ref)
    chunks = []
    for start in range(0, n_points, 4096):
        chunk = points[start : start + 4096].astype(np.float32)
        diff = chunk[:, None, :] - ref[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        nearest = np.partition(dist, kth=k - 1, axis=1)[:, :k]
        if same_ref and nearest.shape[1] > 1:
            nearest = nearest[:, 1:]
        chunks.append(nearest.mean(axis=1))
    avg_dist = np.concatenate(chunks, axis=0)
    avg_dist = np.maximum(avg_dist, max(scene_scale * 1e-4, 1e-5))
    return np.log(avg_dist[:, None]).repeat(3, axis=1).astype(np.float32)


def _select_initial_points(scene: SceneData, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    points = scene.points.numpy().astype(np.float32, copy=True)
    colors = scene.point_colors.numpy().astype(np.float32, copy=True)
    target = int(args.init_num_points)
    if target <= 0:
        target = int(points.shape[0])

    rng = np.random.default_rng(int(args.seed))
    if points.shape[0] > target:
        ids = rng.permutation(points.shape[0])[:target]
        points = points[ids]
        colors = colors[ids]
    elif points.shape[0] < target:
        missing = target - points.shape[0]
        base_ids = np.random.default_rng(int(args.seed) + 7919).integers(0, points.shape[0], size=missing)
        base_points = points[base_ids]
        base_colors = colors[base_ids]
        point_extent = float(np.linalg.norm(points.astype(np.float32).std(axis=0)))
        jitter_std = max(min(point_extent, scene.scene_scale) * 0.0025, scene.scene_scale * 1e-4, 1e-5)
        jitter = np.random.default_rng(int(args.seed) + 1543).normal(0.0, jitter_std, size=(missing, 3)).astype(np.float32)
        points = np.concatenate((points, base_points + jitter), axis=0)
        colors = np.concatenate((colors, base_colors), axis=0)
    return points.astype(np.float32), np.clip(colors.astype(np.float32), 0.0, 1.0)


def _state_digest(state: Dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def create_initial_state(scene: SceneData, args: argparse.Namespace) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    if args.init_state is not None:
        try:
            payload = torch.load(args.init_state, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(args.init_state, map_location="cpu")
        splats = payload.get("splats", payload)
        state = {name: value.detach().float().contiguous() for name, value in splats.items()}
        return state, {
            "source": str(args.init_state),
            "loaded": True,
            "num_gaussians": int(state["means"].shape[0]),
            "digest": _state_digest(state),
            "source_manifest": payload.get("init_manifest") if isinstance(payload, dict) else None,
        }

    points, colors = _select_initial_points(scene, args)
    rng = np.random.default_rng(int(args.seed) + 17)
    quats = rng.normal(size=(points.shape[0], 4)).astype(np.float32)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    log_scales = _estimate_log_scales_np(points, scene.scene_scale, int(args.seed))
    opacities = _logit_np(np.full((points.shape[0],), 0.1, dtype=np.float32))
    color_logits = _logit_np(colors)

    state: Dict[str, Tensor] = {
        "means": torch.from_numpy(points).contiguous(),
        "log_scales": torch.from_numpy(log_scales).contiguous(),
        "quats": torch.from_numpy(quats).contiguous(),
        "opacities": torch.from_numpy(opacities.astype(np.float32)).contiguous(),
        "colors": torch.from_numpy(color_logits.astype(np.float32)).contiguous(),
    }
    sh_coeffs = _sh_coeff_count(int(args.sh_degree))
    if sh_coeffs > 0:
        state["shN"] = torch.zeros((points.shape[0], sh_coeffs * 3), dtype=torch.float32)
    return state, {
        "source": "deterministic_numpy",
        "loaded": False,
        "num_gaussians": int(points.shape[0]),
        "input_points3D": int(scene.points.shape[0]),
        "init_num_points_arg": int(args.init_num_points),
        "seed": int(args.seed),
        "digest": _state_digest(state),
    }


def _params_from_state(state: Dict[str, Tensor], device: torch.device) -> torch.nn.ParameterDict:
    return torch.nn.ParameterDict(
        {name: torch.nn.Parameter(value.detach().clone()) for name, value in state.items()}
    ).to(device)


def _base_lrs(scene_scale: float, lr_scale: float, sh_present: bool) -> List[Tuple[str, float]]:
    groups = [
        ("means", 1.6e-4 * scene_scale * lr_scale),
        ("log_scales", 5e-3 * lr_scale),
        ("quats", 1e-3 * lr_scale),
        ("opacities", 5e-2 * lr_scale),
        ("colors", 2.5e-3 * lr_scale),
    ]
    if sh_present:
        groups.append(("shN", 2.5e-3 / 20.0 * lr_scale))
    return groups


def create_optimizer(params: torch.nn.ParameterDict, scene_scale: float, args: argparse.Namespace) -> torch.optim.Optimizer:
    groups = [
        {"params": [params[name]], "lr": lr, "name": name}
        for name, lr in _base_lrs(scene_scale, float(args.lr_scale), "shN" in params)
    ]
    return torch.optim.Adam(groups, eps=1e-15)


def _scheduled_means_lr(base_lr: float, final_factor: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(base_lr)
    progress = min(max(step - 1, 0), total_steps - 1) / float(total_steps - 1)
    return float(base_lr * (float(final_factor) ** progress))


def set_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
    scene_scale: float,
    args: argparse.Namespace,
    step: int,
) -> Dict[str, float]:
    base = dict(_base_lrs(scene_scale, float(args.lr_scale), any(g.get("name") == "shN" for g in optimizer.param_groups)))
    current = {}
    for group in optimizer.param_groups:
        name = str(group.get("name"))
        lr = base[name]
        if name == "means":
            lr = _scheduled_means_lr(lr, float(args.means_lr_final_factor), step, int(args.iterations))
        group["lr"] = lr
        current[name] = float(lr)
    return current


def _active_sh_degree(step: int, args: argparse.Namespace) -> int:
    if int(args.sh_degree) <= 0:
        return 0
    interval = max(int(args.sh_degree_interval), 1)
    return min(max(int(step), 0) // interval, int(args.sh_degree))


def _frame_index(step: int, num_frames: int, args: argparse.Namespace) -> int:
    position = step - 1
    if not args.shuffle_train:
        return position % num_frames
    epoch, offset = divmod(position, num_frames)
    order = np.random.default_rng(int(args.seed) + int(epoch)).permutation(num_frames)
    return int(order[offset])


def _sample_crop(frame: FrameInfo, step: int, args: argparse.Namespace) -> Optional[CropWindow]:
    size = int(args.train_crop_size)
    if size <= 0:
        return None
    width = min(size, int(frame.width))
    height = min(size, int(frame.height))
    if width == frame.width and height == frame.height:
        return None
    rng = np.random.default_rng(int(args.seed) * 1_000_003 + int(step))
    x = int(rng.integers(0, max(int(frame.width) - width + 1, 1)))
    y = int(rng.integers(0, max(int(frame.height) - height + 1, 1)))
    return CropWindow(x=x, y=y, width=width, height=height)


def _frame_image(frame: FrameInfo) -> Tensor:
    if frame.image is not None:
        return frame.image
    return _load_image(
        Path(frame.image_path),
        resize_in_memory=bool(frame.resized_in_memory),
        factor=int(frame.resize_factor),
    )


def _crop_image_and_K(frame: FrameInfo, crop: Optional[CropWindow]) -> Tuple[Tensor, Tensor]:
    image = _frame_image(frame)
    K = frame.K.clone()
    if crop is None:
        return image, K
    K[0, 2] -= float(crop.x)
    K[1, 2] -= float(crop.y)
    return image[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width].contiguous(), K


def _ssim_loss(render: Tensor, target: Tensor) -> Tensor:
    height, width = int(render.shape[0]), int(render.shape[1])
    window_size = max(1, min(11, height, width))
    if window_size > 1 and window_size % 2 == 0:
        window_size -= 1
    padding = window_size // 2
    render_nchw = render.permute(2, 0, 1).unsqueeze(0)
    target_nchw = target.permute(2, 0, 1).unsqueeze(0)
    mu_render = F.avg_pool2d(render_nchw, window_size, stride=1, padding=padding)
    mu_target = F.avg_pool2d(target_nchw, window_size, stride=1, padding=padding)
    render_sq_mean = F.avg_pool2d(render_nchw * render_nchw, window_size, stride=1, padding=padding)
    target_sq_mean = F.avg_pool2d(target_nchw * target_nchw, window_size, stride=1, padding=padding)
    render_target_mean = F.avg_pool2d(render_nchw * target_nchw, window_size, stride=1, padding=padding)
    mu_render_sq = mu_render.square()
    mu_target_sq = mu_target.square()
    sigma_render_sq = render_sq_mean - mu_render_sq
    sigma_target_sq = target_sq_mean - mu_target_sq
    sigma_render_target = render_target_mean - mu_render * mu_target
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_render * mu_target + c1) * (2.0 * sigma_render_target + c2)
    denominator = (mu_render_sq + mu_target_sq + c1) * (sigma_render_sq + sigma_target_sq + c2)
    return 1.0 - (numerator / denominator.clamp_min(1e-12)).clamp(-1.0, 1.0).mean()


def _render_loss(render: Tensor, target: Tensor, ssim_lambda: float) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    l1 = F.l1_loss(render, target)
    if ssim_lambda <= 0.0:
        return l1, l1, None
    ssim = _ssim_loss(render, target)
    return l1 * (1.0 - float(ssim_lambda)) + ssim * float(ssim_lambda), l1, ssim


def _psnr_from_mse(mse: float) -> Optional[float]:
    if not math.isfinite(mse) or mse <= 0.0:
        return None
    return float(-10.0 * math.log10(mse))


def _sync_device(device: torch.device) -> None:
    if device.type == "supa" and hasattr(torch, "supa"):
        torch.supa.synchronize()


def _set_device(device: torch.device) -> None:
    index = 0 if device.index is None else int(device.index)
    if device.type == "supa" and hasattr(torch, "supa"):
        torch.supa.set_device(index)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "supa" and hasattr(torch, "supa"):
        reset = getattr(torch.supa, "reset_peak_memory_stats", None)
        if callable(reset):
            try:
                reset()
            except TypeError:
                reset(device)


def _memory_stats(device: torch.device) -> Dict[str, Any]:
    if device.type == "supa" and hasattr(torch, "supa"):
        stats: Dict[str, Any] = {}
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
            fn = getattr(torch.supa, name, None)
            if callable(fn):
                try:
                    stats[name] = int(fn())
                except TypeError:
                    stats[name] = int(fn(device))
        return stats
    return {}


def _device_info(device: torch.device) -> Dict[str, Any]:
    if device.type == "supa" and hasattr(torch, "supa"):
        index = 0 if device.index is None else int(device.index)
        info = {"type": "supa", "index": index}
        props_fn = getattr(torch.supa, "get_device_properties", None)
        if callable(props_fn):
            props = props_fn(index)
            info["name"] = getattr(props, "name", str(props))
            for attr in ("total_memory", "major", "minor"):
                if hasattr(props, attr):
                    info[attr] = getattr(props, attr)
        return info
    return {"type": device.type, "index": device.index}


def _tensor_scalar(value: Tensor) -> float:
    return float(value.detach().cpu().item())


def _param_stats(params: torch.nn.ParameterDict) -> Dict[str, Any]:
    with torch.no_grad():
        scales = torch.exp(params["log_scales"].detach())
        opacities = torch.sigmoid(params["opacities"].detach())
        return {
            "num_gaussians": int(params["means"].shape[0]),
            "opacity_mean": _tensor_scalar(opacities.mean()),
            "opacity_min": _tensor_scalar(opacities.min()),
            "opacity_max": _tensor_scalar(opacities.max()),
            "scale_mean": _tensor_scalar(scales.mean()),
            "scale_max": _tensor_scalar(scales.max()),
            "means_abs_max": _tensor_scalar(params["means"].detach().abs().max()),
        }


def _grad_stats(params: torch.nn.ParameterDict) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, param in params.items():
        grad = param.grad
        if grad is None:
            out[name] = {"present": False}
            continue
        finite = torch.isfinite(grad)
        out[name] = {
            "present": True,
            "finite": bool(finite.all().detach().cpu().item()),
            "norm": _tensor_scalar(torch.linalg.vector_norm(grad.detach())),
            "abs_max": _tensor_scalar(grad.detach().abs().max()),
            "mean": _tensor_scalar(grad.detach().mean()),
            "nonzero_fraction": _tensor_scalar((grad.detach() != 0).to(torch.float32).mean()),
        }
    return out


def _meta_stats(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in (
        "backend",
        "projection_backend",
        "covariance_backend",
        "composite_backend",
        "composite_variant",
        "tile_count_backend",
        "tiled_metadata_backend",
    ):
        if key in meta:
            out[key] = meta.get(key)
    for key in ("visible_gaussians", "culled_gaussians", "width", "height", "tile_size", "tile_width", "tile_height"):
        if key in meta:
            value = meta.get(key)
            out[key] = int(value) if isinstance(value, (int, np.integer)) else _as_jsonable(value)
    radii = meta.get("radii")
    if isinstance(radii, Tensor):
        positive = (radii > 0).all(dim=-1)
        out["radii_shape"] = list(radii.shape)
        out["radii_positive_count"] = int(positive.sum().detach().cpu().item())
        out["radii_max"] = int(radii.detach().max().cpu().item()) if radii.numel() else 0
    depths = meta.get("depths")
    if isinstance(depths, Tensor):
        finite = torch.isfinite(depths)
        out["depths_finite_fraction"] = _tensor_scalar(finite.to(torch.float32).mean()) if depths.numel() else None
        finite_depths = depths[finite]
        out["depths_min"] = _tensor_scalar(finite_depths.min()) if finite_depths.numel() else None
        out["depths_max"] = _tensor_scalar(finite_depths.max()) if finite_depths.numel() else None
    tiles = meta.get("tiles_per_gauss")
    if isinstance(tiles, Tensor):
        out["tiles_shape"] = list(tiles.shape)
        out["tiles_sum"] = int(tiles.detach().sum().cpu().item()) if tiles.numel() else 0
        out["tiles_max"] = int(tiles.detach().max().cpu().item()) if tiles.numel() else 0
        out["tiles_mean"] = _tensor_scalar(tiles.detach().to(torch.float32).mean()) if tiles.numel() else 0.0
    for key in ("isect_ids", "flatten_ids", "isect_offsets"):
        value = meta.get(key)
        if isinstance(value, Tensor):
            out[f"{key}_numel"] = int(value.numel())
            out[f"{key}_shape"] = list(value.shape)
    return out


def _prepare_colors(params: torch.nn.ParameterDict, active_sh_degree: int) -> Tuple[Tensor, Optional[int]]:
    base_rgb = torch.sigmoid(params["colors"])
    if active_sh_degree > 0 and "shN" in params:
        sh_tail = params["shN"].reshape(params["means"].shape[0], -1, 3)
        colors = torch.cat((_rgb_to_sh(base_rgb).unsqueeze(1), sh_tail), dim=1)
        return colors, int(active_sh_degree)
    return base_rgb, None


def rasterize_frame(
    rasterization_fn: Callable[..., Tuple[Tensor, Tensor, Dict[str, Any]]],
    params: torch.nn.ParameterDict,
    frame: FrameInfo,
    target_cpu: Tensor,
    K_cpu: Tensor,
    device: torch.device,
    args: argparse.Namespace,
    active_sh_degree: int,
    supa_native_options: Optional[Dict[str, object]] = None,
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    colors, sh_degree = _prepare_colors(params, active_sh_degree)
    raster_kwargs = {
        "means": params["means"],
        "quats": F.normalize(params["quats"], dim=-1),
        "scales": torch.exp(params["log_scales"]),
        "opacities": torch.sigmoid(params["opacities"]),
        "colors": colors,
        "viewmats": frame.viewmat.to(device).unsqueeze(0),
        "Ks": K_cpu.to(device).unsqueeze(0),
        "width": int(target_cpu.shape[1]),
        "height": int(target_cpu.shape[0]),
        "near_plane": 0.01,
        "far_plane": 1e10,
        "radius_clip": float(args.radius_clip),
        "eps2d": 0.3,
        "sh_degree": sh_degree,
        "packed": False,
        "tile_size": 16,
        "backgrounds": None,
        "render_mode": "RGB",
        "rasterize_mode": "classic",
        "channel_chunk": 32,
        "camera_model": "pinhole",
        "covars": None,
    }
    if device.type == "supa":
        raster_kwargs.update(
            {
                "supa_native_options": supa_native_options,
                "supa_use_torch_fallback": True,
            }
        )
    render, alpha, meta = rasterization_fn(**raster_kwargs)
    return render[0], alpha[0], meta


def _save_png(tensor: Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (tensor.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(array).save(path)


def evaluate(
    rasterization_fn: Callable[..., Tuple[Tensor, Tensor, Dict[str, Any]]],
    params: torch.nn.ParameterDict,
    scene: SceneData,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    out_dir: Path,
) -> Dict[str, Any]:
    frames = scene.eval_frames[: max(int(args.eval_max_frames), 0)]
    if not frames:
        return {"num_frames": 0, "l1": None, "mse": None, "psnr": None}
    l1_values = []
    mse_values = []
    last_meta: Dict[str, Any] = {}
    active_sh_degree = min(int(args.sh_degree), _active_sh_degree(max(step, int(args.iterations)), args))
    with torch.no_grad():
        for index, frame in enumerate(frames):
            target_cpu, K_cpu = _crop_image_and_K(frame, None)
            target = target_cpu.to(device)
            render, _, meta = rasterize_frame(
                rasterization_fn,
                params,
                frame,
                target_cpu,
                K_cpu,
                device,
                args,
                active_sh_degree,
                getattr(args, "supa_native_options", None),
            )
            _sync_device(device)
            l1_values.append(_tensor_scalar(F.l1_loss(render, target)))
            mse_values.append(_tensor_scalar(F.mse_loss(render, target)))
            last_meta = meta
            if index == 0:
                artifact_name = _safe_artifact_name(frame.name)
                _save_png(render, out_dir / "renders" / f"eval_step_{step:04d}_{artifact_name}.png")
                _save_png(target, out_dir / "renders" / f"eval_step_{step:04d}_{artifact_name}_target.png")
    mse = float(np.mean(mse_values))
    l1 = float(np.mean(l1_values))
    return {
        "num_frames": len(frames),
        "l1": l1,
        "mse": mse,
        "psnr": _psnr_from_mse(mse),
        "last_meta": _meta_stats(last_meta),
    }


def _environment_payload(
    *,
    backend_name: str,
    device: torch.device,
    gsplat_module: Any,
    extra_environment: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = {
        "backend_name": backend_name,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "pycolmap_version": getattr(pycolmap, "__version__", None),
        "gsplat_version": getattr(gsplat_module, "__version__", None),
        "gsplat_file": getattr(gsplat_module, "__file__", None),
        "device": _device_info(device),
        "argv": sys.argv,
    }
    if extra_environment:
        payload.update(extra_environment)
    return payload


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.iterations) < 0:
        raise ValueError("iterations must be non-negative")
    if int(args.data_factor) <= 0:
        raise ValueError("data-factor must be positive")
    if int(args.test_every) <= 0:
        raise ValueError("test-every must be positive")
    if int(args.sh_degree) < 0 or int(args.sh_degree) > 3:
        raise ValueError("sh-degree must be in [0, 3] for this comparison script")
    if not 0.0 <= float(args.ssim_lambda) <= 1.0:
        raise ValueError("ssim-lambda must be in [0, 1]")
    if not 0.0 < float(args.means_lr_final_factor) <= 1.0:
        raise ValueError("means-lr-final-factor must be in (0, 1]")


def run_training(
    args: argparse.Namespace,
    *,
    backend_name: str,
    gsplat_module: Any,
    rasterization_fn: Callable[..., Tuple[Tensor, Tensor, Dict[str, Any]]],
    extra_environment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _validate_args(args)
    if args.auto_output_root is not None:
        result_dir = _auto_result_dir(Path(args.auto_output_root), args, backend_name)
        args.result_dir = result_dir
    else:
        result_dir = Path(args.result_dir or args._default_result_dir)
        args.result_dir = result_dir
    if result_dir.exists() and any(result_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{result_dir} is not empty; pass --overwrite true to reuse it.")
    result_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    _set_device(device)

    scene = load_colmap_scene(args)
    initial_state, init_manifest = create_initial_state(scene, args)
    params = _params_from_state(initial_state, device)
    optimizer = create_optimizer(params, scene.scene_scale, args)
    metrics_path = result_dir / "step_metrics.jsonl"
    if metrics_path.exists() and args.overwrite:
        metrics_path.unlink()

    if args.save_init_state:
        init_state_path = args.init_state_out or (result_dir / "init_state.pt")
        torch.save({"splats": initial_state, "init_manifest": init_manifest}, init_state_path)
        init_manifest["saved_to"] = str(init_state_path)

    environment = _environment_payload(
        backend_name=backend_name,
        device=device,
        gsplat_module=gsplat_module,
        extra_environment=extra_environment,
    )
    config_payload = {
        "args": vars(args),
        "environment": environment,
        "dataset": scene.dataset_info,
        "init_manifest": init_manifest,
        "comparison_contract": {
            "packed": False,
            "tile_size": 16,
            "render_mode": "RGB",
            "camera_model": "pinhole",
            "optimizer": "torch.optim.Adam grouped like gsplat-biren trainer",
            "initialization": "CPU float32 tensors from deterministic NumPy or --init-state",
            "densification": "disabled in this diagnostic script",
        },
    }
    _write_json(result_dir / "run_config.json", config_payload)

    losses: List[float] = []
    l1_losses: List[float] = []
    ssim_losses: List[Optional[float]] = []
    psnrs: List[Optional[float]] = []
    step_times: List[float] = []
    forward_times: List[float] = []
    backward_times: List[float] = []
    optimizer_times: List[float] = []
    eval_records: List[Dict[str, Any]] = []
    last_meta: Dict[str, Any] = {}

    for step in range(1, int(args.iterations) + 1):
        _reset_peak_memory(device)
        _sync_device(device)
        total_start = time.perf_counter()
        current_lrs = set_optimizer_lrs(optimizer, scene.scene_scale, args, step)
        frame_index = _frame_index(step, len(scene.train_frames), args)
        frame = scene.train_frames[frame_index]
        crop = _sample_crop(frame, step, args)

        target_cpu, K_cpu = _crop_image_and_K(frame, crop)
        transfer_start = time.perf_counter()
        target = target_cpu.to(device)
        _sync_device(device)
        transfer_s = time.perf_counter() - transfer_start

        active_sh_degree = _active_sh_degree(step, args)
        optimizer.zero_grad(set_to_none=True)

        forward_start = time.perf_counter()
        render, alpha, meta = rasterize_frame(
            rasterization_fn,
            params,
            frame,
            target_cpu,
            K_cpu,
            device,
            args,
            active_sh_degree,
            getattr(args, "supa_native_options", None),
        )
        _sync_device(device)
        forward_s = time.perf_counter() - forward_start

        loss_start = time.perf_counter()
        loss, l1, ssim = _render_loss(render, target, float(args.ssim_lambda))
        mse = F.mse_loss(render.detach(), target.detach())
        _sync_device(device)
        loss_s = time.perf_counter() - loss_start

        backward_start = time.perf_counter()
        loss.backward()
        _sync_device(device)
        backward_s = time.perf_counter() - backward_start

        step_start = time.perf_counter()
        optimizer.step()
        _sync_device(device)
        optimizer_s = time.perf_counter() - step_start
        total_s = time.perf_counter() - total_start

        loss_value = _tensor_scalar(loss)
        l1_value = _tensor_scalar(l1)
        ssim_value = None if ssim is None else _tensor_scalar(ssim)
        mse_value = _tensor_scalar(mse)
        psnr_value = _psnr_from_mse(mse_value)
        losses.append(loss_value)
        l1_losses.append(l1_value)
        ssim_losses.append(ssim_value)
        psnrs.append(psnr_value)
        step_times.append(total_s)
        forward_times.append(forward_s)
        backward_times.append(backward_s)
        optimizer_times.append(optimizer_s)
        last_meta = meta

        if int(args.record_every) > 0 and step % int(args.record_every) == 0:
            grad_payload = (
                _grad_stats(params)
                if int(args.diagnostic_every) > 0
                and step % int(args.diagnostic_every) == 0
                else {}
            )
            record = {
                "type": "train_step",
                "step": step,
                "frame_index": frame_index,
                "frame_name": frame.name,
                "frame": {
                    "dataset_index": frame.dataset_index,
                    "camera_id": frame.camera_id,
                    "image_path": frame.image_path,
                    "full_width": frame.width,
                    "full_height": frame.height,
                    "render_width": int(target_cpu.shape[1]),
                    "render_height": int(target_cpu.shape[0]),
                    "K": K_cpu.tolist(),
                },
                "crop": asdict(crop) if crop is not None else None,
                "loss": loss_value,
                "l1": l1_value,
                "ssim": ssim_value,
                "mse": mse_value,
                "psnr": psnr_value,
                "active_sh_degree": active_sh_degree,
                "lrs": current_lrs,
                "timing_s": {
                    "total": total_s,
                    "host_to_device_target": transfer_s,
                    "raster_forward": forward_s,
                    "loss": loss_s,
                    "backward": backward_s,
                    "optimizer": optimizer_s,
                },
                "memory": _memory_stats(device),
                "meta": _meta_stats(meta),
                "param_stats_after_step": _param_stats(params),
                "grad_stats": grad_payload,
                "render_stats": {
                    "mean": _tensor_scalar(render.detach().mean()),
                    "min": _tensor_scalar(render.detach().min()),
                    "max": _tensor_scalar(render.detach().max()),
                    "alpha_mean": _tensor_scalar(alpha.detach().mean()),
                },
                "target_stats": {
                    "mean": _tensor_scalar(target.detach().mean()),
                    "min": _tensor_scalar(target.detach().min()),
                    "max": _tensor_scalar(target.detach().max()),
                },
            }
            _append_jsonl(metrics_path, record)

        if args.stop_on_nonfinite and not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss_value}")

        if int(args.print_every) > 0 and (step % int(args.print_every) == 0 or step == int(args.iterations)):
            log_info(
                LOGGER,
                f"{backend_name} step {step:04d}/{int(args.iterations):04d} "
                f"loss={loss_value:.6f} l1={l1_value:.6f} "
                f"psnr={psnr_value if psnr_value is not None else 'nan'} "
                f"forward={forward_s:.4f}s backward={backward_s:.4f}s "
                f"total={total_s:.4f}s "
                f"projection={meta.get('projection_backend')} composite={meta.get('composite_backend')}",
            )

        if int(args.save_every) > 0 and step % int(args.save_every) == 0:
            _save_png(render, result_dir / "renders" / f"train_step_{step:04d}.png")

        if int(args.eval_every) > 0 and step % int(args.eval_every) == 0:
            eval_record = evaluate(rasterization_fn, params, scene, device, args, step, result_dir)
            eval_record["step"] = step
            eval_records.append(eval_record)
            _append_jsonl(metrics_path, {"type": "eval", **eval_record})

    final_eval = evaluate(rasterization_fn, params, scene, device, args, int(args.iterations), result_dir)
    final_eval["step"] = int(args.iterations)
    eval_records.append(final_eval)

    if args.save_checkpoint:
        checkpoint = {
            "splats": {name: value.detach().cpu() for name, value in params.items()},
            "optimizer": optimizer.state_dict(),
            "step": int(args.iterations),
            "config": config_payload,
        }
        torch.save(checkpoint, result_dir / "final_checkpoint.pt")

    summary = {
        "backend_name": backend_name,
        "result_dir": str(result_dir),
        "step": int(args.iterations),
        "num_gaussians": int(params["means"].shape[0]),
        "init_digest": init_manifest["digest"],
        "train": {
            "num_steps": len(losses),
            "last_loss": losses[-1] if losses else None,
            "last_l1": l1_losses[-1] if l1_losses else None,
            "last_ssim": ssim_losses[-1] if ssim_losses else None,
            "last_psnr": psnrs[-1] if psnrs else None,
            "mean_step_time_s": float(np.mean(step_times)) if step_times else None,
            "median_step_time_s": float(np.median(step_times)) if step_times else None,
            "mean_forward_time_s": float(np.mean(forward_times)) if forward_times else None,
            "mean_backward_time_s": float(np.mean(backward_times)) if backward_times else None,
            "mean_optimizer_time_s": float(np.mean(optimizer_times)) if optimizer_times else None,
        },
        "eval": final_eval,
        "last_meta": _meta_stats(last_meta),
        "memory": _memory_stats(device),
        "environment": environment,
        "dataset": scene.dataset_info,
        "diagnostic_notes": [
            "A matching init_digest across SUPA fallback/native runs is the strongest check that initial splats match.",
            "The scripts use packed=False because gsplat-biren SUPA fallback does not support packed=True.",
            "Densification is disabled here so Gaussian count changes cannot hide backend/rendering differences.",
            "Large raster_forward/backward timings with meta backend=torch_fallback point to Python/PyTorch fallback overhead.",
            "Low radii_positive_count or tiles_sum can explain weak gradients and poor quality independent of backend speed.",
        ],
    }
    _write_json(result_dir / "summary.json", summary)
    log_info(
        LOGGER,
        f"{backend_name} done result_dir={result_dir} "
        f"init_digest={init_manifest['digest']} "
        f"final_eval_psnr={final_eval.get('psnr')} "
        f"summary={result_dir / 'summary.json'}",
    )
    return summary
