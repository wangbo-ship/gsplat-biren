#!/usr/bin/env /usr/bin/python3
"""Convert COLMAP binary reconstructions to multiview trainer NPZ files."""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from gsplat._logging import format_key_value_table, get_logger, log_info


LOGGER = get_logger(__name__)


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def _read_next_bytes(handle, num_bytes: int, fmt: str):
    data = handle.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"expected {num_bytes} bytes, got {len(data)}")
    return struct.unpack(fmt, data)


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras = {}
    with path.open("rb") as handle:
        (num_cameras,) = _read_next_bytes(handle, 8, "<Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next_bytes(handle, 24, "<iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"unsupported COLMAP camera model id {model_id}")
            model, num_params = CAMERA_MODELS[model_id]
            params = np.asarray(
                _read_next_bytes(handle, 8 * num_params, "<" + "d" * num_params),
                dtype=np.float64,
            )
            cameras[int(camera_id)] = Camera(
                camera_id=int(camera_id),
                model=model,
                width=int(width),
                height=int(height),
                params=params,
            )
    return cameras


def _read_c_string(handle) -> str:
    chars = bytearray()
    while True:
        char = handle.read(1)
        if char == b"":
            raise EOFError("unexpected EOF while reading COLMAP image name")
        if char == b"\x00":
            return chars.decode("utf-8")
        chars.extend(char)


def read_images_binary(path: Path) -> list[ColmapImage]:
    images = []
    with path.open("rb") as handle:
        (num_images,) = _read_next_bytes(handle, 8, "<Q")
        for _ in range(num_images):
            image_id = _read_next_bytes(handle, 4, "<i")[0]
            qvec = np.asarray(_read_next_bytes(handle, 32, "<dddd"), dtype=np.float64)
            tvec = np.asarray(_read_next_bytes(handle, 24, "<ddd"), dtype=np.float64)
            camera_id = _read_next_bytes(handle, 4, "<i")[0]
            name = _read_c_string(handle)
            (num_points2d,) = _read_next_bytes(handle, 8, "<Q")
            handle.seek(num_points2d * 24, 1)
            images.append(
                ColmapImage(
                    image_id=int(image_id),
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=int(camera_id),
                    name=name,
                )
            )
    return images


def read_points3d_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points = []
    colors = []
    with path.open("rb") as handle:
        (num_points,) = _read_next_bytes(handle, 8, "<Q")
        for _ in range(num_points):
            handle.seek(8, 1)  # point3D_id
            xyz = _read_next_bytes(handle, 24, "<ddd")
            rgb = _read_next_bytes(handle, 3, "<BBB")
            handle.seek(8, 1)  # reprojection error
            (track_len,) = _read_next_bytes(handle, 8, "<Q")
            handle.seek(track_len * 8, 1)
            points.append(xyz)
            colors.append(rgb)
    return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.uint8)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = qvec
    return np.asarray(
        [
            [
                1.0 - 2.0 * q2 * q2 - 2.0 * q3 * q3,
                2.0 * q1 * q2 - 2.0 * q0 * q3,
                2.0 * q3 * q1 + 2.0 * q0 * q2,
            ],
            [
                2.0 * q1 * q2 + 2.0 * q0 * q3,
                1.0 - 2.0 * q1 * q1 - 2.0 * q3 * q3,
                2.0 * q2 * q3 - 2.0 * q0 * q1,
            ],
            [
                2.0 * q3 * q1 - 2.0 * q0 * q2,
                2.0 * q2 * q3 + 2.0 * q0 * q1,
                1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2,
            ],
        ],
        dtype=np.float64,
    )


def camera_K(camera: Camera, actual_width: int, actual_height: int) -> np.ndarray:
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params
        K = np.asarray(
            [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
        K = np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    else:
        raise ValueError(
            f"{camera.model} has distortion; undistort images before NPZ conversion"
        )

    K[0, :] *= float(actual_width) / float(camera.width)
    K[1, :] *= float(actual_height) / float(camera.height)
    return K.astype(np.float32)


def image_to_viewmat(image: ColmapImage) -> np.ndarray:
    viewmat = np.eye(4, dtype=np.float64)
    viewmat[:3, :3] = qvec_to_rotmat(image.qvec)
    viewmat[:3, 3] = image.tvec
    return viewmat.astype(np.float32)


def split_indices(num_images: int, test_every: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(num_images, dtype=np.int64)
    if test_every <= 0 or num_images == 1:
        return indices, np.asarray([0], dtype=np.int64)
    train = indices[indices % test_every != 0]
    eval_indices = indices[indices % test_every == 0]
    if train.size == 0:
        train = indices
    if eval_indices.size == 0:
        eval_indices = train[:1]
    return train, eval_indices


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _resolve_model_dir(data_dir: Path, sparse_dir: Path | None) -> Path:
    if sparse_dir is not None:
        return sparse_dir
    nested = data_dir / "sparse" / "0"
    if nested.is_dir():
        return nested
    return data_dir / "sparse"


def convert_dataset(
    data_dir: Path,
    output_path: Path,
    test_every: int,
    *,
    sparse_dir: Path | None = None,
    image_dir: Path | None = None,
) -> None:
    model_dir = _resolve_model_dir(data_dir, sparse_dir)
    image_dir = image_dir or data_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing image directory: {image_dir}")

    cameras = read_cameras_binary(model_dir / "cameras.bin")
    images = sorted(read_images_binary(model_dir / "images.bin"), key=lambda item: item.name)
    points, point_colors = read_points3d_binary(model_dir / "points3D.bin")
    if not images:
        raise ValueError(f"no registered COLMAP images in {data_dir}")

    image_arrays = []
    viewmats = []
    Ks = []
    image_names = []
    for image in images:
        image_path = image_dir / image.name
        if not image_path.is_file():
            raise FileNotFoundError(f"missing image referenced by COLMAP: {image_path}")
        rgb = load_rgb(image_path)
        height, width = rgb.shape[:2]
        image_arrays.append(rgb)
        viewmats.append(image_to_viewmat(image))
        Ks.append(camera_K(cameras[image.camera_id], width, height))
        image_names.append(image.name)

    first_shape = image_arrays[0].shape
    mismatched = [
        (name, array.shape)
        for name, array in zip(image_names, image_arrays)
        if array.shape != first_shape
    ]
    if mismatched:
        raise ValueError(f"images do not share one shape; first mismatch: {mismatched[0]}")

    camtoworlds = np.linalg.inv(np.stack(viewmats, axis=0).astype(np.float64))
    camera_locations = camtoworlds[:, :3, 3]
    scene_center = camera_locations.mean(axis=0)
    scene_scale = np.linalg.norm(camera_locations - scene_center, axis=1).max()
    train_indices, eval_indices = split_indices(len(image_arrays), test_every)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        images=np.stack(image_arrays, axis=0),
        viewmats=np.stack(viewmats, axis=0).astype(np.float32),
        Ks=np.stack(Ks, axis=0).astype(np.float32),
        points=points.astype(np.float32),
        point_colors=point_colors.astype(np.uint8),
        train_indices=train_indices,
        eval_indices=eval_indices,
        scene_scale=np.asarray(scene_scale, dtype=np.float32),
        image_names=np.asarray(image_names),
    )
    log_info(
        LOGGER,
        format_key_value_table(
            "COLMAP Binary Conversion",
            [
                ("output", output_path),
                ("images", len(image_arrays)),
                ("shape", first_shape),
                ("points", points.shape[0]),
                ("train", train_indices.size),
                ("eval", eval_indices.size),
                ("scene_scale", f"{scene_scale:.6f}"),
            ],
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path, help="COLMAP dataset root(s).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ path. Only valid when one dataset is provided.",
    )
    parser.add_argument(
        "--suffix",
        default="original",
        help="Batch output suffix: <dataset>/<dataset-name>-<suffix>.npz.",
    )
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--sparse-dir", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.output is not None and len(args.datasets) != 1:
        raise SystemExit("--output can only be used with a single dataset")
    for data_dir in args.datasets:
        output_path = args.output or data_dir / f"{data_dir.name}-{args.suffix}.npz"
        convert_dataset(
            data_dir,
            output_path,
            args.test_every,
            sparse_dir=args.sparse_dir,
            image_dir=args.image_dir,
        )


if __name__ == "__main__":
    main()
