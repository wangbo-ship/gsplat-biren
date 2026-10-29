# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing_extensions import assert_never

from gsplat._logging import get_logger, log_info, log_warning

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


LOGGER = get_logger(__name__)

try:
    import pycolmap as _pycolmap
except ImportError:  # pragma: no cover - depends on the host Python environment
    _pycolmap = None

PycolmapReconstruction = (
    getattr(_pycolmap, "Reconstruction", None) if _pycolmap is not None else None
)
SceneManager = (
    getattr(_pycolmap, "SceneManager", None) if _pycolmap is not None else None
)


def _import_cv2():
    try:
        import cv2  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - depends on the host Python environment
        raise ImportError(
            "OpenCV (cv2) is required only for COLMAP image undistortion or "
            "debug point visualization. Install it when those features are "
            "needed; pycolmap-based sparse model loading does not require cv2."
        ) from exc
    return cv2


def _import_imageio():
    try:
        import imageio.v2 as imageio  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - depends on the host Python environment
        raise ImportError(
            "imageio is required only for COLMAP image reading, resizing, or "
            "debug video writing. Sparse COLMAP model loading does not require "
            "imageio."
        ) from exc
    return imageio


def _read_rgb_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def _write_rgb_image(path: str | Path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image).astype(np.uint8)).save(path)


@dataclass(frozen=True)
class _TextColmapCamera:
    camera_id: int
    camera_type: str
    width: int
    height: int
    params: np.ndarray

    @property
    def fx(self) -> float:
        if self.camera_type in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            return float(self.params[0])
        return float(self.params[0])

    @property
    def fy(self) -> float:
        if self.camera_type in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            return float(self.params[0])
        return float(self.params[1])

    @property
    def cx(self) -> float:
        if self.camera_type in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            return float(self.params[1])
        return float(self.params[2])

    @property
    def cy(self) -> float:
        if self.camera_type in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            return float(self.params[2])
        return float(self.params[3])

    @property
    def k1(self) -> float:
        if self.camera_type == "SIMPLE_RADIAL":
            return float(self.params[3])
        if self.camera_type == "RADIAL":
            return float(self.params[3])
        if self.camera_type in ("OPENCV", "OPENCV_FISHEYE"):
            return float(self.params[4])
        return 0.0

    @property
    def k2(self) -> float:
        if self.camera_type == "RADIAL":
            return float(self.params[4])
        if self.camera_type in ("OPENCV", "OPENCV_FISHEYE"):
            return float(self.params[5])
        return 0.0

    @property
    def p1(self) -> float:
        if self.camera_type == "OPENCV":
            return float(self.params[6])
        return 0.0

    @property
    def p2(self) -> float:
        if self.camera_type == "OPENCV":
            return float(self.params[7])
        return 0.0

    @property
    def k3(self) -> float:
        if self.camera_type == "OPENCV_FISHEYE":
            return float(self.params[6])
        return 0.0

    @property
    def k4(self) -> float:
        if self.camera_type == "OPENCV_FISHEYE":
            return float(self.params[7])
        return 0.0


@dataclass(frozen=True)
class _TextColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    point3D_ids: np.ndarray

    def R(self) -> np.ndarray:
        return _qvec_to_rotmat(self.qvec)


@dataclass(frozen=True)
class _PoseColmapImage:
    image_id: int
    rotation: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str

    def R(self) -> np.ndarray:
        return self.rotation


@dataclass(frozen=True)
class _ColmapModel:
    cameras: Dict[int, Any]
    images: Dict[int, Any]
    points3D: np.ndarray
    point3D_errors: np.ndarray
    point3D_colors: np.ndarray
    point_indices: Dict[str, np.ndarray]


_CAMERA_MODEL_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
}


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    log_info(LOGGER, f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = _read_rgb_image(image_path)
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        _write_rgb_image(resized_path, resized_image)
    return resized_dir


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    norm = np.linalg.norm(qvec)
    if norm > 0.0:
        qvec = qvec / norm
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


def _read_cameras_text(path: str) -> Dict[int, _TextColmapCamera]:
    cameras: Dict[int, _TextColmapCamera] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                raise ValueError(f"Malformed COLMAP camera line in {path}: {stripped!r}")
            camera_id = int(parts[0])
            camera_type = parts[1]
            if camera_type not in _CAMERA_MODEL_NUM_PARAMS:
                raise ValueError(
                    f"Unsupported COLMAP camera model {camera_type!r} in {path}. "
                    "Supported text fallback models are SIMPLE_PINHOLE, PINHOLE, "
                    "SIMPLE_RADIAL, RADIAL, OPENCV, and OPENCV_FISHEYE."
                )
            expected = _CAMERA_MODEL_NUM_PARAMS[camera_type]
            params = np.asarray([float(value) for value in parts[4:]], dtype=np.float64)
            if params.size != expected:
                raise ValueError(
                    f"COLMAP camera {camera_id} model {camera_type} expects "
                    f"{expected} params, got {params.size}"
                )
            cameras[camera_id] = _TextColmapCamera(
                camera_id=camera_id,
                camera_type=camera_type,
                width=int(parts[2]),
                height=int(parts[3]),
                params=params,
            )
    return cameras


def _read_images_text(path: str) -> Dict[int, _TextColmapImage]:
    images: Dict[int, _TextColmapImage] = {}
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=9)
        if len(parts) != 10:
            raise ValueError(f"Malformed COLMAP image line in {path}: {stripped!r}")
        image_id = int(parts[0])
        qvec = np.asarray([float(value) for value in parts[1:5]], dtype=np.float64)
        tvec = np.asarray([float(value) for value in parts[5:8]], dtype=np.float64)
        camera_id = int(parts[8])
        name = parts[9]
        points_line = lines[index].strip() if index < len(lines) else ""
        index += 1
        point_tokens = points_line.split()
        if len(point_tokens) % 3 != 0:
            raise ValueError(
                f"Malformed COLMAP POINTS2D line for image {image_id} in {path}"
            )
        point_ids = [
            int(point_tokens[offset])
            for offset in range(2, len(point_tokens), 3)
            if int(point_tokens[offset]) >= 0
        ]
        images[image_id] = _TextColmapImage(
            image_id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            point3D_ids=np.asarray(point_ids, dtype=np.int64),
        )
    return images


def _read_points3d_text(
    path: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[int, int],
    Dict[int, List[tuple[int, int]]],
]:
    points = []
    errors = []
    colors = []
    id_to_index: Dict[int, int] = {}
    point_tracks: Dict[int, List[tuple[int, int]]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                raise ValueError(f"Malformed COLMAP point line in {path}: {stripped!r}")
            point_id = int(parts[0])
            id_to_index[point_id] = len(points)
            points.append([float(parts[1]), float(parts[2]), float(parts[3])])
            colors.append([int(parts[4]), int(parts[5]), int(parts[6])])
            errors.append(float(parts[7]))
            track_tokens = parts[8:]
            if len(track_tokens) % 2 != 0:
                raise ValueError(
                    f"Malformed COLMAP point track for point {point_id} in {path}"
                )
            point_tracks[point_id] = [
                (int(track_tokens[offset]), int(track_tokens[offset + 1]))
                for offset in range(0, len(track_tokens), 2)
            ]

    return (
        np.asarray(points, dtype=np.float32).reshape(-1, 3),
        np.asarray(errors, dtype=np.float32),
        np.asarray(colors, dtype=np.uint8).reshape(-1, 3),
        id_to_index,
        point_tracks,
    )


def _load_colmap_text_model(colmap_dir: str) -> _ColmapModel:
    cameras = _read_cameras_text(os.path.join(colmap_dir, "cameras.txt"))
    images = _read_images_text(os.path.join(colmap_dir, "images.txt"))
    (
        points,
        point_errors,
        point_colors,
        point_id_to_index,
        point_tracks,
    ) = _read_points3d_text(os.path.join(colmap_dir, "points3D.txt"))
    point_indices = {}
    image_id_to_name = {image.image_id: image.name for image in images.values()}
    for point_id, track in point_tracks.items():
        point_idx = point_id_to_index[point_id]
        for image_id, _ in track:
            image_name = image_id_to_name.get(image_id)
            if image_name is not None:
                point_indices.setdefault(image_name, []).append(point_idx)
    point_indices = {
        image_name: np.asarray(indices, dtype=np.int32)
        for image_name, indices in point_indices.items()
    }
    return _ColmapModel(
        cameras=cameras,
        images=images,
        points3D=points,
        point3D_errors=point_errors,
        point3D_colors=point_colors,
        point_indices=point_indices,
    )


def _load_colmap_reconstruction_model(colmap_dir: str) -> _ColmapModel:
    if PycolmapReconstruction is None:  # pragma: no cover - caller gates this
        raise RuntimeError("pycolmap.Reconstruction is not available")

    reconstruction = PycolmapReconstruction(colmap_dir)

    cameras = {}
    for camera_id, camera in reconstruction.cameras.items():
        camera_type = str(camera.model_name)
        if camera_type not in _CAMERA_MODEL_NUM_PARAMS:
            raise ValueError(
                f"Unsupported COLMAP camera model {camera_type!r} in {colmap_dir}. "
                "Supported pycolmap models are SIMPLE_PINHOLE, PINHOLE, "
                "SIMPLE_RADIAL, RADIAL, OPENCV, and OPENCV_FISHEYE."
            )
        params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
        expected = _CAMERA_MODEL_NUM_PARAMS[camera_type]
        if params.size != expected:
            raise ValueError(
                f"COLMAP camera {camera_id} model {camera_type} expects "
                f"{expected} params, got {params.size}"
            )
        cameras[int(camera_id)] = _TextColmapCamera(
            camera_id=int(camera_id),
            camera_type=camera_type,
            width=int(camera.width),
            height=int(camera.height),
            params=params,
        )

    images = {}
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        if not image.has_pose:
            continue
        cam_from_world = image.cam_from_world().matrix()
        images[int(image_id)] = _PoseColmapImage(
            image_id=int(image.image_id),
            rotation=np.asarray(cam_from_world[:, :3], dtype=np.float64),
            tvec=np.asarray(cam_from_world[:, 3], dtype=np.float64),
            camera_id=int(image.camera_id),
            name=str(image.name),
        )

    point_ids = list(reconstruction.points3D.keys())
    point_id_to_index = {
        int(point_id): index for index, point_id in enumerate(point_ids)
    }
    points = np.empty((len(point_ids), 3), dtype=np.float32)
    point_errors = np.empty((len(point_ids),), dtype=np.float32)
    point_colors = np.empty((len(point_ids), 3), dtype=np.uint8)
    point_indices = {}
    image_id_to_name = {image.image_id: image.name for image in images.values()}
    for point_id in point_ids:
        point = reconstruction.points3D[point_id]
        point_idx = point_id_to_index[int(point_id)]
        points[point_idx] = np.asarray(point.xyz, dtype=np.float32).reshape(3)
        point_errors[point_idx] = np.float32(point.error)
        point_colors[point_idx] = np.asarray(point.color, dtype=np.uint8).reshape(3)
        for track_element in point.track.elements:
            image_name = image_id_to_name.get(int(track_element.image_id))
            if image_name is not None:
                point_indices.setdefault(image_name, []).append(point_idx)
    point_indices = {
        image_name: np.asarray(indices, dtype=np.int32)
        for image_name, indices in point_indices.items()
    }

    return _ColmapModel(
        cameras=cameras,
        images=images,
        points3D=points,
        point3D_errors=point_errors,
        point3D_colors=point_colors,
        point_indices=point_indices,
    )


def _load_colmap_model(colmap_dir: str) -> _ColmapModel:
    if PycolmapReconstruction is not None:
        return _load_colmap_reconstruction_model(colmap_dir)

    if SceneManager is not None:
        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        point_indices = {}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.asarray(v, dtype=np.int32) for k, v in point_indices.items()
        }
        return _ColmapModel(
            cameras=manager.cameras,
            images=manager.images,
            points3D=manager.points3D.astype(np.float32),
            point3D_errors=manager.point3D_errors.astype(np.float32),
            point3D_colors=manager.point3D_colors.astype(np.uint8),
            point_indices=point_indices,
        )

    required = ("cameras.txt", "images.txt", "points3D.txt")
    missing = [name for name in required if not os.path.exists(os.path.join(colmap_dir, name))]
    if missing:
        raise RuntimeError(
            "pycolmap is not installed and the COLMAP text fallback cannot run "
            f"because {colmap_dir} is missing: {', '.join(missing)}"
        )
    log_info(LOGGER, "[Parser] pycolmap is not installed; reading COLMAP text sparse files.")
    return _load_colmap_text_model(colmap_dir)


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        load_exposure: bool = False,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.load_exposure = load_exposure

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        model = _load_colmap_model(colmap_dir)

        # Extract extrinsic matrices in world-to-camera format.
        imdata = model.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        camtype_dict = dict()
        has_distortion = False
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = model.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            camtype_dict[camera_id] = camtype
            has_distortion = has_distortion or len(params) > 0
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        log_info(
            LOGGER,
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if has_distortion:
            log_warning(LOGGER, "COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        if not os.path.exists(colmap_image_dir):
            raise ValueError(f"Image folder {colmap_image_dir} does not exist.")
        if not os.path.exists(image_dir):
            if factor > 1 and not self.extconf["no_factor_suffix"]:
                image_dir = _resize_image_folder(
                    colmap_image_dir, image_dir + "_png", factor=factor
                )
            else:
                raise ValueError(f"Image folder {image_dir} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if len(image_files) != len(colmap_files):
            raise ValueError(
                "Downsampled COLMAP image folder does not match original image "
                f"folder: {image_dir} has {len(image_files)} files, "
                f"{colmap_image_dir} has {len(colmap_files)} files."
            )
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
            if len(image_files) != len(colmap_files):
                raise ValueError(
                    "Resized COLMAP image folder does not match original image "
                    f"folder: {image_dir} has {len(image_files)} files, "
                    f"{colmap_image_dir} has {len(colmap_files)} files."
                )
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = model.points3D.astype(np.float32)
        points_err = model.point3D_errors.astype(np.float32)
        points_rgb = model.point3D_colors.astype(np.uint8)
        point_indices = model.point_indices

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # Create 0-based contiguous camera indices from COLMAP camera_ids.
        # This is useful for camera-based embeddings/modules.
        unique_camera_ids = sorted(set(camera_ids))
        self.camera_id_to_idx = {cid: idx for idx, cid in enumerate(unique_camera_ids)}
        self.camera_indices = [self.camera_id_to_idx[cid] for cid in camera_ids]
        self.num_cameras = len(unique_camera_ids)

        # Load EXIF exposure data if requested.
        # Always read from original (non-downscaled) images since PNG doesn't support EXIF.
        if load_exposure:
            try:
                from .exif import compute_exposure_from_exif
            except ImportError as exc:
                raise RuntimeError(
                    "COLMAP EXIF exposure loading requires the optional piexif "
                    "dependency, which is not available in this environment."
                ) from exc

            exposure_values: List[Optional[float]] = []
            for image_name in tqdm(image_names, desc="Loading EXIF exposure"):
                original_path = Path(colmap_image_dir) / image_name
                exposure_values.append(compute_exposure_from_exif(original_path))

            # Compute mean across all valid exposures and subtract
            valid_exposures = [e for e in exposure_values if e is not None]
            if valid_exposures:
                exposure_mean = sum(valid_exposures) / len(valid_exposures)
                self.exposure_values: List[Optional[float]] = [
                    (e - exposure_mean) if e is not None else None
                    for e in exposure_values
                ]
                log_info(
                    LOGGER,
                    f"[Parser] Loaded exposure for {len(valid_exposures)}/{len(exposure_values)} images "
                    f"(mean={exposure_mean:.3f} EV)"
                )
            else:
                self.exposure_values = [None] * len(exposure_values)
                log_info(LOGGER, "[Parser] No valid EXIF exposure data found in any image.")
        else:
            self.exposure_values = [None] * len(image_paths)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = _read_rgb_image(self.image_paths[0])
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            camtype = camtype_dict[camera_id]
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                cv2 = _import_cv2()
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = _read_rgb_image(self.parser.image_paths[index])
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            cv2 = _import_cv2()
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "camera_idx": self.parser.camera_indices[
                index
            ],  # 0-based contiguous camera index
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        # Add exposure if available for this image
        exposure = self.parser.exposure_values[index]
        if exposure is not None:
            data["exposure"] = torch.tensor(exposure, dtype=torch.float32)

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    log_info(LOGGER, f"Dataset: {len(dataset)} images.")

    imageio = _import_imageio()
    writer = imageio.get_writer("results/points.mp4", fps=30)
    cv2 = _import_cv2()
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
