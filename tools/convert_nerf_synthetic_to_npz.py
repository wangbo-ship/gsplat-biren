#!/usr/bin/env /usr/bin/python3
"""Convert NeRF synthetic transforms and RGBA images to trainer NPZ format."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from gsplat._logging import format_key_value_table, get_logger, log_info


LOGGER = get_logger(__name__)


def _resolve_image_path(dataset_dir: Path, file_path: str) -> Path:
    path = Path(file_path)
    if not path.is_absolute():
        path = dataset_dir / path
    if path.suffix:
        return path
    png_path = path.with_suffix(".png")
    if png_path.is_file():
        return png_path
    return path


def _parse_background(value: str) -> np.ndarray:
    aliases = {
        "white": (1.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0),
    }
    if value in aliases:
        return np.asarray(aliases[value], dtype=np.float32)
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "background must be 'white', 'black', or an r,g,b triple"
        )
    try:
        rgb = np.asarray([float(part) for part in parts], dtype=np.float32)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid RGB background {value!r}") from exc
    if np.any(rgb < 0.0) or np.any(rgb > 1.0):
        raise argparse.ArgumentTypeError("background RGB values must be in [0, 1]")
    return rgb


def _load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def _composite_rgb(rgba: np.ndarray, background: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    composed = rgb * alpha + background.reshape(1, 1, 3) * (1.0 - alpha)
    return np.clip(np.rint(composed * 255.0), 0, 255).astype(np.uint8)


def _camera_intrinsics(width: int, height: int, metadata: dict) -> np.ndarray:
    if "fl_x" in metadata:
        fx = float(metadata["fl_x"])
    else:
        fx = 0.5 * float(width) / math.tan(0.5 * float(metadata["camera_angle_x"]))
    if "fl_y" in metadata:
        fy = float(metadata["fl_y"])
    elif "camera_angle_y" in metadata:
        fy = 0.5 * float(height) / math.tan(0.5 * float(metadata["camera_angle_y"]))
    else:
        fy = fx
    cx = float(metadata.get("cx", width * 0.5))
    cy = float(metadata.get("cy", height * 0.5))
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _convert_camtoworld(raw: np.ndarray, convention: str) -> np.ndarray:
    if convention == "opencv":
        # NeRF synthetic matrices use OpenGL camera axes. Flip camera Y/Z so
        # the resulting view matrix has positive depth for objects in front.
        flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        return raw @ flip
    if convention == "raw":
        return raw
    raise ValueError(f"unsupported coordinate convention {convention!r}")


def convert_nerf_synthetic_to_npz(
    *,
    dataset_dir: Path,
    transforms_path: Path,
    output_path: Path,
    background: np.ndarray,
    background_label: str,
    coordinate_convention: str,
    mask_threshold: int,
    eval_every: int,
    scene_scale: float,
) -> None:
    dataset_dir = dataset_dir.resolve()
    transforms_path = transforms_path.resolve()
    output_path = output_path.resolve()

    metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = metadata["frames"]
    if not frames:
        raise ValueError(f"{transforms_path} contains no frames")

    images = []
    masks = []
    alphas = []
    camtoworlds = []
    frame_names = []
    image_size = None

    for index, frame in enumerate(frames):
        image_path = _resolve_image_path(dataset_dir, frame["file_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"missing image for frame {index}: {image_path}")
        rgba = _load_rgba(image_path)
        if image_size is None:
            image_size = rgba.shape[:2]
        elif rgba.shape[:2] != image_size:
            raise ValueError(
                f"image {image_path} has size {rgba.shape[:2]}, expected {image_size}"
            )
        images.append(_composite_rgb(rgba, background))
        alpha = rgba[..., 3]
        alphas.append(alpha)
        masks.append(alpha > int(mask_threshold))
        raw_c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
        if raw_c2w.shape != (4, 4):
            raise ValueError(f"frame {index} transform_matrix has shape {raw_c2w.shape}")
        camtoworlds.append(_convert_camtoworld(raw_c2w, coordinate_convention))
        frame_names.append(Path(frame["file_path"]).name)

    height, width = image_size
    K = _camera_intrinsics(width, height, metadata)
    Ks = np.repeat(K[None, :, :], len(frames), axis=0).astype(np.float32)
    camtoworlds_np = np.stack(camtoworlds, axis=0).astype(np.float32)
    viewmats_np = np.linalg.inv(camtoworlds_np).astype(np.float32)
    train_indices = np.arange(len(frames), dtype=np.int64)
    eval_indices = np.arange(0, len(frames), max(1, int(eval_every)), dtype=np.int64)
    if eval_indices.size == 0:
        eval_indices = np.asarray([0], dtype=np.int64)
    image_names = np.asarray(frame_names)

    payload = {
        "images": np.stack(images, axis=0),
        "masks": np.stack(masks, axis=0),
        "alphas": np.stack(alphas, axis=0),
        "camtoworlds": camtoworlds_np,
        "viewmats": viewmats_np,
        "Ks": Ks,
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "frame_names": image_names,
        "image_names": image_names,
        "scene_scale": np.asarray(float(scene_scale), dtype=np.float32),
        "camera_angle_x": np.asarray(float(metadata["camera_angle_x"]), dtype=np.float32),
        "background": np.asarray(background_label),
        "coordinate_convention": np.asarray(coordinate_convention),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    log_info(
        LOGGER,
        format_key_value_table(
            "NeRF Synthetic Conversion",
            [
                ("output", output_path),
                ("images", f"{payload['images'].shape} {payload['images'].dtype}"),
                ("masks", f"{payload['masks'].shape} valid={int(payload['masks'].sum())}"),
                ("camtoworlds", payload["camtoworlds"].shape),
                ("viewmats", payload["viewmats"].shape),
                (
                    "Ks",
                    f"{payload['Ks'].shape} fx={float(K[0, 0]):.6f} "
                    f"fy={float(K[1, 1]):.6f}",
                ),
                ("train_indices", train_indices.shape),
                ("eval_indices", eval_indices.shape),
            ],
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory that contains the referenced image paths.",
    )
    parser.add_argument(
        "--transforms",
        type=Path,
        required=True,
        help="NeRF synthetic transforms JSON.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output trainer NPZ.")
    parser.add_argument(
        "--background",
        type=str,
        default="white",
        help="RGBA compositing background: white, black, or an r,g,b triple in [0, 1].",
    )
    parser.add_argument(
        "--coordinate-convention",
        choices=("opencv", "raw"),
        default="opencv",
        help="Convert NeRF synthetic OpenGL camera axes to OpenCV axes by default.",
    )
    parser.add_argument("--mask-threshold", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=8)
    parser.add_argument("--scene-scale", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    convert_nerf_synthetic_to_npz(
        dataset_dir=args.dataset_dir,
        transforms_path=args.transforms,
        output_path=args.output,
        background=_parse_background(args.background),
        background_label=args.background,
        coordinate_convention=args.coordinate_convention,
        mask_threshold=args.mask_threshold,
        eval_every=args.eval_every,
        scene_scale=args.scene_scale,
    )


if __name__ == "__main__":
    main()
