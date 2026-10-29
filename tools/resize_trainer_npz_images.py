#!/usr/bin/env /usr/bin/python3
"""Resize trainer NPZ images and scale intrinsics to match."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from gsplat._logging import format_key_value_table, get_logger, log_info


LOGGER = get_logger(__name__)


def _target_size(
    source_width: int,
    source_height: int,
    target_width: int | None,
    target_height: int | None,
) -> tuple[int, int]:
    if target_width is None and target_height is None:
        raise ValueError("at least one of target_width or target_height is required")
    if target_width is None:
        target_width = int(round(source_width * int(target_height) / source_height))
    if target_height is None:
        target_height = int(round(source_height * int(target_width) / source_width))
    if int(target_width) <= 0 or int(target_height) <= 0:
        raise ValueError(f"invalid target size {target_width}x{target_height}")
    return int(target_width), int(target_height)


def _resize_array(array: np.ndarray, size: tuple[int, int], *, nearest: bool) -> np.ndarray:
    pil_mode = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    image = Image.fromarray(array)
    resized = image.resize(size, pil_mode)
    return np.asarray(resized, dtype=array.dtype)


def _source_image_paths(source: np.lib.npyio.NpzFile, image_dir: Path, key: str) -> list[Path]:
    if key not in source:
        raise KeyError(f"source NPZ does not contain image name key {key!r}")
    names = [str(name) for name in source[key]]
    if not names:
        raise ValueError(f"source NPZ key {key!r} contains no image names")
    return [image_dir / name for name in names]


def _load_images_from_dir(
    source: np.lib.npyio.NpzFile,
    image_dir: Path,
    image_name_key: str,
    target_size_: tuple[int, int] | None,
    target_width: int | None,
    target_height: int | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    paths = _source_image_paths(source, image_dir, image_name_key)
    images = []
    resolved_size = target_size_
    first_source_size = None

    for index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(f"missing source image {index}: {path}")
        with Image.open(path) as image:
            image = image.convert("RGB")
            source_width, source_height = image.size
            if first_source_size is None:
                first_source_size = (source_width, source_height)
                if resolved_size is None:
                    resolved_size = _target_size(
                        source_width,
                        source_height,
                        target_width,
                        target_height,
                    )
                log_info(
                    LOGGER,
                    f"resize {source_width}x{source_height} -> "
                    f"{resolved_size[0]}x{resolved_size[1]}"
                )
            elif (source_width, source_height) != first_source_size:
                raise ValueError(
                    f"unexpected source size for {path}: "
                    f"{source_width}x{source_height}, expected {first_source_size}"
                )
            images.append(np.asarray(image.resize(resolved_size, Image.Resampling.LANCZOS), dtype=np.uint8))
        if (index + 1) % 16 == 0 or index + 1 == len(paths):
            log_info(LOGGER, f"processed {index + 1}/{len(paths)} images")

    assert resolved_size is not None
    return np.stack(images, axis=0), resolved_size


def _resize_npz_images(
    images: np.ndarray,
    target_width: int | None,
    target_height: int | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    if images.ndim != 4 or images.shape[-1] not in (1, 3, 4):
        raise ValueError(f"images must have shape [M, H, W, C], got {images.shape}")
    source_height, source_width = images.shape[1:3]
    size = _target_size(source_width, source_height, target_width, target_height)
    resized = [
        _resize_array(images[index], size, nearest=False)
        for index in range(images.shape[0])
    ]
    log_info(LOGGER, f"resize {source_width}x{source_height} -> {size[0]}x{size[1]}")
    return np.stack(resized, axis=0).astype(images.dtype, copy=False), size


def _resize_optional_image_stack(
    payload: dict[str, np.ndarray],
    key: str,
    source_hw: tuple[int, int],
    target_size_: tuple[int, int],
    *,
    nearest: bool,
) -> None:
    if key not in payload:
        return
    array = payload[key]
    if array.ndim < 3 or tuple(array.shape[1:3]) != tuple(source_hw):
        return
    resized = [
        _resize_array(array[index], target_size_, nearest=nearest)
        for index in range(array.shape[0])
    ]
    payload[key] = np.stack(resized, axis=0).astype(array.dtype, copy=False)


def resize_trainer_npz_images(
    *,
    input_path: Path,
    output_path: Path,
    target_width: int | None,
    target_height: int | None,
    image_dir: Path | None,
    image_name_key: str,
) -> None:
    source = np.load(input_path)
    if "images" not in source:
        raise KeyError("source NPZ must contain an images array")
    source_images = source["images"]
    if source_images.ndim != 4:
        raise ValueError(f"source images must have shape [M, H, W, C], got {source_images.shape}")
    source_hw = tuple(int(value) for value in source_images.shape[1:3])
    payload = {key: source[key] for key in source.files}

    if image_dir is None:
        images, target_size_ = _resize_npz_images(source_images, target_width, target_height)
    else:
        images, target_size_ = _load_images_from_dir(
            source,
            image_dir,
            image_name_key,
            None,
            target_width,
            target_height,
        )
    payload["images"] = images

    if "Ks" in payload:
        target_w, target_h = target_size_
        source_h, source_w = source_hw
        k_width_scale = target_w / float(source_w)
        k_height_scale = target_h / float(source_h)
        Ks = payload["Ks"].astype(np.float32, copy=True)
        Ks[:, 0, :] *= np.float32(k_width_scale)
        Ks[:, 1, :] *= np.float32(k_height_scale)
        payload["Ks"] = Ks
        log_info(LOGGER, f"Ks scale width={k_width_scale:.6f} height={k_height_scale:.6f}")

    _resize_optional_image_stack(
        payload,
        "masks",
        source_hw,
        target_size_,
        nearest=True,
    )
    _resize_optional_image_stack(
        payload,
        "mask",
        source_hw,
        target_size_,
        nearest=True,
    )
    _resize_optional_image_stack(
        payload,
        "alphas",
        source_hw,
        target_size_,
        nearest=False,
    )
    _resize_optional_image_stack(
        payload,
        "alpha",
        source_hw,
        target_size_,
        nearest=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    log_info(
        LOGGER,
        format_key_value_table(
            "Trainer NPZ Resize",
            [
                ("output", output_path),
                ("images", f"{payload['images'].shape} {payload['images'].dtype}"),
            ],
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input trainer NPZ.")
    parser.add_argument("--output", type=Path, required=True, help="Output trainer NPZ.")
    parser.add_argument("--target-width", type=int, default=None)
    parser.add_argument("--target-height", type=int, default=None)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional directory of source RGB images named by the NPZ image-name key.",
    )
    parser.add_argument(
        "--image-name-key",
        type=str,
        default="image_names",
        help="NPZ key containing image names when --image-dir is used.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    resize_trainer_npz_images(
        input_path=args.input,
        output_path=args.output,
        target_width=args.target_width,
        target_height=args.target_height,
        image_dir=args.image_dir,
        image_name_key=args.image_name_key,
    )


if __name__ == "__main__":
    main()
