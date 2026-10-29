#!/usr/bin/env /usr/bin/python3
"""Diagnose COLMAP sparse models and optionally remove point outliers."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from gsplat._logging import (
    configure_logging,
    format_key_value_table,
    format_table,
    get_logger,
    log_info,
    log_warning,
)


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class DiagnosticConfig:
    drop_point_distance_percent: float = 0.0
    drop_point_error_percent: float = 0.0
    max_point_distance_mad: float = 0.0
    max_point_error_mad: float = 0.0
    min_point_track_length: int = 0
    drop_camera_distance_percent: float = 0.0
    max_camera_distance_mad: float = 0.0
    min_image_points: int = 0
    min_image_point_ratio: float = 0.0
    init_num_points: int = -1
    test_every: int = 8
    max_listed_items: int = 200


@dataclass(frozen=True)
class CleanConfig:
    clean: bool = False
    clean_images: bool = False
    output_sparse: Path | None = None
    output_format: str = "binary"
    overwrite: bool = False


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def _check_percent(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0 or value > 100.0:
        raise ValueError(f"{name} must be in [0, 100], got {value}")
    return value


def _require_pycolmap():
    try:
        import pycolmap  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError(
            "pycolmap is required for COLMAP diagnostics. It is expected to be "
            "provided by the fixed runtime environment; do not install it from "
            "this tool."
        ) from exc
    return pycolmap


def _resolve_sparse_dir(data_dir: Path, sparse_dir: Path | None) -> Path:
    if sparse_dir is not None:
        return sparse_dir
    nested = data_dir / "sparse" / "0"
    if nested.is_dir():
        return nested
    return data_dir / "sparse"


def _mapping_values(mapping: Mapping[Any, Any]) -> Iterable[Any]:
    if hasattr(mapping, "values"):
        return mapping.values()
    return (mapping[key] for key in mapping)


def _mapping_keys(mapping: Mapping[Any, Any]) -> list[int]:
    return sorted(int(key) for key in mapping.keys())


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isfinite(number):
        return number
    return None


def _json_number(value: Any) -> float | None:
    try:
        return _to_float(value)
    except (TypeError, ValueError):
        return None


def _quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    result: dict[str, float | int | None] = {
        "count": int(values.size),
        "finite": int(finite.size),
    }
    if finite.size == 0:
        for key in ("min", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean"):
            result[key] = None
        return result
    percentile_names = [
        ("min", 0.0),
        ("p1", 1.0),
        ("p5", 5.0),
        ("p10", 10.0),
        ("p25", 25.0),
        ("p50", 50.0),
        ("p75", 75.0),
        ("p90", 90.0),
        ("p95", 95.0),
        ("p99", 99.0),
        ("max", 100.0),
    ]
    for name, percentile in percentile_names:
        result[name] = float(np.percentile(finite, percentile))
    result["mean"] = float(np.mean(finite))
    return result


def _robust_scores(values: np.ndarray) -> tuple[np.ndarray, float | None, float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    scores = np.full(values.shape, np.inf, dtype=np.float64)
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    if finite.size == 0:
        return scores, None, None
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if scale <= 1.0e-12:
        finite_scores = np.zeros_like(finite, dtype=np.float64)
        finite_scores[np.abs(finite - median) > 1.0e-12] = np.inf
    else:
        finite_scores = np.abs(finite - median) / scale
    scores[finite_mask] = finite_scores
    return scores, median, scale


def _select_top_percent(
    ids: Sequence[int],
    values: np.ndarray,
    percent: float,
    *,
    largest: bool,
) -> set[int]:
    percent = _check_percent(percent, "percent")
    if percent <= 0.0 or not ids:
        return set()
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if valid_indices.size == 0:
        return set()
    count = int(math.ceil(valid_indices.size * percent / 100.0))
    if count <= 0:
        return set()
    order = np.argsort(values[valid_indices], kind="mergesort")
    selected = valid_indices[order[-count:] if largest else order[:count]]
    return {int(ids[index]) for index in selected}


def _round(value: Any, digits: int = 6) -> float | None:
    number = _json_number(value)
    if number is None:
        return None
    return round(number, digits)


def _center_of_image(image: Any) -> np.ndarray:
    if hasattr(image, "projection_center"):
        center = image.projection_center()
    else:
        cam_from_world = image.cam_from_world().matrix()
        rotation = np.asarray(cam_from_world[:, :3], dtype=np.float64)
        translation = np.asarray(cam_from_world[:, 3], dtype=np.float64)
        center = -rotation.T @ translation
    return np.asarray(center, dtype=np.float64).reshape(3)


def _num_points2d(image: Any) -> int:
    value = getattr(image, "num_points2D")
    return int(value() if callable(value) else value)


def _num_points3d(image: Any) -> int:
    value = getattr(image, "num_points3D")
    return int(value() if callable(value) else value)


def _track_length(point: Any) -> int:
    track = point.track
    if hasattr(track, "length"):
        return int(track.length())
    if hasattr(track, "elements"):
        return int(len(track.elements))
    return 0


def _camera_scene_scale(centers: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    centers = np.asarray(centers, dtype=np.float64)
    if mask is not None:
        centers = centers[np.asarray(mask, dtype=bool)]
    if centers.size == 0:
        return None
    finite_mask = np.isfinite(centers).all(axis=1)
    centers = centers[finite_mask]
    if centers.size == 0:
        return None
    scene_center = np.mean(centers, axis=0)
    return float(np.max(np.linalg.norm(centers - scene_center, axis=1)))


def _split_name(sorted_index: int, test_every: int) -> str:
    if test_every <= 0:
        return "all"
    return "eval" if sorted_index % test_every == 0 else "train"


def diagnose_reconstruction(reconstruction: Any, config: DiagnosticConfig) -> dict[str, Any]:
    point_ids = _mapping_keys(reconstruction.points3D)
    image_ids = sorted(
        [int(image_id) for image_id in reconstruction.reg_image_ids()],
        key=lambda image_id: str(reconstruction.image(image_id).name),
    )

    xyzs = np.empty((len(point_ids), 3), dtype=np.float64)
    point_errors = np.empty((len(point_ids),), dtype=np.float64)
    track_lengths = np.empty((len(point_ids),), dtype=np.int64)
    for index, point_id in enumerate(point_ids):
        point = reconstruction.points3D[point_id]
        xyzs[index] = np.asarray(point.xyz, dtype=np.float64).reshape(3)
        point_errors[index] = float(point.error)
        track_lengths[index] = _track_length(point)

    finite_xyz = np.isfinite(xyzs).all(axis=1)
    if finite_xyz.any():
        point_center = np.median(xyzs[finite_xyz], axis=0)
        point_distances = np.linalg.norm(xyzs - point_center.reshape(1, 3), axis=1)
    else:
        point_center = np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
        point_distances = np.full((len(point_ids),), np.inf, dtype=np.float64)
    point_distance_scores, point_distance_median, point_distance_scale = _robust_scores(
        point_distances
    )
    point_error_scores, point_error_median, point_error_scale = _robust_scores(point_errors)

    point_reasons: dict[int, list[str]] = {point_id: [] for point_id in point_ids}
    for index, point_id in enumerate(point_ids):
        if not finite_xyz[index] or not math.isfinite(float(point_errors[index])):
            point_reasons[point_id].append("nonfinite")
        if (
            config.min_point_track_length > 0
            and int(track_lengths[index]) < config.min_point_track_length
        ):
            point_reasons[point_id].append("short_track")
        if (
            config.max_point_distance_mad > 0.0
            and point_distance_scores[index] > config.max_point_distance_mad
        ):
            point_reasons[point_id].append("distance_mad_outlier")
        if (
            config.max_point_error_mad > 0.0
            and point_error_scores[index] > config.max_point_error_mad
        ):
            point_reasons[point_id].append("error_mad_outlier")

    distance_drop = _select_top_percent(
        point_ids,
        point_distances,
        config.drop_point_distance_percent,
        largest=True,
    )
    for point_id in sorted(distance_drop):
        point_reasons[point_id].append("distance_percent_outlier")

    error_drop = _select_top_percent(
        point_ids,
        point_errors,
        config.drop_point_error_percent,
        largest=True,
    )
    for point_id in sorted(error_drop):
        point_reasons[point_id].append("error_percent_outlier")

    dropped_point_ids = sorted(
        point_id for point_id, reasons in point_reasons.items() if reasons
    )

    centers = np.empty((len(image_ids), 3), dtype=np.float64)
    image_observations = np.empty((len(image_ids),), dtype=np.int64)
    image_points2d = np.empty((len(image_ids),), dtype=np.int64)
    image_names = []
    for index, image_id in enumerate(image_ids):
        image = reconstruction.image(image_id)
        image_names.append(str(image.name))
        centers[index] = _center_of_image(image)
        image_observations[index] = _num_points3d(image)
        image_points2d[index] = _num_points2d(image)

    finite_centers = np.isfinite(centers).all(axis=1)
    if finite_centers.any():
        camera_center = np.median(centers[finite_centers], axis=0)
        camera_distances = np.linalg.norm(centers - camera_center.reshape(1, 3), axis=1)
    else:
        camera_center = np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
        camera_distances = np.full((len(image_ids),), np.inf, dtype=np.float64)
    camera_distance_scores, camera_distance_median, camera_distance_scale = _robust_scores(
        camera_distances
    )
    image_ratios = image_observations.astype(np.float64) / np.maximum(
        image_points2d, 1
    ).astype(np.float64)

    image_reasons: dict[int, list[str]] = {image_id: [] for image_id in image_ids}
    for index, image_id in enumerate(image_ids):
        if not finite_centers[index]:
            image_reasons[image_id].append("nonfinite_pose")
        if config.min_image_points > 0 and int(image_observations[index]) < config.min_image_points:
            image_reasons[image_id].append("low_observations")
        if (
            config.min_image_point_ratio > 0.0
            and float(image_ratios[index]) < config.min_image_point_ratio
        ):
            image_reasons[image_id].append("low_triangulated_ratio")
        if (
            config.max_camera_distance_mad > 0.0
            and camera_distance_scores[index] > config.max_camera_distance_mad
        ):
            image_reasons[image_id].append("camera_distance_mad_outlier")

    camera_drop = _select_top_percent(
        image_ids,
        camera_distances,
        config.drop_camera_distance_percent,
        largest=True,
    )
    for image_id in sorted(camera_drop):
        image_reasons[image_id].append("camera_distance_percent_outlier")

    flagged_image_ids = sorted(image_id for image_id, reasons in image_reasons.items() if reasons)
    kept_camera_mask = np.asarray(
        [image_id not in set(flagged_image_ids) for image_id in image_ids],
        dtype=bool,
    )

    point_index_by_id = {point_id: index for index, point_id in enumerate(point_ids)}
    image_index_by_id = {image_id: index for index, image_id in enumerate(image_ids)}

    point_rows = []
    for point_id in dropped_point_ids:
        index = point_index_by_id[point_id]
        point_rows.append(
            {
                "point3D_id": int(point_id),
                "xyz": [_round(value) for value in xyzs[index].tolist()],
                "distance": _round(point_distances[index]),
                "reprojection_error": _round(point_errors[index]),
                "track_length": int(track_lengths[index]),
                "distance_mad_score": _round(point_distance_scores[index]),
                "error_mad_score": _round(point_error_scores[index]),
                "reasons": sorted(set(point_reasons[point_id])),
            }
        )
    point_rows.sort(
        key=lambda row: (
            row["distance"] if row["distance"] is not None else float("inf"),
            row["reprojection_error"] if row["reprojection_error"] is not None else float("inf"),
        ),
        reverse=True,
    )

    image_rows = []
    for image_id in flagged_image_ids:
        index = image_index_by_id[image_id]
        image_rows.append(
            {
                "image_id": int(image_id),
                "name": image_names[index],
                "sorted_index": int(index),
                "split": _split_name(index, config.test_every),
                "triangulated_points": int(image_observations[index]),
                "points2D": int(image_points2d[index]),
                "triangulated_ratio": _round(image_ratios[index]),
                "camera_center": [_round(value) for value in centers[index].tolist()],
                "camera_distance": _round(camera_distances[index]),
                "camera_distance_mad_score": _round(camera_distance_scores[index]),
                "reasons": sorted(set(image_reasons[image_id])),
            }
        )
    image_rows.sort(
        key=lambda row: (
            row["camera_distance"] if row["camera_distance"] is not None else float("inf"),
            -int(row["triangulated_points"]),
        ),
        reverse=True,
    )

    real_point_fraction = None
    if config.init_num_points > 0 and point_ids:
        real_point_fraction = min(float(config.init_num_points), float(len(point_ids))) / float(
            len(point_ids)
        )

    summary = {
        "num_cameras": int(len(reconstruction.cameras)),
        "num_registered_images": int(len(image_ids)),
        "num_points3D": int(len(point_ids)),
        "mean_reprojection_error": _round(
            getattr(reconstruction, "compute_mean_reprojection_error")()
            if hasattr(reconstruction, "compute_mean_reprojection_error")
            else None
        ),
        "point_center_median": [_round(value) for value in point_center.tolist()],
        "camera_center_median": [_round(value) for value in camera_center.tolist()],
        "camera_scene_scale_all": _round(_camera_scene_scale(centers)),
        "camera_scene_scale_without_flagged": _round(
            _camera_scene_scale(centers, kept_camera_mask)
        ),
        "flagged_images": int(len(flagged_image_ids)),
        "dropped_point_candidates": int(len(dropped_point_ids)),
        "init_num_points": int(config.init_num_points),
        "init_point_fraction": _round(real_point_fraction),
    }

    return {
        "summary": summary,
        "thresholds": {
            "drop_point_distance_percent": float(config.drop_point_distance_percent),
            "drop_point_error_percent": float(config.drop_point_error_percent),
            "max_point_distance_mad": float(config.max_point_distance_mad),
            "max_point_error_mad": float(config.max_point_error_mad),
            "min_point_track_length": int(config.min_point_track_length),
            "drop_camera_distance_percent": float(config.drop_camera_distance_percent),
            "max_camera_distance_mad": float(config.max_camera_distance_mad),
            "min_image_points": int(config.min_image_points),
            "min_image_point_ratio": float(config.min_image_point_ratio),
            "test_every": int(config.test_every),
        },
        "robust_stats": {
            "point_distance_median": _round(point_distance_median),
            "point_distance_mad_scale": _round(point_distance_scale),
            "point_error_median": _round(point_error_median),
            "point_error_mad_scale": _round(point_error_scale),
            "camera_distance_median": _round(camera_distance_median),
            "camera_distance_mad_scale": _round(camera_distance_scale),
        },
        "quantiles": {
            "point_distance": _quantiles(point_distances),
            "point_reprojection_error": _quantiles(point_errors),
            "point_track_length": _quantiles(track_lengths),
            "image_triangulated_points": _quantiles(image_observations),
            "image_triangulated_ratio": _quantiles(image_ratios),
            "camera_distance": _quantiles(camera_distances),
        },
        "flagged_images": image_rows[: max(0, int(config.max_listed_items))],
        "dropped_points": point_rows[: max(0, int(config.max_listed_items))],
        "all_flagged_image_ids": flagged_image_ids,
        "all_dropped_point_ids": dropped_point_ids,
        "all_flagged_image_names": [
            image_names[image_index_by_id[image_id]] for image_id in flagged_image_ids
        ],
        "_point_reasons": point_reasons,
        "_image_reasons": image_reasons,
    }


def _manifest_dir(report_path: Path | None, clean_config: CleanConfig) -> Path | None:
    if report_path is not None:
        return report_path.parent
    if clean_config.output_sparse is not None:
        return clean_config.output_sparse.parent
    return None


def _write_text_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_manifests(
    diagnostics: dict[str, Any],
    report_path: Path | None,
    clean_config: CleanConfig,
) -> dict[str, str]:
    directory = _manifest_dir(report_path, clean_config)
    if directory is None:
        return {}
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    bad_images = diagnostics["flagged_images"]
    if diagnostics["all_flagged_image_ids"]:
        path = directory / "bad_images.txt"
        rows = ["image_id\tname\treasons"]
        name_by_id = dict(
            zip(
                diagnostics["all_flagged_image_ids"],
                diagnostics["all_flagged_image_names"],
            )
        )
        for row in bad_images:
            rows.append(
                f"{row['image_id']}\t{row['name']}\t{','.join(row['reasons'])}"
            )
        listed = {int(row["image_id"]) for row in bad_images}
        image_reasons = diagnostics["_image_reasons"]
        for image_id in diagnostics["all_flagged_image_ids"]:
            if int(image_id) in listed:
                continue
            rows.append(
                f"{image_id}\t{name_by_id[int(image_id)]}\t"
                f"{','.join(image_reasons[int(image_id)])}"
            )
        _write_text_lines(path, rows)
        paths["bad_images"] = str(path)

    dropped_points = diagnostics["dropped_points"]
    if diagnostics["all_dropped_point_ids"]:
        path = directory / "dropped_points.txt"
        rows = ["point3D_id\treasons"]
        for row in dropped_points:
            rows.append(f"{row['point3D_id']}\t{','.join(row['reasons'])}")
        listed = {int(row["point3D_id"]) for row in dropped_points}
        point_reasons = diagnostics["_point_reasons"]
        for point_id in diagnostics["all_dropped_point_ids"]:
            if int(point_id) in listed:
                continue
            rows.append(f"{point_id}\t{','.join(point_reasons[int(point_id)])}")
        _write_text_lines(path, rows)
        paths["dropped_points"] = str(path)

    return paths


def _strip_internal_keys(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in diagnostics.items()
        if not key.startswith("_") and not key.startswith("all_")
    } | {
        "all_flagged_image_ids": diagnostics["all_flagged_image_ids"],
        "all_flagged_image_names": diagnostics["all_flagged_image_names"],
        "all_dropped_point_ids": diagnostics["all_dropped_point_ids"],
    }


def write_report(
    report_path: Path,
    diagnostics: dict[str, Any],
    *,
    input_info: dict[str, Any],
    manifest_paths: dict[str, str],
    clean_result: dict[str, Any] | None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _strip_internal_keys(diagnostics)
    payload["input"] = input_info
    payload["manifests"] = manifest_paths
    if clean_result is not None:
        payload["clean_result"] = clean_result
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _prepare_output_dir(output_sparse: Path, overwrite: bool) -> None:
    colmap_names = {
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "rigs.bin",
        "frames.bin",
        "cameras.txt",
        "images.txt",
        "points3D.txt",
        "rigs.txt",
        "frames.txt",
    }
    if output_sparse.exists():
        existing_colmap_files = [name for name in colmap_names if (output_sparse / name).exists()]
        if existing_colmap_files and not overwrite:
            raise FileExistsError(
                f"{output_sparse} already contains COLMAP files. Pass --overwrite true "
                "to replace them."
            )
        if existing_colmap_files and overwrite:
            for name in existing_colmap_files:
                (output_sparse / name).unlink()
    output_sparse.mkdir(parents=True, exist_ok=True)


def clean_reconstruction(
    reconstruction: Any,
    diagnostics: dict[str, Any],
    clean_config: CleanConfig,
) -> dict[str, Any]:
    if not clean_config.clean:
        return {"cleaned": False}
    if clean_config.output_sparse is None:
        raise ValueError("--output-sparse is required when --clean true")
    if clean_config.output_format not in {"binary", "text", "both"}:
        raise ValueError("--output-format must be one of: binary, text, both")

    _prepare_output_dir(clean_config.output_sparse, clean_config.overwrite)

    requested_point_ids = [int(point_id) for point_id in diagnostics["all_dropped_point_ids"]]
    deleted_point_ids = []
    for point_id in requested_point_ids:
        if reconstruction.exists_point3D(point_id):
            reconstruction.delete_point3D(point_id)
            deleted_point_ids.append(point_id)

    requested_image_ids = [int(image_id) for image_id in diagnostics["all_flagged_image_ids"]]
    deregistered_image_ids = []
    if clean_config.clean_images:
        for image_id in requested_image_ids:
            if not reconstruction.exists_image(image_id):
                continue
            image = reconstruction.image(image_id)
            frame_id = int(image.frame_id)
            reconstruction.deregister_frame(frame_id)
            deregistered_image_ids.append(image_id)

    if clean_config.output_format in {"binary", "both"}:
        reconstruction.write(clean_config.output_sparse)
    if clean_config.output_format in {"text", "both"}:
        reconstruction.write_text(clean_config.output_sparse)

    return {
        "cleaned": True,
        "output_sparse": str(clean_config.output_sparse),
        "output_format": clean_config.output_format,
        "deleted_points": len(deleted_point_ids),
        "deregistered_images": len(deregistered_image_ids),
        "clean_images": bool(clean_config.clean_images),
    }


def _format_float(value: Any) -> str:
    number = _json_number(value)
    if number is None:
        return "None"
    if abs(number) >= 1000.0:
        return f"{number:.3f}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def log_diagnostics(
    diagnostics: dict[str, Any],
    *,
    data_dir: Path,
    sparse_dir: Path,
    report_path: Path | None,
    manifest_paths: dict[str, str],
    clean_result: dict[str, Any] | None,
) -> None:
    summary = diagnostics["summary"]
    log_info(
        LOGGER,
        format_key_value_table(
            "COLMAP Diagnostics",
            [
                ("data", data_dir),
                ("sparse_dir", sparse_dir),
                ("registered_images", summary["num_registered_images"]),
                ("points3D", summary["num_points3D"]),
                ("mean_reprojection_error", _format_float(summary["mean_reprojection_error"])),
                ("camera_scene_scale_all", _format_float(summary["camera_scene_scale_all"])),
                (
                    "camera_scene_scale_without_flagged",
                    _format_float(summary["camera_scene_scale_without_flagged"]),
                ),
                ("flagged_images", summary["flagged_images"]),
                ("dropped_point_candidates", summary["dropped_point_candidates"]),
                ("init_point_fraction", _format_float(summary["init_point_fraction"])),
                ("report", report_path if report_path is not None else "not written"),
            ],
        ),
    )

    quantiles = diagnostics["quantiles"]
    log_info(
        LOGGER,
        format_table(
            "Quality Quantiles",
            ["metric", "p50", "p90", "p99", "max"],
            [
                [
                    "point_distance",
                    _format_float(quantiles["point_distance"]["p50"]),
                    _format_float(quantiles["point_distance"]["p90"]),
                    _format_float(quantiles["point_distance"]["p99"]),
                    _format_float(quantiles["point_distance"]["max"]),
                ],
                [
                    "point_error",
                    _format_float(quantiles["point_reprojection_error"]["p50"]),
                    _format_float(quantiles["point_reprojection_error"]["p90"]),
                    _format_float(quantiles["point_reprojection_error"]["p99"]),
                    _format_float(quantiles["point_reprojection_error"]["max"]),
                ],
                [
                    "track_length",
                    _format_float(quantiles["point_track_length"]["p50"]),
                    _format_float(quantiles["point_track_length"]["p90"]),
                    _format_float(quantiles["point_track_length"]["p99"]),
                    _format_float(quantiles["point_track_length"]["max"]),
                ],
                [
                    "image_points",
                    _format_float(quantiles["image_triangulated_points"]["p50"]),
                    _format_float(quantiles["image_triangulated_points"]["p90"]),
                    _format_float(quantiles["image_triangulated_points"]["p99"]),
                    _format_float(quantiles["image_triangulated_points"]["max"]),
                ],
                [
                    "camera_distance",
                    _format_float(quantiles["camera_distance"]["p50"]),
                    _format_float(quantiles["camera_distance"]["p90"]),
                    _format_float(quantiles["camera_distance"]["p99"]),
                    _format_float(quantiles["camera_distance"]["max"]),
                ],
            ],
        ),
    )

    image_rows = diagnostics["flagged_images"][:12]
    if image_rows:
        log_warning(
            LOGGER,
            format_table(
                "Flagged Images",
                ["name", "split", "points", "ratio", "cam_dist", "reasons"],
                [
                    [
                        row["name"],
                        row["split"],
                        row["triangulated_points"],
                        _format_float(row["triangulated_ratio"]),
                        _format_float(row["camera_distance"]),
                        ",".join(row["reasons"]),
                    ]
                    for row in image_rows
                ],
            ),
        )
    else:
        log_info(LOGGER, "No images matched the configured low-quality thresholds.")

    point_rows = diagnostics["dropped_points"][:12]
    if point_rows:
        log_warning(
            LOGGER,
            format_table(
                "Point Outlier Candidates",
                ["point3D_id", "distance", "error", "track", "reasons"],
                [
                    [
                        row["point3D_id"],
                        _format_float(row["distance"]),
                        _format_float(row["reprojection_error"]),
                        row["track_length"],
                        ",".join(row["reasons"]),
                    ]
                    for row in point_rows
                ],
            ),
        )
    else:
        log_info(LOGGER, "No 3D points matched the configured outlier thresholds.")

    if summary["init_point_fraction"] is not None and summary["init_point_fraction"] < 0.5:
        log_warning(
            LOGGER,
            "Initial Gaussian count covers only "
            f"{summary['init_point_fraction']:.3f} of the COLMAP point cloud. "
            "For SFM initialization, consider --init-num-points -1 or a value "
            "closer to the sparse point count after cleaning.",
        )

    if manifest_paths:
        log_info(
            LOGGER,
            format_key_value_table(
                "Diagnostic Manifests",
                [(key, value) for key, value in sorted(manifest_paths.items())],
            ),
        )
    if clean_result is not None and clean_result.get("cleaned"):
        log_info(
            LOGGER,
            format_key_value_table(
                "Clean Result",
                [
                    ("output_sparse", clean_result["output_sparse"]),
                    ("output_format", clean_result["output_format"]),
                    ("deleted_points", clean_result["deleted_points"]),
                    ("deregistered_images", clean_result["deregistered_images"]),
                    ("clean_images", clean_result["clean_images"]),
                ],
            ),
        )


def _default_report_path(data_dir: Path) -> Path:
    return Path(".agents/output/colmap-diagnostics") / f"{data_dir.name}_report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose a COLMAP sparse model and optionally remove 3D point "
            "outliers by robust distance, reprojection error, or track length."
        )
    )
    parser.add_argument("--data", type=Path, required=True, help="COLMAP dataset root")
    parser.add_argument("--sparse-dir", type=Path, default=None, help="COLMAP sparse dir")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path. Defaults to .agents/output/colmap-diagnostics/<dataset>_report.json",
    )
    parser.add_argument("--write-manifests", type=_parse_bool, default=True)
    parser.add_argument("--clean", type=_parse_bool, default=False)
    parser.add_argument(
        "--clean-images",
        type=_parse_bool,
        default=False,
        help="Deregister flagged images when writing a cleaned sparse model.",
    )
    parser.add_argument("--output-sparse", type=Path, default=None)
    parser.add_argument(
        "--output-format",
        choices=("binary", "text", "both"),
        default="binary",
        help="COLMAP format to write when --clean true.",
    )
    parser.add_argument("--overwrite", type=_parse_bool, default=False)
    parser.add_argument("--drop-point-distance-percent", type=float, default=0.0)
    parser.add_argument("--drop-point-error-percent", type=float, default=0.0)
    parser.add_argument("--max-point-distance-mad", type=float, default=0.0)
    parser.add_argument("--max-point-error-mad", type=float, default=0.0)
    parser.add_argument("--min-point-track-length", type=int, default=0)
    parser.add_argument("--drop-camera-distance-percent", type=float, default=0.0)
    parser.add_argument("--max-camera-distance-mad", type=float, default=0.0)
    parser.add_argument("--min-image-points", type=int, default=0)
    parser.add_argument("--min-image-point-ratio", type=float, default=0.0)
    parser.add_argument("--init-num-points", type=int, default=-1)
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--max-listed-items", type=int, default=200)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    for name in (
        "drop_point_distance_percent",
        "drop_point_error_percent",
        "drop_camera_distance_percent",
    ):
        _check_percent(getattr(args, name), f"--{name.replace('_', '-')}")
    if args.min_point_track_length < 0:
        raise ValueError("--min-point-track-length must be non-negative")
    if args.min_image_points < 0:
        raise ValueError("--min-image-points must be non-negative")
    if args.min_image_point_ratio < 0.0 or args.min_image_point_ratio > 1.0:
        raise ValueError("--min-image-point-ratio must be in [0, 1]")

    data_dir = args.data
    sparse_dir = _resolve_sparse_dir(data_dir, args.sparse_dir)
    if not sparse_dir.is_dir():
        raise FileNotFoundError(f"COLMAP sparse directory does not exist: {sparse_dir}")

    pycolmap = _require_pycolmap()
    reconstruction = pycolmap.Reconstruction(sparse_dir)

    diagnostic_config = DiagnosticConfig(
        drop_point_distance_percent=float(args.drop_point_distance_percent),
        drop_point_error_percent=float(args.drop_point_error_percent),
        max_point_distance_mad=float(args.max_point_distance_mad),
        max_point_error_mad=float(args.max_point_error_mad),
        min_point_track_length=int(args.min_point_track_length),
        drop_camera_distance_percent=float(args.drop_camera_distance_percent),
        max_camera_distance_mad=float(args.max_camera_distance_mad),
        min_image_points=int(args.min_image_points),
        min_image_point_ratio=float(args.min_image_point_ratio),
        init_num_points=int(args.init_num_points),
        test_every=int(args.test_every),
        max_listed_items=int(args.max_listed_items),
    )
    clean_config = CleanConfig(
        clean=bool(args.clean),
        clean_images=bool(args.clean_images),
        output_sparse=args.output_sparse,
        output_format=str(args.output_format),
        overwrite=bool(args.overwrite),
    )
    report_path = args.report if args.report is not None else _default_report_path(data_dir)

    diagnostics = diagnose_reconstruction(reconstruction, diagnostic_config)
    manifest_paths = (
        write_manifests(diagnostics, report_path, clean_config)
        if args.write_manifests
        else {}
    )
    clean_result = clean_reconstruction(reconstruction, diagnostics, clean_config)
    write_report(
        report_path,
        diagnostics,
        input_info={
            "data": str(data_dir),
            "sparse_dir": str(sparse_dir),
            "pycolmap_version": getattr(pycolmap, "__version__", None),
        },
        manifest_paths=manifest_paths,
        clean_result=clean_result,
    )
    log_diagnostics(
        diagnostics,
        data_dir=data_dir,
        sparse_dir=sparse_dir,
        report_path=report_path,
        manifest_paths=manifest_paths,
        clean_result=clean_result,
    )


if __name__ == "__main__":
    main()
