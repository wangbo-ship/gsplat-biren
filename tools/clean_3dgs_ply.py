#!/usr/bin/env /usr/bin/python3
"""Clean oversized and spatial-outlier splats from a standard 3DGS PLY file."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gsplat._logging import (  # noqa: E402
    configure_logging,
    format_key_value_table,
    format_table,
    get_logger,
    log_info,
)
from tools.filter_3dgs_ply import (  # noqa: E402
    PlyDocument,
    _finite_mask,
    _point_positions,
    _read_ply,
    _scale_field_names,
    _write_ply,
)


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class CleanOptions:
    remove_large_percent: float = 0.0
    remove_outlier_percent: float = 0.0
    remove_combined_percent: float = 0.0
    auto: bool = True
    auto_large_threshold: float = 3.5
    auto_outlier_threshold: float = 3.5
    auto_combined_threshold: float = 7.0
    auto_max_remove_percent: float = 5.0
    knn: int = 8
    knn_exact_max_points: int = 50_000
    knn_reference_points: int = 20_000
    knn_chunk_size: int = 512
    sample_seed: int = 42
    scale_domain: str = "log"
    large_weight: float = 1.0
    outlier_weight: float = 1.0
    synergy_weight: float = 1.0
    max_report_items: int = 50


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _check_percent(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"{name} must be in [0, 100], got {value}")
    return value


def _scale_metric(doc: PlyDocument, scale_domain: str) -> np.ndarray:
    scale_fields = _scale_field_names(doc.properties)
    scale_values = np.stack(
        [np.asarray(doc.data[name], dtype=np.float64) for name in scale_fields],
        axis=1,
    )
    if scale_domain == "log":
        # gsplat-biren trainer PLY files store the trainable log-scale values.
        scale_values = np.exp(np.clip(scale_values, -60.0, 60.0))
    elif scale_domain == "raw":
        scale_values = np.abs(scale_values)
    else:
        raise ValueError("--scale-domain must be one of: log, raw")
    return scale_values.max(axis=1)


def _robust_upper_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    scores = np.zeros(values.shape, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return scores
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if scale <= 1.0e-12:
        q25, q75 = np.percentile(finite, [25.0, 75.0])
        scale = float(q75 - q25)
    if scale <= 1.0e-12:
        scale = max(abs(median), 1.0) * 1.0e-12
    finite_mask = np.isfinite(values)
    scores[finite_mask] = np.maximum((values[finite_mask] - median) / scale, 0.0)
    return scores


def _top_percent_mask(scores: np.ndarray, percent: float) -> np.ndarray:
    percent = _check_percent(percent, "percent")
    count = int(math.ceil(scores.shape[0] * percent / 100.0))
    if count <= 0 or scores.shape[0] == 0:
        return np.zeros(scores.shape[0], dtype=bool)
    order = np.argsort(scores, kind="stable")
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[order[-count:]] = True
    return mask


def _sample_reference_indices(total: int, max_reference: int, seed: int) -> np.ndarray:
    if total <= max_reference:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(total, size=int(max_reference), replace=False)).astype(
        np.int64
    )


def _knn_distances_scipy(
    points: np.ndarray,
    reference_indices: np.ndarray,
    knn: int,
) -> tuple[np.ndarray, str] | None:
    try:
        from scipy.spatial import cKDTree  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None

    reference_points = points[reference_indices]
    query_k = min(int(knn) + 1, reference_points.shape[0])
    if query_k <= 0:
        return np.zeros(points.shape[0], dtype=np.float64), "empty"
    tree = cKDTree(reference_points)
    distances, ref_positions = tree.query(points, k=query_k)
    distances = np.asarray(distances, dtype=np.float64)
    ref_positions = np.asarray(ref_positions)
    if distances.ndim == 1:
        distances = distances[:, None]
        ref_positions = ref_positions[:, None]

    original_indices = np.arange(points.shape[0], dtype=np.int64)[:, None]
    ref_original_indices = reference_indices[ref_positions]
    distances = distances.copy()
    distances[ref_original_indices == original_indices] = np.inf
    distances.sort(axis=1)
    neighbor_count = min(int(knn), distances.shape[1])
    selected = distances[:, :neighbor_count]
    selected[~np.isfinite(selected)] = np.nan
    mean_distances = np.nanmean(selected, axis=1)
    mean_distances[~np.isfinite(mean_distances)] = 0.0
    mode = "scipy_exact" if reference_indices.shape[0] == points.shape[0] else "scipy_sampled"
    return mean_distances, mode


def _knn_distances_numpy(
    points: np.ndarray,
    reference_indices: np.ndarray,
    knn: int,
    chunk_size: int,
) -> tuple[np.ndarray, str]:
    reference_points = points[reference_indices]
    if points.shape[0] <= 1 or reference_points.shape[0] <= 1:
        return np.zeros(points.shape[0], dtype=np.float64), "empty"

    neighbor_count = min(int(knn), max(reference_points.shape[0] - 1, 1))
    out = np.zeros(points.shape[0], dtype=np.float64)
    chunk_size = max(int(chunk_size), 1)
    ref_original = reference_indices.reshape(1, -1)

    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        chunk = points[start:end]
        diff = chunk[:, None, :] - reference_points[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        original = np.arange(start, end, dtype=np.int64).reshape(-1, 1)
        dist2[ref_original == original] = np.inf
        kth = min(neighbor_count, dist2.shape[1])
        nearest = np.partition(dist2, kth - 1, axis=1)[:, :kth]
        nearest[~np.isfinite(nearest)] = np.nan
        values = np.sqrt(nearest)
        mean_values = np.nanmean(values, axis=1)
        mean_values[~np.isfinite(mean_values)] = 0.0
        out[start:end] = mean_values

    mode = "numpy_exact" if reference_indices.shape[0] == points.shape[0] else "numpy_sampled"
    return out, mode


def _outlier_metric(
    points: np.ndarray,
    *,
    knn: int,
    exact_max_points: int,
    reference_points: int,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, str, int]:
    if knn <= 0:
        raise ValueError(f"--knn must be positive, got {knn}")
    if points.shape[0] <= 1:
        return (
            np.zeros(points.shape[0], dtype=np.float64),
            "empty",
            int(points.shape[0]),
        )
    if exact_max_points < 2:
        raise ValueError("--knn-exact-max-points must be at least 2")
    if reference_points < 2:
        raise ValueError("--knn-reference-points must be at least 2")

    max_reference = (
        points.shape[0] if points.shape[0] <= exact_max_points else reference_points
    )
    reference_indices = _sample_reference_indices(
        points.shape[0],
        min(int(max_reference), points.shape[0]),
        seed,
    )
    scipy_result = _knn_distances_scipy(points, reference_indices, knn)
    if scipy_result is not None:
        distances, mode = scipy_result
    else:
        distances, mode = _knn_distances_numpy(points, reference_indices, knn, chunk_size)
    return distances, mode, int(reference_indices.shape[0])


def _combined_scores(
    large_scores: np.ndarray,
    outlier_scores: np.ndarray,
    *,
    large_weight: float,
    outlier_weight: float,
    synergy_weight: float,
) -> np.ndarray:
    if large_weight < 0.0 or outlier_weight < 0.0 or synergy_weight < 0.0:
        raise ValueError("score weights must be non-negative")
    return (
        float(large_weight) * large_scores
        + float(outlier_weight) * outlier_scores
        + float(synergy_weight) * np.minimum(large_scores, outlier_scores)
    )


def _cap_mask_by_score(
    mask: np.ndarray,
    scores: np.ndarray,
    max_percent: float,
) -> tuple[np.ndarray, bool]:
    max_percent = _check_percent(max_percent, "auto_max_remove_percent")
    if max_percent <= 0.0 or mask.sum() == 0:
        return mask, False
    max_count = int(math.ceil(mask.shape[0] * max_percent / 100.0))
    if max_count <= 0 or mask.sum() <= max_count:
        return mask, False
    selected_indices = np.flatnonzero(mask)
    selected_scores = scores[selected_indices]
    order = np.argsort(selected_scores, kind="stable")
    keep_selected = selected_indices[order[-max_count:]]
    capped = np.zeros(mask.shape[0], dtype=bool)
    capped[keep_selected] = True
    return capped, True


def _score_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    result: dict[str, float | int | None] = {
        "count": int(values.size),
        "finite": int(finite.size),
    }
    if finite.size == 0:
        for key in ("min", "p50", "p90", "p95", "p99", "max"):
            result[key] = None
        return result
    for key, percentile in (
        ("min", 0.0),
        ("p50", 50.0),
        ("p90", 90.0),
        ("p95", 95.0),
        ("p99", 99.0),
        ("max", 100.0),
    ):
        result[key] = float(np.percentile(finite, percentile))
    return result


def _removed_rows(
    *,
    original_indices: np.ndarray,
    points: np.ndarray,
    scale_metric: np.ndarray,
    outlier_metric: np.ndarray,
    large_scores: np.ndarray,
    outlier_scores: np.ndarray,
    combined_scores: np.ndarray,
    reason_masks: dict[str, np.ndarray],
    remove_mask: np.ndarray,
    max_items: int,
) -> list[dict[str, Any]]:
    if max_items <= 0 or remove_mask.sum() == 0:
        return []
    removed = np.flatnonzero(remove_mask)
    order = np.argsort(combined_scores[removed], kind="stable")[::-1]
    rows = []
    for local_index in removed[order[:max_items]]:
        reasons = [
            name for name, mask in reason_masks.items() if bool(mask[local_index])
        ]
        rows.append(
            {
                "vertex_index": int(original_indices[local_index]),
                "xyz": [float(value) for value in points[local_index].tolist()],
                "scale_metric": float(scale_metric[local_index]),
                "outlier_distance": float(outlier_metric[local_index]),
                "large_score": float(large_scores[local_index]),
                "outlier_score": float(outlier_scores[local_index]),
                "combined_score": float(combined_scores[local_index]),
                "reasons": reasons,
            }
        )
    return rows


def _clean_document(
    doc: PlyDocument,
    options: CleanOptions,
) -> tuple[PlyDocument, dict[str, Any]]:
    total = int(doc.data.shape[0])
    finite_mask = _finite_mask(doc)
    valid_data = doc.data[finite_mask]
    valid_doc = PlyDocument(doc.header_lines, doc.format_name, doc.properties, valid_data)
    valid_total = int(valid_data.shape[0])

    if valid_total == 0:
        filtered = PlyDocument(doc.header_lines, doc.format_name, doc.properties, valid_data)
        return filtered, {
            "input_vertices": total,
            "valid_vertices": 0,
            "removed_invalid": total,
            "removed_large_percentile": 0,
            "removed_outlier_percentile": 0,
            "removed_combined_percentile": 0,
            "removed_auto_collaborative": 0,
            "auto_capped": False,
            "removed_total": total,
            "removed_valid_total": 0,
            "kept": 0,
            "reason_overlap": 0,
            "knn_mode": "empty",
            "knn_reference_points": 0,
            "score_summary": {},
            "removed_samples": [],
        }

    points = _point_positions(valid_doc)
    scale_metric = _scale_metric(valid_doc, options.scale_domain)
    outlier_metric, knn_mode, knn_reference_count = _outlier_metric(
        points,
        knn=options.knn,
        exact_max_points=options.knn_exact_max_points,
        reference_points=options.knn_reference_points,
        chunk_size=options.knn_chunk_size,
        seed=options.sample_seed,
    )
    large_scores = _robust_upper_score(scale_metric)
    outlier_scores = _robust_upper_score(outlier_metric)
    combined_scores = _combined_scores(
        large_scores,
        outlier_scores,
        large_weight=options.large_weight,
        outlier_weight=options.outlier_weight,
        synergy_weight=options.synergy_weight,
    )

    reason_masks: dict[str, np.ndarray] = {
        "large_percentile": _top_percent_mask(
            large_scores,
            options.remove_large_percent,
        ),
        "outlier_percentile": _top_percent_mask(
            outlier_scores,
            options.remove_outlier_percent,
        ),
        "combined_percentile": _top_percent_mask(
            combined_scores,
            options.remove_combined_percent,
        ),
    }

    auto_mask = np.zeros(valid_total, dtype=bool)
    auto_capped = False
    if options.auto:
        dual_metric = (
            (large_scores >= float(options.auto_large_threshold))
            & (outlier_scores >= float(options.auto_outlier_threshold))
        )
        combined_metric = combined_scores >= float(options.auto_combined_threshold)
        auto_mask = dual_metric | combined_metric
        auto_mask, auto_capped = _cap_mask_by_score(
            auto_mask,
            combined_scores,
            options.auto_max_remove_percent,
        )
    reason_masks["auto_collaborative"] = auto_mask

    remove_valid = np.zeros(valid_total, dtype=bool)
    for mask in reason_masks.values():
        remove_valid |= mask

    keep_valid = ~remove_valid
    keep_mask = np.zeros(total, dtype=bool)
    keep_mask[np.flatnonzero(finite_mask)] = keep_valid
    filtered_data = doc.data[keep_mask]
    filtered_doc = PlyDocument(
        doc.header_lines,
        doc.format_name,
        doc.properties,
        filtered_data,
    )

    original_valid_indices = np.flatnonzero(finite_mask)
    stats: dict[str, Any] = {
        "input_vertices": total,
        "valid_vertices": valid_total,
        "removed_invalid": total - valid_total,
        "removed_large_percentile": int(reason_masks["large_percentile"].sum()),
        "removed_outlier_percentile": int(reason_masks["outlier_percentile"].sum()),
        "removed_combined_percentile": int(reason_masks["combined_percentile"].sum()),
        "removed_auto_collaborative": int(reason_masks["auto_collaborative"].sum()),
        "auto_capped": bool(auto_capped),
        "removed_total": int(total - filtered_data.shape[0]),
        "removed_valid_total": int(remove_valid.sum()),
        "kept": int(filtered_data.shape[0]),
        "reason_overlap": int(
            sum(int(mask.sum()) for mask in reason_masks.values()) - int(remove_valid.sum())
        ),
        "knn_mode": knn_mode,
        "knn_reference_points": knn_reference_count,
        "score_summary": {
            "scale_metric": _score_summary(scale_metric),
            "outlier_distance": _score_summary(outlier_metric),
            "large_score": _score_summary(large_scores),
            "outlier_score": _score_summary(outlier_scores),
            "combined_score": _score_summary(combined_scores),
        },
        "removed_samples": _removed_rows(
            original_indices=original_valid_indices,
            points=points,
            scale_metric=scale_metric,
            outlier_metric=outlier_metric,
            large_scores=large_scores,
            outlier_scores=outlier_scores,
            combined_scores=combined_scores,
            reason_masks=reason_masks,
            remove_mask=remove_valid,
            max_items=int(options.max_report_items),
        ),
    }
    return filtered_doc, stats


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def clean_3dgs_ply(
    input_path: Path,
    *,
    output_path: Path | None = None,
    report_path: Path | None = None,
    dry_run: bool = False,
    options: CleanOptions | None = None,
) -> dict[str, Any]:
    """Clean a 3DGS PLY file by combining scale and spatial-outlier scores."""
    options = options or CleanOptions()
    doc = _read_ply(input_path)
    filtered_doc, stats = _clean_document(doc, options)
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_cleaned.ply")
    if not dry_run:
        _write_ply(output_path, filtered_doc)

    stats.update(
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "report_path": str(report_path) if report_path is not None else None,
            "dry_run": bool(dry_run),
            "options": {key: value for key, value in options.__dict__.items()},
        }
    )
    if report_path is not None:
        _write_report(report_path, stats)
    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clean standard vertex-only 3DGS PLY files by removing oversized "
            "splats, spatial outliers, and splats selected by a combined "
            "large-plus-outlier score."
        )
    )
    parser.add_argument("ply_path", type=Path, help="Input standard 3DGS PLY file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PLY path. Defaults to <input>_cleaned.ply.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--dry-run",
        type=_parse_bool,
        default=False,
        metavar="true|false",
        help="Compute masks and write the optional report without writing a PLY.",
    )
    parser.add_argument(
        "--auto",
        type=_parse_bool,
        default=True,
        metavar="true|false",
        help="Enable conservative automatic cleanup from collaborative scores.",
    )
    parser.add_argument("--auto-large-threshold", type=float, default=3.5)
    parser.add_argument("--auto-outlier-threshold", type=float, default=3.5)
    parser.add_argument("--auto-combined-threshold", type=float, default=7.0)
    parser.add_argument(
        "--auto-max-remove-percent",
        type=float,
        default=5.0,
        help="Safety cap for automatic cleanup. Use 100 to disable the cap.",
    )
    parser.add_argument("--remove-large-percent", type=float, default=0.0)
    parser.add_argument("--remove-outlier-percent", type=float, default=0.0)
    parser.add_argument("--remove-combined-percent", type=float, default=0.0)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--knn-exact-max-points", type=int, default=50_000)
    parser.add_argument("--knn-reference-points", type=int, default=20_000)
    parser.add_argument("--knn-chunk-size", type=int, default=512)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--scale-domain",
        choices=("log", "raw"),
        default="log",
        help="Interpret scale_* fields as log scales or raw scale magnitudes.",
    )
    parser.add_argument("--large-weight", type=float, default=1.0)
    parser.add_argument("--outlier-weight", type=float, default=1.0)
    parser.add_argument(
        "--synergy-weight",
        type=float,
        default=1.0,
        help="Extra weight applied to min(large_score, outlier_score).",
    )
    parser.add_argument("--max-report-items", type=int, default=50)
    return parser


def _options_from_args(args: argparse.Namespace) -> CleanOptions:
    for name in (
        "remove_large_percent",
        "remove_outlier_percent",
        "remove_combined_percent",
        "auto_max_remove_percent",
    ):
        _check_percent(getattr(args, name), f"--{name.replace('_', '-')}")
    if args.max_report_items < 0:
        raise ValueError("--max-report-items must be non-negative")
    return CleanOptions(
        remove_large_percent=float(args.remove_large_percent),
        remove_outlier_percent=float(args.remove_outlier_percent),
        remove_combined_percent=float(args.remove_combined_percent),
        auto=bool(args.auto),
        auto_large_threshold=float(args.auto_large_threshold),
        auto_outlier_threshold=float(args.auto_outlier_threshold),
        auto_combined_threshold=float(args.auto_combined_threshold),
        auto_max_remove_percent=float(args.auto_max_remove_percent),
        knn=int(args.knn),
        knn_exact_max_points=int(args.knn_exact_max_points),
        knn_reference_points=int(args.knn_reference_points),
        knn_chunk_size=int(args.knn_chunk_size),
        sample_seed=int(args.sample_seed),
        scale_domain=str(args.scale_domain),
        large_weight=float(args.large_weight),
        outlier_weight=float(args.outlier_weight),
        synergy_weight=float(args.synergy_weight),
        max_report_items=int(args.max_report_items),
    )


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    stats = clean_3dgs_ply(
        args.ply_path,
        output_path=args.output,
        report_path=args.report,
        dry_run=bool(args.dry_run),
        options=_options_from_args(args),
    )

    log_info(
        LOGGER,
        format_key_value_table(
            "3DGS PLY Cleanup",
            [
                ("input", stats["input_path"]),
                ("output", "not written" if stats["dry_run"] else stats["output_path"]),
                ("report", stats["report_path"] or "not written"),
                ("input_vertices", stats["input_vertices"]),
                ("kept", stats["kept"]),
                ("removed_invalid", stats["removed_invalid"]),
                ("removed_large_percentile", stats["removed_large_percentile"]),
                ("removed_outlier_percentile", stats["removed_outlier_percentile"]),
                ("removed_combined_percentile", stats["removed_combined_percentile"]),
                ("removed_auto_collaborative", stats["removed_auto_collaborative"]),
                ("removed_total", stats["removed_total"]),
                ("reason_overlap", stats["reason_overlap"]),
                ("knn_mode", stats["knn_mode"]),
                ("knn_reference_points", stats["knn_reference_points"]),
            ],
        ),
    )
    samples = stats.get("removed_samples", [])[:8]
    if samples:
        log_info(
            LOGGER,
            format_table(
                "Top Removed Splats",
                ["idx", "scale", "knn", "large_z", "outlier_z", "combined", "reasons"],
                [
                    [
                        row["vertex_index"],
                        f"{row['scale_metric']:.6g}",
                        f"{row['outlier_distance']:.6g}",
                        f"{row['large_score']:.3g}",
                        f"{row['outlier_score']:.3g}",
                        f"{row['combined_score']:.3g}",
                        ",".join(row["reasons"]),
                    ]
                    for row in samples
                ],
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
