"""Filter unreasonable 3DGS splats from a standard PLY file.

The script supports two percentile-based filters:
large splats ranked by their scale fields, and isolated splats ranked by
local k-nearest-neighbor spacing. Rows with non-finite numeric values are
always dropped.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from gsplat._logging import format_key_value_table, get_logger, log_info


LOGGER = get_logger(__name__)


_PLY_SCALAR_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

_PLY_FLOAT_TYPES = {"float", "float32", "double", "float64"}
_PLY_INT_TYPES = {
    "char",
    "int8",
    "uchar",
    "uint8",
    "short",
    "int16",
    "ushort",
    "uint16",
    "int",
    "int32",
    "uint",
    "uint32",
}


@dataclass(frozen=True)
class PlyProperty:
    dtype_name: str
    name: str


@dataclass(frozen=True)
class PlyDocument:
    header_lines: tuple[str, ...]
    format_name: str
    properties: tuple[PlyProperty, ...]
    data: np.ndarray


def _structured_dtype(properties: Sequence[PlyProperty], *, byteorder: str | None) -> np.dtype:
    fields = []
    for prop in properties:
        if prop.dtype_name not in _PLY_SCALAR_TYPES:
            raise ValueError(f"unsupported PLY property type: {prop.dtype_name!r}")
        field_dtype = np.dtype(_PLY_SCALAR_TYPES[prop.dtype_name])
        if byteorder is not None:
            field_dtype = field_dtype.newbyteorder(byteorder)
        fields.append((prop.name, field_dtype))
    return np.dtype(fields)


def _read_ply(path: Path) -> PlyDocument:
    with path.open("rb") as handle:
        magic = handle.readline().decode("ascii", errors="strict").strip()
        if magic != "ply":
            raise ValueError(f"{path} is not a PLY file")

        header_lines = [magic]
        format_line = handle.readline().decode("ascii", errors="strict").strip()
        header_lines.append(format_line)
        format_tokens = format_line.split()
        if len(format_tokens) != 3 or format_tokens[0] != "format":
            raise ValueError("invalid PLY format line")
        format_name = format_tokens[1]
        if format_name not in {"ascii", "binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"unsupported PLY format: {format_name!r}")

        elements: list[tuple[str, int, list[PlyProperty]]] = []
        current_name: str | None = None
        current_count = 0
        current_props: list[PlyProperty] = []

        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError("unexpected EOF while reading PLY header")
            line = raw.decode("ascii", errors="strict").rstrip("\r\n")
            header_lines.append(line)
            stripped = line.strip()
            if stripped == "end_header":
                if current_name is not None:
                    elements.append((current_name, current_count, current_props))
                break
            if not stripped or stripped.startswith("comment") or stripped.startswith("obj_info"):
                continue

            tokens = stripped.split()
            keyword = tokens[0]
            if keyword == "element":
                if len(tokens) != 3:
                    raise ValueError(f"invalid element line: {line!r}")
                if current_name is not None:
                    elements.append((current_name, current_count, current_props))
                current_name = tokens[1]
                current_count = int(tokens[2])
                current_props = []
            elif keyword == "property":
                if current_name is None:
                    raise ValueError("property line appears before any element")
                if len(tokens) == 3:
                    current_props.append(PlyProperty(tokens[1], tokens[2]))
                elif len(tokens) >= 4 and tokens[1] == "list":
                    raise ValueError("PLY list properties are not supported")
                else:
                    raise ValueError(f"invalid property line: {line!r}")
            else:
                raise ValueError(f"unsupported PLY header line: {line!r}")

        if len(elements) != 1 or elements[0][0] != "vertex":
            raise ValueError(
                "this tool only supports standard 3DGS PLY files with a single vertex element"
            )

        _, vertex_count, properties = elements[0]
        if format_name == "ascii":
            dtype = _structured_dtype(properties, byteorder=None)
            if vertex_count == 0:
                data = np.empty(0, dtype=dtype)
            else:
                data = np.loadtxt(handle, dtype=dtype, max_rows=vertex_count)
                data = np.atleast_1d(data)
        else:
            byteorder = "<" if format_name == "binary_little_endian" else ">"
            dtype = _structured_dtype(properties, byteorder=byteorder)
            if vertex_count == 0:
                data = np.empty(0, dtype=dtype)
            else:
                payload = handle.read(vertex_count * dtype.itemsize)
                if len(payload) != vertex_count * dtype.itemsize:
                    raise ValueError(
                        f"expected {vertex_count} vertices, read fewer bytes from {path}"
                    )
                data = np.frombuffer(payload, dtype=dtype, count=vertex_count).copy()

    return PlyDocument(
        header_lines=tuple(header_lines),
        format_name=format_name,
        properties=tuple(properties),
        data=data,
    )


def _scale_field_names(properties: Sequence[PlyProperty]) -> list[str]:
    scale_fields = [prop.name for prop in properties if prop.name.startswith("scale_")]
    if not scale_fields:
        raise ValueError("the PLY file does not contain scale_0/scale_1/scale_2 fields")

    def _sort_key(name: str) -> tuple[int, str]:
        suffix = name.split("_", 1)[1]
        return (int(suffix) if suffix.isdigit() else math.inf, name)

    return sorted(scale_fields, key=_sort_key)


def _point_positions(doc: PlyDocument) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(doc.data["x"], dtype=np.float64),
            np.asarray(doc.data["y"], dtype=np.float64),
            np.asarray(doc.data["z"], dtype=np.float64),
        ]
    )


def _large_point_scores(doc: PlyDocument) -> np.ndarray:
    scale_fields = _scale_field_names(doc.properties)
    scale_values = np.stack([np.asarray(doc.data[name], dtype=np.float64) for name in scale_fields], axis=1)
    return scale_values.max(axis=1)


def _isolated_point_scores(points: np.ndarray, isolated_knn: int) -> np.ndarray:
    if isolated_knn <= 0:
        raise ValueError(f"isolated_knn must be positive, got {isolated_knn}")
    if points.shape[0] <= 1:
        return np.zeros(points.shape[0], dtype=np.float64)

    neighbor_count = min(int(isolated_knn), points.shape[0] - 1)
    if neighbor_count <= 0:
        return np.zeros(points.shape[0], dtype=np.float64)

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        points = np.asarray(points, dtype=np.float64)
        count = int(points.shape[0])
        scores = np.empty(count, dtype=np.float64)
        norms = np.einsum("ij,ij->i", points, points)
        target_pairs = 2_000_000
        chunk_size = max(1, min(count, target_pairs // max(count, 1)))
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            dist2 = (
                norms[start:stop, None]
                + norms[None, :]
                - 2.0 * (points[start:stop] @ points.T)
            )
            np.maximum(dist2, 0.0, out=dist2)
            row_ids = np.arange(start, stop)
            dist2[np.arange(stop - start), row_ids] = np.inf
            nearest = np.partition(dist2, kth=neighbor_count - 1, axis=1)[
                :, :neighbor_count
            ]
            scores[start:stop] = np.sqrt(nearest).mean(axis=1)
        return scores

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=neighbor_count + 1)
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim == 1:
        distances = distances[:, None]
    return distances[:, 1:].mean(axis=1)


def _top_percent_mask(scores: np.ndarray, percent: float) -> np.ndarray:
    if not 0.0 <= percent <= 100.0:
        raise ValueError(f"percent must be in [0, 100], got {percent}")
    count = int(math.ceil(scores.shape[0] * (percent / 100.0)))
    if count <= 0 or scores.shape[0] == 0:
        return np.zeros(scores.shape[0], dtype=bool)
    order = np.argsort(scores, kind="stable")
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[order[-count:]] = True
    return mask


def _finite_mask(doc: PlyDocument) -> np.ndarray:
    if doc.data.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    mask = np.ones(doc.data.shape[0], dtype=bool)
    for prop in doc.properties:
        if prop.dtype_name in _PLY_FLOAT_TYPES | _PLY_INT_TYPES:
            mask &= np.isfinite(doc.data[prop.name])
    return mask


def _filter_document(
    doc: PlyDocument,
    remove_large_percent: float,
    remove_isolated_percent: float,
    isolated_knn: int,
) -> tuple[PlyDocument, dict[str, int]]:
    total = int(doc.data.shape[0])
    valid_mask = _finite_mask(doc)
    valid_total = int(valid_mask.sum())
    valid_data = doc.data[valid_mask]

    large_scores = _large_point_scores(PlyDocument(doc.header_lines, doc.format_name, doc.properties, valid_data))
    remove_large_mask = _top_percent_mask(large_scores, remove_large_percent)

    if remove_isolated_percent > 0.0:
        isolated_scores = _isolated_point_scores(_point_positions(PlyDocument(doc.header_lines, doc.format_name, doc.properties, valid_data)), isolated_knn)
        remove_isolated_mask = _top_percent_mask(isolated_scores, remove_isolated_percent)
    else:
        remove_isolated_mask = np.zeros(valid_total, dtype=bool)

    remove_mask = remove_large_mask | remove_isolated_mask
    keep_valid = ~remove_mask

    keep_mask = np.zeros(total, dtype=bool)
    keep_mask[np.flatnonzero(valid_mask)] = keep_valid
    filtered_data = doc.data[keep_mask]

    return (
        PlyDocument(
            header_lines=doc.header_lines,
            format_name=doc.format_name,
            properties=doc.properties,
            data=filtered_data,
        ),
        {
            "input_vertices": total,
            "valid_vertices": valid_total,
            "removed_invalid": total - valid_total,
            "removed_large": int(remove_large_mask.sum()),
            "removed_isolated": int(remove_isolated_mask.sum()),
            "removed_overlap": int((remove_large_mask & remove_isolated_mask).sum()),
            "removed_total": int(remove_mask.sum()),
            "kept": int(filtered_data.shape[0]),
        },
    )


def _write_ply(path: Path, doc: PlyDocument) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_count = int(doc.data.shape[0])
    with path.open("wb") as handle:
        for line in doc.header_lines:
            if line.startswith("element vertex "):
                line = f"element vertex {vertex_count}"
            handle.write((line + "\n").encode("ascii"))

        if doc.format_name == "ascii":
            for row in doc.data:
                values = []
                for prop in doc.properties:
                    value = row[prop.name]
                    if prop.dtype_name in _PLY_INT_TYPES:
                        values.append(str(int(value)))
                    else:
                        values.append(f"{float(value):.9g}")
                handle.write((" ".join(values) + "\n").encode("ascii"))
            return

        dtype = _structured_dtype(
            doc.properties,
            byteorder="<" if doc.format_name == "binary_little_endian" else ">",
        )
        handle.write(doc.data.astype(dtype, copy=False).tobytes())


def filter_3dgs_ply(
    input_path: Path,
    output_path: Path | None = None,
    remove_large_percent: float = 0.0,
    remove_isolated_percent: float = 0.0,
    isolated_knn: int = 8,
) -> dict[str, object]:
    """Filter a standard 3DGS PLY file and write the pruned result."""
    doc = _read_ply(input_path)
    filtered_doc, stats = _filter_document(
        doc,
        float(remove_large_percent),
        float(remove_isolated_percent),
        int(isolated_knn),
    )
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_filtered.ply")
    _write_ply(output_path, filtered_doc)
    stats.update(
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "remove_large_percent": float(remove_large_percent),
            "remove_isolated_percent": float(remove_isolated_percent),
            "isolated_knn": int(isolated_knn),
        }
    )
    return stats


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove the largest percentage of 3DGS splats from a standard "
            "vertex-only PLY file."
        )
    )
    parser.add_argument("ply_path", type=Path, help="Input standard 3DGS PLY file.")
    parser.add_argument(
        "--remove-large-percent",
        "--remove-big-percent",
        "-t",
        "--threshold",
        dest="remove_large_percent",
        type=float,
        default=0.0,
        help="Percentage of large splats to remove.",
    )
    parser.add_argument(
        "--remove-isolated-percent",
        "--remove-free-percent",
        dest="remove_isolated_percent",
        type=float,
        default=0.0,
        help="Percentage of isolated splats to remove.",
    )
    parser.add_argument(
        "--isolated-knn",
        type=int,
        default=8,
        help="Number of nearest neighbors used for isolated-point scoring.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PLY path. Defaults to <input>_filtered.ply.",
    )
    args = parser.parse_args(argv)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    stats = filter_3dgs_ply(
        args.ply_path,
        output_path=args.output,
        remove_large_percent=args.remove_large_percent,
        remove_isolated_percent=args.remove_isolated_percent,
        isolated_knn=args.isolated_knn,
    )
    log_info(
        LOGGER,
        format_key_value_table(
            "PLY Filter",
            [
                ("output", stats["output_path"]),
                ("kept", stats["kept"]),
                ("removed_invalid", stats["removed_invalid"]),
                ("removed_large", stats["removed_large"]),
                ("removed_isolated", stats["removed_isolated"]),
                ("removed_total", stats["removed_total"]),
                ("large_percent", f"{stats['remove_large_percent']:.3g}"),
                ("isolated_percent", f"{stats['remove_isolated_percent']:.3g}"),
                ("isolated_knn", stats["isolated_knn"]),
            ],
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
