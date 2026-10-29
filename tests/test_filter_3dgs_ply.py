from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "filter_3dgs_ply.py"
    spec = importlib.util.spec_from_file_location("filter_3dgs_ply", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_clean_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "clean_3dgs_ply.py"
    spec = importlib.util.spec_from_file_location("clean_3dgs_ply", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_standard_3dgs_ply(path: Path) -> None:
    rows = [
        (
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0] * 45,
            0.0,
            [0.1, 0.1, 0.1],
            [1.0, 0.0, 0.0, 0.0],
        ),
        (
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0] * 45,
            0.0,
            [0.2, 0.2, 0.2],
            [1.0, 0.0, 0.0, 0.0],
        ),
        (
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.0] * 45,
            0.0,
            [0.3, 0.3, 0.3],
            [1.0, 0.0, 0.0, 0.0],
        ),
        (
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0] * 45,
            0.0,
            [2.0, 2.0, 2.0],
            [1.0, 0.0, 0.0, 0.0],
        ),
    ]

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment test standard 3DGS ply",
        f"element vertex {len(rows)}",
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
    header_lines.extend(f"property float f_rest_{index}" for index in range(45))
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

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
            *[(f"f_rest_{index}", "<f4") for index in range(45)],
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )
    data = np.zeros(len(rows), dtype=dtype)
    for index, (x, normal, f_dc, f_rest, opacity, scales, quat) in enumerate(rows):
        data[index]["x"], data[index]["y"], data[index]["z"] = x
        data[index]["nx"], data[index]["ny"], data[index]["nz"] = normal
        data[index]["f_dc_0"], data[index]["f_dc_1"], data[index]["f_dc_2"] = f_dc
        for rest_index, value in enumerate(f_rest):
            data[index][f"f_rest_{rest_index}"] = value
        data[index]["opacity"] = opacity
        data[index]["scale_0"], data[index]["scale_1"], data[index]["scale_2"] = scales
        data[index]["rot_0"], data[index]["rot_1"], data[index]["rot_2"], data[index]["rot_3"] = quat

    with path.open("wb") as handle:
        for line in header_lines:
            handle.write((line + "\n").encode("ascii"))
        handle.write(data.tobytes())


def test_filter_3dgs_ply_removes_top_percent(tmp_path):
    tool = _load_tool_module()
    input_path = tmp_path / "scene.ply"
    output_path = tmp_path / "scene_filtered.ply"
    _make_standard_3dgs_ply(input_path)

    stats = tool.filter_3dgs_ply(
        input_path,
        output_path=output_path,
        remove_large_percent=25.0,
    )
    doc = tool._read_ply(output_path)

    assert stats["input_vertices"] == 4
    assert stats["valid_vertices"] == 4
    assert stats["removed_invalid"] == 0
    assert stats["removed_large"] == 1
    assert stats["removed_isolated"] == 0
    assert stats["removed_total"] == 1
    assert stats["kept"] == 3
    assert doc.data.shape[0] == 3
    assert list(doc.data["x"]) == [0.0, 1.0, 2.0]
    assert float(doc.data["scale_0"].max()) == 0.30000001192092896


def test_filter_3dgs_ply_removes_isolated_percent(tmp_path):
    tool = _load_tool_module()
    input_path = tmp_path / "scene.ply"
    output_path = tmp_path / "scene_filtered.ply"
    rows = [
        (
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.0] * 45,
            0.0,
            [0.1, 0.1, 0.1],
            [1.0, 0.0, 0.0, 0.0],
        ),
        (
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.0] * 45,
            0.0,
            [0.1, 0.1, 0.1],
            [1.0, 0.0, 0.0, 0.0],
        ),
        (
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.0] * 45,
            0.0,
            [0.1, 0.1, 0.1],
            [1.0, 0.0, 0.0, 0.0],
        ),
        (
            [20.0, 20.0, 20.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.0] * 45,
            0.0,
            [0.1, 0.1, 0.1],
            [1.0, 0.0, 0.0, 0.0],
        ),
    ]
    # Reuse the helper's binary layout by writing the file inline.
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment test isolated 3DGS ply",
        f"element vertex {len(rows)}",
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
    header_lines.extend(f"property float f_rest_{index}" for index in range(45))
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
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
            *[(f"f_rest_{index}", "<f4") for index in range(45)],
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )
    data = np.zeros(len(rows), dtype=dtype)
    for index, (x, normal, f_dc, f_rest, opacity, scales, quat) in enumerate(rows):
        data[index]["x"], data[index]["y"], data[index]["z"] = x
        data[index]["nx"], data[index]["ny"], data[index]["nz"] = normal
        data[index]["f_dc_0"], data[index]["f_dc_1"], data[index]["f_dc_2"] = f_dc
        for rest_index, value in enumerate(f_rest):
            data[index][f"f_rest_{rest_index}"] = value
        data[index]["opacity"] = opacity
        data[index]["scale_0"], data[index]["scale_1"], data[index]["scale_2"] = scales
        data[index]["rot_0"], data[index]["rot_1"], data[index]["rot_2"], data[index]["rot_3"] = quat

    with input_path.open("wb") as handle:
        for line in header_lines:
            handle.write((line + "\n").encode("ascii"))
        handle.write(data.tobytes())

    stats = tool.filter_3dgs_ply(
        input_path,
        output_path=output_path,
        remove_isolated_percent=25.0,
        isolated_knn=2,
    )
    doc = tool._read_ply(output_path)

    assert stats["removed_large"] == 0
    assert stats["removed_isolated"] == 1
    assert stats["removed_total"] == 1
    assert stats["kept"] == 3
    assert doc.data.shape[0] == 3
    assert float(doc.data["x"].max()) == np.float32(0.1)


def test_filter_3dgs_ply_cli_parses_threshold(tmp_path):
    tool = _load_tool_module()
    input_path = tmp_path / "scene.ply"
    output_path = tmp_path / "scene_out.ply"
    _make_standard_3dgs_ply(input_path)

    args = tool._parse_args(
        [
            str(input_path),
            "--remove-large-percent",
            "50",
            "--remove-isolated-percent",
            "25",
            "--isolated-knn",
            "4",
            "--output",
            str(output_path),
        ]
    )

    assert args.ply_path == input_path
    assert args.remove_large_percent == 50.0
    assert args.remove_isolated_percent == 25.0
    assert args.isolated_knn == 4
    assert args.output == output_path


def test_clean_3dgs_ply_auto_collaborative_removes_large_outlier(tmp_path):
    filter_tool = _load_tool_module()
    clean_tool = _load_clean_tool_module()
    input_path = tmp_path / "scene.ply"
    output_path = tmp_path / "scene_cleaned.ply"
    _make_standard_3dgs_ply(input_path)

    doc = filter_tool._read_ply(input_path)
    doc.data["x"][-1] = 20.0
    doc.data["y"][-1] = 20.0
    doc.data["z"][-1] = 20.0
    doc.data["scale_0"][-1] = 2.0
    doc.data["scale_1"][-1] = 2.0
    doc.data["scale_2"][-1] = 2.0
    filter_tool._write_ply(input_path, doc)

    stats = clean_tool.clean_3dgs_ply(
        input_path,
        output_path=output_path,
        options=clean_tool.CleanOptions(
            auto=True,
            auto_large_threshold=1.0,
            auto_outlier_threshold=1.0,
            auto_combined_threshold=2.0,
            auto_max_remove_percent=100.0,
            knn=2,
            scale_domain="log",
        ),
    )
    cleaned = filter_tool._read_ply(output_path)

    assert stats["input_vertices"] == 4
    assert stats["removed_auto_collaborative"] == 1
    assert stats["removed_total"] == 1
    assert stats["kept"] == 3
    assert cleaned.data.shape[0] == 3
    assert float(cleaned.data["x"].max()) == 2.0
    assert stats["removed_samples"][0]["vertex_index"] == 3
    assert "auto_collaborative" in stats["removed_samples"][0]["reasons"]


def test_clean_3dgs_ply_dry_run_writes_report_without_output(tmp_path):
    clean_tool = _load_clean_tool_module()
    input_path = tmp_path / "scene.ply"
    output_path = tmp_path / "scene_cleaned.ply"
    report_path = tmp_path / "cleanup_report.json"
    _make_standard_3dgs_ply(input_path)

    stats = clean_tool.clean_3dgs_ply(
        input_path,
        output_path=output_path,
        report_path=report_path,
        dry_run=True,
        options=clean_tool.CleanOptions(
            auto=False,
            remove_combined_percent=25.0,
            knn=2,
            scale_domain="log",
        ),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert stats["dry_run"] is True
    assert not output_path.exists()
    assert report["dry_run"] is True
    assert report["removed_combined_percentile"] == 1
    assert report["output_path"] == str(output_path)
