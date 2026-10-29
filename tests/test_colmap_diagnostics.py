from __future__ import annotations

from pathlib import Path

import pytest

from tools import diagnose_clean_colmap as diagnostics


def _require_pycolmap():
    try:
        import pycolmap  # pylint: disable=import-outside-toplevel
    except ImportError:
        pytest.skip("pycolmap is not installed")
    return pycolmap


def _write_diagnostic_colmap(path: Path) -> None:
    sparse_dir = path / "sparse"
    sparse_dir.mkdir(parents=True)
    (sparse_dir / "cameras.txt").write_text(
        "\n".join(
            [
                "# Camera list with one line of data per camera:",
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
                "1 SIMPLE_PINHOLE 8 6 8.0 4.0 3.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (sparse_dir / "images.txt").write_text(
        "\n".join(
            [
                "# Image list with two lines of data per image:",
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
                "#   POINTS2D[] as (X, Y, POINT3D_ID)",
                "1 1 0 0 0 0 0 0 1 image_001.jpg",
                "4.0 3.0 1 4.5 3.0 2 4.0 3.5 3",
                "2 1 0 0 0 -0.1 0 0 1 image_002.jpg",
                "4.0 3.0 1 4.5 3.0 2 4.0 3.5 3",
                "3 1 0 0 0 -0.2 0 0 1 image_003.jpg",
                "4.0 3.0 1 4.5 3.0 2 4.0 3.5 3 5.0 4.0 4",
                "4 1 0 0 0 -100.0 0 0 1 image_004.jpg",
                "5.0 4.0 4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (sparse_dir / "points3D.txt").write_text(
        "\n".join(
            [
                "# 3D point list with one line of data per point:",
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]",
                "1 0.0 0.0 2.0 128 64 32 0.1 1 0 2 0 3 0",
                "2 0.1 0.0 2.0 32 128 64 0.1 1 1 2 1 3 1",
                "3 0.0 0.1 2.0 64 32 128 0.1 1 2 2 2 3 2",
                "4 1000.0 0.0 0.0 255 255 255 10.0 3 3 4 0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_colmap_diagnostics_flags_bad_image_and_point(tmp_path):
    pycolmap = _require_pycolmap()
    data_dir = tmp_path / "dataset"
    _write_diagnostic_colmap(data_dir)
    reconstruction = pycolmap.Reconstruction(data_dir / "sparse")

    report = diagnostics.diagnose_reconstruction(
        reconstruction,
        diagnostics.DiagnosticConfig(
            drop_point_distance_percent=25.0,
            drop_camera_distance_percent=25.0,
            min_image_points=2,
            init_num_points=2,
            test_every=2,
        ),
    )

    assert report["summary"]["num_registered_images"] == 4
    assert report["summary"]["num_points3D"] == 4
    assert report["summary"]["init_point_fraction"] == 0.5
    assert report["all_dropped_point_ids"] == [4]
    assert report["all_flagged_image_names"] == ["image_004.jpg"]
    flagged = report["flagged_images"][0]
    assert flagged["split"] == "train"
    assert "camera_distance_percent_outlier" in flagged["reasons"]
    assert "low_observations" in flagged["reasons"]


def test_colmap_cleaning_writes_readable_sparse(tmp_path):
    pycolmap = _require_pycolmap()
    data_dir = tmp_path / "dataset"
    _write_diagnostic_colmap(data_dir)
    reconstruction = pycolmap.Reconstruction(data_dir / "sparse")
    report = diagnostics.diagnose_reconstruction(
        reconstruction,
        diagnostics.DiagnosticConfig(
            drop_point_distance_percent=25.0,
            drop_camera_distance_percent=25.0,
            min_image_points=2,
        ),
    )
    output_sparse = tmp_path / "cleaned_sparse"

    result = diagnostics.clean_reconstruction(
        reconstruction,
        report,
        diagnostics.CleanConfig(
            clean=True,
            clean_images=True,
            output_sparse=output_sparse,
            output_format="text",
        ),
    )

    cleaned = pycolmap.Reconstruction(output_sparse)
    assert result["deleted_points"] == 1
    assert result["deregistered_images"] == 1
    assert not cleaned.exists_point3D(4)
    reg_names = {cleaned.image(image_id).name for image_id in cleaned.reg_image_ids()}
    assert reg_names == {"image_001.jpg", "image_002.jpg", "image_003.jpg"}
