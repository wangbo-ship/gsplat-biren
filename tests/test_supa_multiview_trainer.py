# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _load_trainer_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "applications"
        / "supa_multiview_trainer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "applications.supa_multiview_trainer",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_supa_devices(count: int, *, require_brcc: bool = False) -> None:
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pytest.skip("torch_br is not installed")

    if not hasattr(torch, "supa") or not torch.supa.is_available():
        pytest.skip("SUPA device is not available")
    device_count = int(torch.supa.device_count())
    if device_count < count:
        pytest.skip(f"need at least {count} SUPA devices, got {device_count}")
    if require_brcc and shutil.which("brcc") is None:
        pytest.skip("BRCC must be on PATH for BRPyTorch SUDNN JIT")


def _write_tiny_npz(path: Path, *, num_frames: int = 4) -> None:
    images = np.zeros((num_frames, 6, 6, 3), dtype=np.float32)
    for index in range(num_frames):
        images[index, :, :, index % 3] = 0.55 + 0.1 * (index % 2)
    viewmats = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    for index in range(num_frames):
        viewmats[index, 0, 3] = 0.03 * index
    Ks = np.tile(
        np.array([[7.0, 0.0, 3.0], [0.0, 7.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        (num_frames, 1, 1),
    )
    points = np.array(
        [
            [-0.1, -0.1, 2.0],
            [0.1, -0.1, 2.1],
            [-0.1, 0.1, 2.2],
            [0.1, 0.1, 2.3],
        ],
        dtype=np.float32,
    )
    point_colors = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.7, 0.1], [0.1, 0.1, 0.6], [0.6, 0.6, 0.2]],
        dtype=np.float32,
    )
    train_indices = np.arange(min(2, num_frames), dtype=np.int64)
    eval_indices = np.arange(len(train_indices), num_frames, dtype=np.int64)
    if eval_indices.size == 0:
        eval_indices = np.array([num_frames - 1], dtype=np.int64)
    np.savez(
        path,
        images=images,
        viewmats=viewmats,
        Ks=Ks,
        points=points,
        point_colors=point_colors,
        train_indices=train_indices,
        eval_indices=eval_indices,
        scene_scale=np.array(1.0, dtype=np.float32),
    )


def _write_tiny_colmap_text_dataset(path: Path) -> None:
    image_dir = path / "images"
    sparse_dir = path / "sparse"
    image_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)

    colors = [(180, 30, 30), (30, 180, 30)]
    for index, color in enumerate(colors, start=1):
        image = Image.new("RGB", (8, 6), color)
        image.save(image_dir / f"image_{index:03d}.jpg")

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
                "2 1 0 0 0 0.1 0 0 1 image_002.jpg",
                "4.0 3.0 1 4.5 3.0 2 4.0 3.5 3",
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
                "1 0.0 0.0 2.0 128 64 32 0.1 1 0 2 0",
                "2 0.1 0.0 2.0 32 128 64 0.1 1 1 2 1",
                "3 0.0 0.1 2.0 64 32 128 0.1 1 2 2 2",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _assert_three_frame_distributed_eval(result_dir: Path, metrics: dict) -> None:
    assert metrics["eval"]["num_frames"] == 3
    assert [item["global_eval_index"] for item in metrics["eval"]["frames"]] == [0, 1, 2]
    assert [item["name"] for item in metrics["eval"]["frames"]] == [
        "frame_0002",
        "frame_0003",
        "frame_0004",
    ]
    assert metrics["eval"]["grid"] == str(result_dir / "renders" / "eval_grid.png")
    assert Path(metrics["eval"]["grid"]).is_file()
    assert (result_dir / "renders" / "eval_0000_frame_0002.png").is_file()
    assert (result_dir / "renders" / "eval_0001_frame_0003.png").is_file()
    assert (result_dir / "renders" / "eval_0002_frame_0004.png").is_file()
    assert (result_dir / "rank_0000" / "renders" / "eval_0000_frame_0002.png").is_file()
    assert (result_dir / "rank_0000" / "renders" / "eval_0002_frame_0004.png").is_file()
    assert (result_dir / "rank_0001" / "renders" / "eval_0001_frame_0003.png").is_file()


def _assert_two_rank_train_step_grid(result_dir: Path, step: int) -> None:
    root_path = result_dir / "renders" / f"train_step_{step:04d}.png"
    rank0_path = result_dir / "rank_0000" / "renders" / f"train_step_{step:04d}.png"
    rank1_path = result_dir / "rank_0001" / "renders" / f"train_step_{step:04d}.png"
    assert root_path.is_file()
    assert rank0_path.is_file()
    assert rank1_path.is_file()
    with Image.open(root_path) as root_image:
        root_size = root_image.size
    with Image.open(rank0_path) as rank0_image:
        rank0_size = rank0_image.size
    with Image.open(rank1_path) as rank1_image:
        rank1_size = rank1_image.size
    assert rank0_size == rank1_size
    assert root_size == (rank0_size[0], rank0_size[1] * 2)


def test_supa_multiview_trainer_colmap_text_loads_without_pycolmap(tmp_path, monkeypatch):
    trainer = _load_trainer_module()
    data_dir = tmp_path / "tiny_colmap"
    _write_tiny_colmap_text_dataset(data_dir)

    import applications.datasets.colmap as colmap_module

    monkeypatch.setattr(colmap_module, "PycolmapReconstruction", None)
    monkeypatch.setattr(colmap_module, "SceneManager", None)
    cfg = trainer.TrainConfig(
        data=data_dir,
        result_dir=tmp_path / "results",
        data_type="colmap",
        device="cpu",
        iterations=1,
        init_num_points=1,
        data_factor=1,
        test_every=2,
        normalize_world_space=False,
        save_ply=False,
        eval_max_frames=0,
    )
    data = trainer.load_multiview_data(cfg)

    assert len(data.train_frames) == 1
    assert len(data.eval_frames) == 1
    assert colmap_module._load_colmap_text_model(str(data_dir / "sparse")).point_indices[
        "image_002.jpg"
    ].tolist() == [0, 1, 2]
    assert data.points.shape == (3, 3)
    assert data.point_colors.shape == (3, 3)
    assert data.train_frames[0].image.shape == (6, 8, 3)
    torch.testing.assert_close(
        data.train_frames[0].K,
        torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 3.0], [0.0, 0.0, 1.0]]),
    )


def test_colmap_pycolmap_reconstruction_loads_text_dataset(tmp_path):
    import applications.datasets.colmap as colmap_module

    if colmap_module.PycolmapReconstruction is None:
        pytest.skip("pycolmap.Reconstruction is not installed")

    data_dir = tmp_path / "tiny_colmap"
    _write_tiny_colmap_text_dataset(data_dir)

    model = colmap_module._load_colmap_model(str(data_dir / "sparse"))

    assert len(model.cameras) == 1
    assert len(model.images) == 2
    assert model.points3D.shape == (3, 3)
    assert model.point3D_colors.shape == (3, 3)
    assert model.point_indices["image_002.jpg"].tolist() == [0, 1, 2]
    assert next(iter(model.cameras.values())).camera_type == "SIMPLE_PINHOLE"


def test_supa_multiview_trainer_npz_smoke_cpu(tmp_path):
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pass

    trainer = _load_trainer_module()
    data_path = tmp_path / "tiny_multiview.npz"
    result_dir = tmp_path / "results"

    images = np.zeros((3, 6, 6, 3), dtype=np.float32)
    images[0, :, :, 0] = 0.8
    images[1, :, :, 1] = 0.7
    images[2, :, :, 2] = 0.6
    viewmats = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    viewmats[1, 0, 3] = 0.05
    viewmats[2, 1, 3] = -0.05
    Ks = np.tile(
        np.array([[7.0, 0.0, 3.0], [0.0, 7.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        (3, 1, 1),
    )
    points = np.array(
        [
            [-0.1, -0.1, 2.0],
            [0.1, -0.1, 2.1],
            [-0.1, 0.1, 2.2],
            [0.1, 0.1, 2.3],
        ],
        dtype=np.float32,
    )
    point_colors = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.7, 0.1], [0.1, 0.1, 0.6], [0.6, 0.6, 0.2]],
        dtype=np.float32,
    )
    np.savez(
        data_path,
        images=images,
        viewmats=viewmats,
        Ks=Ks,
        points=points,
        point_colors=point_colors,
        train_indices=np.array([0, 1], dtype=np.int64),
        eval_indices=np.array([2], dtype=np.int64),
        scene_scale=np.array(1.0, dtype=np.float32),
    )

    cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=result_dir,
        data_type="npz",
        device="cpu",
        iterations=2,
        init_num_points=4,
        save_every=2,
        seed=7,
    )
    stats = trainer.train_multiview(cfg)

    assert len(stats["losses"]) == 2
    assert stats["num_gaussians"] == 4
    assert Path(stats["final_render"]).is_file()
    assert Path(stats["checkpoint"]).is_file()
    step_ckpt = result_dir / "ckpts" / "step_0002.pt"
    step_ply = result_dir / "ply" / "step_0002_points.ply"
    assert step_ckpt.is_file()
    assert step_ply.is_file()
    assert Path(stats["metrics"]).is_file()
    metrics = json.loads(Path(stats["metrics"]).read_text(encoding="utf-8"))
    assert metrics["step"] == 2
    assert metrics["num_gaussians"] == 4
    assert metrics["config"]["data"] == str(data_path)
    assert metrics["config"]["result_dir"] == str(result_dir)
    assert metrics["config"]["init_num_points"] == 4
    assert metrics["config"]["init_opa"] == 0.1
    assert metrics["config"]["init_scale"] == 1.0
    assert metrics["config"]["eval_max_frames"] == 8
    assert metrics["config"]["ssim_lambda"] == 0.2
    assert metrics["config"]["shuffle_train"] is True
    assert metrics["config"]["means_lr_final_factor"] == 0.01
    assert metrics["config"]["means_lr"] == 1.6e-4
    assert metrics["config"]["scales_lr"] == 5e-3
    assert metrics["config"]["opacities_lr"] == 5e-2
    assert metrics["config"]["quats_lr"] == 1e-3
    assert metrics["config"]["sh0_lr"] == 2.5e-3
    assert metrics["config"]["shN_lr"] == 2.5e-3 / 20.0
    assert metrics["config"]["sh_degree"] == 3
    assert metrics["config"]["sh_degree_interval"] == 1000
    assert metrics["config"]["global_scale"] == 1.0
    assert metrics["config"]["near_plane"] == 0.01
    assert metrics["config"]["far_plane"] == 1e10
    assert metrics["config"]["random_bkgd"] is False
    assert metrics["config"]["refine_gaussians"] is True
    assert metrics["config"]["refine_start_iter"] == 500
    assert metrics["config"]["refine_every"] == 100
    assert metrics["train"]["num_steps"] == 2
    step_times = metrics["train"]["step_times_s"]
    step_time_summary = metrics["train"]["step_time_summary_s"]
    assert len(step_times) == 2
    assert step_time_summary["count"] == 2
    assert step_time_summary["min"] == pytest.approx(min(step_times))
    assert step_time_summary["mean"] == pytest.approx(sum(step_times) / len(step_times))
    assert step_time_summary["median"] == pytest.approx(sum(step_times) / len(step_times))
    assert step_time_summary["max"] == pytest.approx(max(step_times))
    assert stats["step_time_summary_s"]["count"] == 2
    phase_names = metrics["train"]["phase_names"]
    phase_times = metrics["train"]["phase_times_s"]
    phase_summaries = metrics["train"]["phase_time_summaries_s"]
    for phase in (
        "data_and_crop",
        "target_to_device",
        "rasterize",
        "loss",
        "backward",
        "optimizer_step",
        "synchronize",
        "metrics_cpu",
    ):
        assert phase in phase_names
        assert len(phase_times[phase]) == 2
        assert phase_summaries[phase]["count"] == 2
        assert phase_summaries[phase]["median"] is not None
    assert len(phase_times["save_step"]) == 2
    assert phase_summaries["save_step"]["count"] >= 1
    assert stats["phase_time_summaries_s"]["rasterize"]["count"] == 2
    assert len(metrics["train"]["l1_losses"]) == 2
    assert len(metrics["train"]["ssim_losses"]) == 2
    assert metrics["train"]["last_l1_loss"] == metrics["train"]["l1_losses"][-1]
    assert metrics["train"]["last_ssim_loss"] == metrics["train"]["ssim_losses"][-1]
    assert len(metrics["train"]["means_lrs"]) == 2
    assert metrics["train"]["last_means_lr"] == metrics["train"]["means_lrs"][-1]
    assert metrics["train"]["means_lrs"][1] < metrics["train"]["means_lrs"][0]
    assert metrics["train"]["sh_degrees"] == [0, 0]
    assert metrics["train"]["last_sh_degree"] == 0
    assert metrics["train"]["num_gaussians"] == [4, 4]
    assert metrics["train"]["last_num_gaussians"] == 4
    assert metrics["train"]["refinement_events"] == []
    assert metrics["diagnostics"]["params"]["num_gaussians"] == 4
    assert metrics["diagnostics"]["params"]["opacity"]["finite"] == 4
    assert metrics["diagnostics"]["params"]["max_scale_scene_fraction"]["max"] > 0.0
    assert metrics["diagnostics"]["refinement_state"]["observed"] >= 0
    assert metrics["eval"]["num_frames"] == 1
    assert metrics["eval"]["l1"] is not None
    assert metrics["eval"]["mse"] is not None
    assert metrics["eval"]["grid"] is not None
    assert metrics["composite_variant"] == "torch"
    assert metrics["tile_count_backend"] == "torch"
    assert metrics["tiled_metadata_backend"] is None
    assert metrics["train"]["last_composite_variant"] == "torch"
    assert metrics["train"]["last_tile_count_backend"] == "torch"
    assert metrics["train"]["last_tiled_metadata_backend"] is None
    assert stats["composite_variant"] == "torch"
    assert stats["tile_count_backend"] == "torch"
    assert stats["tiled_metadata_backend"] is None
    assert Path(metrics["eval"]["grid"]).is_file()
    assert stats["ply"] is not None
    ply_path = Path(stats["ply"])
    assert ply_path.is_file()
    ckpt = torch.load(stats["checkpoint"], map_location="cpu")
    assert ckpt["step"] == 2
    assert ckpt["config"]["data"] == str(data_path)
    assert ckpt["config"]["init_num_points"] == 4
    assert ckpt["config"]["eval_max_frames"] == 8
    assert ckpt["splats"]["means"].shape == (4, 3)
    assert ckpt["splats"]["scales"].shape == (4, 3)
    assert ckpt["splats"]["sh0"].shape == (4, 1, 3)
    assert ckpt["splats"]["shN"].shape == (4, 15, 3)
    assert "log_scales" not in ckpt["splats"]
    assert "colors" not in ckpt["splats"]
    assert "optimizer" in ckpt
    assert ckpt["optimizer"]["format"] == "optimizer_dict"
    assert set(ckpt["optimizer"]["optimizers"]) >= {
        "means",
        "scales",
        "quats",
        "opacities",
        "sh0",
        "shN",
    }
    periodic_ckpt = torch.load(step_ckpt, map_location="cpu")
    assert periodic_ckpt["step"] == 2
    assert periodic_ckpt["config"]["data"] == str(data_path)
    assert periodic_ckpt["config"]["init_num_points"] == 4
    assert periodic_ckpt["splats"]["means"].shape == (4, 3)
    assert periodic_ckpt["splats"]["scales"].shape == (4, 3)
    assert periodic_ckpt["splats"]["sh0"].shape == (4, 1, 3)
    assert periodic_ckpt["splats"]["shN"].shape == (4, 15, 3)
    assert "optimizer" in periodic_ckpt
    assert periodic_ckpt["optimizer"]["format"] == "optimizer_dict"
    ply_data = ply_path.read_bytes()
    header_bytes, payload = ply_data.split(b"end_header\n", 1)
    ply_header = header_bytes.decode("ascii").splitlines() + ["end_header"]
    assert "element vertex 4" in ply_header
    assert "format binary_little_endian 1.0" in ply_header
    assert "property float opacity" in ply_header
    assert "property float f_dc_0" in ply_header
    assert "property float f_rest_44" in ply_header
    assert "property float scale_0" in ply_header
    assert "property float rot_0" in ply_header
    property_count = sum(line.startswith("property float ") for line in ply_header)
    assert property_count == 62
    assert len(payload) == 4 * property_count * 4
    vertices = np.frombuffer(payload, dtype="<f4").reshape(4, property_count)
    np.testing.assert_allclose(vertices[:, 9:54], 0.0)


def test_supa_multiview_trainer_negative_init_uses_point_count_cpu(tmp_path):
    trainer = _load_trainer_module()
    data_path = tmp_path / "tiny_multiview.npz"
    result_dir = tmp_path / "results"
    _write_tiny_npz(data_path, num_frames=3)

    cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=result_dir,
        data_type="npz",
        device="cpu",
        iterations=1,
        init_num_points=-1,
        save_ply=False,
        eval_max_frames=0,
        ssim_lambda=0.0,
        sh_degree=0,
    )
    stats = trainer.train_multiview(cfg)

    assert stats["num_gaussians"] == 4
    metrics = json.loads(Path(stats["metrics"]).read_text(encoding="utf-8"))
    assert metrics["config"]["init_num_points"] == 4
    ckpt = torch.load(stats["checkpoint"], map_location="cpu")
    assert ckpt["config"]["init_num_points"] == 4
    assert ckpt["splats"]["means"].shape == (4, 3)


def test_supa_multiview_trainer_negative_init_requires_point_cloud(tmp_path):
    trainer = _load_trainer_module()
    data_path = tmp_path / "no_points.npz"

    images = np.zeros((2, 6, 6, 3), dtype=np.float32)
    viewmats = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    Ks = np.tile(
        np.array([[7.0, 0.0, 3.0], [0.0, 7.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        (2, 1, 1),
    )
    np.savez(
        data_path,
        images=images,
        viewmats=viewmats,
        Ks=Ks,
        train_indices=np.array([0], dtype=np.int64),
        eval_indices=np.array([1], dtype=np.int64),
    )

    cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=tmp_path / "results",
        data_type="npz",
        device="cpu",
        iterations=1,
        init_num_points=-1,
        save_ply=False,
        eval_max_frames=0,
    )
    with pytest.raises(ValueError, match="requires an input point cloud"):
        trainer.train_multiview(cfg)


def test_supa_multiview_trainer_parse_explicit_bool_cli(tmp_path, monkeypatch):
    trainer = _load_trainer_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supa_multiview_trainer.py",
            "--data",
            str(tmp_path / "data.npz"),
            "--init-num-points",
            "-1",
            "--init-opa",
            "0.25",
            "--init-scale",
            "0.5",
            "--print-all-step-info",
            "true",
            "--print-debug-info",
            "true",
            "--native-tiled",
            "auto",
            "--native-tiled-metadata",
            "true",
            "--native-tiled-train",
            "true",
            "--native-tiled-backward-mode",
            "parallel",
            "--native-alpha-threshold",
            "0.0039215686",
            "--train-crop-size",
            "4",
            "--near-plane",
            "0.02",
            "--far-plane",
            "1000",
            "--radius-clip",
            "0.5",
            "--log-step-time",
            "true",
            "--global-scale",
            "1.25",
            "--ssim-lambda",
            "0.35",
            "--random-bkgd",
            "true",
            "--shuffle-train",
            "false",
            "--means-lr-final-factor",
            "0.25",
            "--means-lr",
            "0.0002",
            "--scales-lr",
            "0.006",
            "--opacities-lr",
            "0.06",
            "--quats-lr",
            "0.002",
            "--sh0-lr",
            "0.003",
            "--shN-lr",
            "0.0003",
            "--sh-degree",
            "2",
            "--sh-degree-interval",
            "5",
            "--refine-gaussians",
            "false",
            "--refine-profile",
            "crop-balanced",
            "--refine-start-iter",
            "2",
            "--refine-stop-iter",
            "20",
            "--refine-every",
            "3",
            "--reset-every",
            "9",
            "--pause-refine-after-reset",
            "1",
            "--prune-opa",
            "0.01",
            "--grow-grad2d",
            "0.0003",
            "--grow-scale3d",
            "0.02",
            "--grow-scale2d",
            "0.06",
            "--prune-scale3d",
            "0.2",
            "--prune-scale2d",
            "0.25",
            "--refine-scale2d-stop-iter",
            "10",
            "--max-gaussians",
            "123",
            "--max-gaussians-ramp-steps",
            "50",
            "--refine-exposure",
            "true",
            "--exposure-lr",
            "0.002",
            "--exposure-l2",
            "0.003",
        ],
    )

    cfg = trainer._parse_args()

    assert cfg.init_num_points == -1
    assert cfg.init_opa == 0.25
    assert cfg.init_scale == 0.5
    assert cfg.print_all_step_info is True
    assert cfg.print_debug_info is True
    assert cfg.native_tiled == "auto"
    assert cfg.native_tiled_metadata is True
    assert cfg.native_tiled_train is True
    assert cfg.native_tiled_backward_mode == "parallel"
    assert cfg.native_alpha_threshold == 0.0039215686
    assert cfg.train_crop_size == 4
    assert cfg.near_plane == 0.02
    assert cfg.far_plane == 1000.0
    assert cfg.radius_clip == 0.5
    assert cfg.log_step_time is True
    assert cfg.global_scale == 1.25
    assert cfg.ssim_lambda == 0.35
    assert cfg.random_bkgd is True
    assert cfg.shuffle_train is False
    assert cfg.means_lr_final_factor == 0.25
    assert cfg.means_lr == 0.0002
    assert cfg.scales_lr == 0.006
    assert cfg.opacities_lr == 0.06
    assert cfg.quats_lr == 0.002
    assert cfg.sh0_lr == 0.003
    assert cfg.shN_lr == 0.0003
    assert cfg.sh_degree == 2
    assert cfg.sh_degree_interval == 5
    assert cfg.refine_gaussians is False
    assert cfg.refine_profile == "crop-balanced"
    assert cfg.refine_start_iter == 2
    assert cfg.refine_stop_iter == 20
    assert cfg.refine_every == 3
    assert cfg.reset_every == 9
    assert cfg.pause_refine_after_reset == 1
    assert cfg.prune_opa == 0.01
    assert cfg.grow_grad2d == 0.0003
    assert cfg.grow_scale3d == 0.02
    assert cfg.grow_scale2d == 0.06
    assert cfg.prune_scale3d == 0.2
    assert cfg.prune_scale2d == 0.25
    assert cfg.refine_scale2d_stop_iter == 10
    assert cfg.max_gaussians == 123
    assert cfg.max_gaussians_ramp_steps == 50
    assert cfg.refine_exposure is True
    assert cfg.exposure_lr == 0.002
    assert cfg.exposure_l2 == 0.003


def test_supa_multiview_trainer_parse_auto_output_root(tmp_path, monkeypatch):
    trainer = _load_trainer_module()
    output_root = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supa_multiview_trainer.py",
            "--data",
            str(tmp_path / "south-building.npz"),
            "--auto-output-root",
            str(output_root),
            "--iterations",
            "12",
            "--init-num-points",
            "34",
            "--sh-degree",
            "2",
            "--seed",
            "7",
            "--device",
            "supa:0",
            "--native",
            "true",
            "--native-tiled",
            "auto",
            "--train-crop-size",
            "96",
            "--random-bkgd",
            "true",
            "--refine-gaussians",
            "false",
        ],
    )

    cfg = trainer._parse_args()

    assert cfg.auto_output_root == output_root
    assert cfg.result_dir.parent == output_root
    name = cfg.result_dir.name
    assert "south-building" in name
    assert "it12" in name
    assert "pts34" in name
    assert "sh2" in name
    assert "seed7" in name
    assert "supa0" in name
    assert "native" in name
    assert "tiled-auto" in name
    assert "crop96" in name
    assert "randbkgd" in name
    assert "refine-false" in name


def test_supa_multiview_trainer_output_paths_are_mutually_exclusive(tmp_path, monkeypatch):
    trainer = _load_trainer_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supa_multiview_trainer.py",
            "--data",
            str(tmp_path / "data.npz"),
            "--result-dir",
            str(tmp_path / "exact"),
            "--auto-output-root",
            str(tmp_path / "runs"),
        ],
    )

    with pytest.raises(SystemExit):
        trainer._parse_args()


def test_supa_multiview_trainer_train_frame_index_shuffle_is_deterministic():
    trainer = _load_trainer_module()

    sequential = [
        trainer._train_frame_index(step, 3, seed=5, shuffle=False)
        for step in range(1, 7)
    ]
    shuffled = [
        trainer._train_frame_index(step, 3, seed=5, shuffle=True)
        for step in range(1, 7)
    ]
    shuffled_again = [
        trainer._train_frame_index(step, 3, seed=5, shuffle=True)
        for step in range(1, 7)
    ]
    distributed_epoch = [
        trainer._train_frame_index(
            step,
            4,
            seed=7,
            shuffle=True,
            stream_rank=rank,
            stream_world_size=2,
        )
        for step in range(1, 3)
        for rank in range(2)
    ]

    assert sequential == [0, 1, 2, 0, 1, 2]
    assert shuffled == shuffled_again
    assert shuffled == [2, 1, 0, 1, 2, 0]
    assert sorted(shuffled[:3]) == [0, 1, 2]
    assert distributed_epoch == [3, 2, 0, 1]
    assert sorted(distributed_epoch) == [0, 1, 2, 3]


def test_supa_multiview_trainer_train_crop_is_step_deterministic():
    trainer = _load_trainer_module()
    frame = trainer.Frame(
        image=torch.zeros(32, 48, 3),
        viewmat=torch.eye(4),
        K=torch.eye(3),
        name="crop_test",
    )

    step1 = trainer._sample_train_crop(frame, 16, seed=7, step=1)
    step241 = trainer._sample_train_crop(frame, 16, seed=7, step=241)
    resumed_step241 = trainer._sample_train_crop(frame, 16, seed=7, step=241)
    rank1_step241 = trainer._sample_train_crop(
        frame,
        16,
        seed=7,
        step=241,
        stream_rank=1,
        stream_world_size=2,
    )

    assert step241 == resumed_step241
    assert step241 != step1
    assert rank1_step241 != step241


def test_supa_multiview_trainer_train_crop_prefers_masked_pixels():
    trainer = _load_trainer_module()
    mask = torch.zeros(32, 48, dtype=torch.bool)
    mask[20, 33] = True
    frame = trainer.Frame(
        image=torch.zeros(32, 48, 3),
        viewmat=torch.eye(4),
        K=torch.eye(3),
        name="masked_crop_test",
        mask=mask,
    )

    crop = trainer._sample_train_crop(frame, 8, seed=3, step=17)
    assert crop is not None
    cropped_mask = mask[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
    assert bool(cropped_mask.any().item())


def test_supa_multiview_trainer_means_lr_schedule():
    trainer = _load_trainer_module()

    base_lr = 0.001
    assert trainer._scheduled_means_lr(base_lr, 0.01, 1, 3) == pytest.approx(base_lr)
    assert trainer._scheduled_means_lr(base_lr, 0.01, 3, 3) == pytest.approx(
        base_lr * (0.01 ** (2.0 / 3.0))
    )
    assert trainer._scheduled_means_lr(base_lr, 0.5, 1, 1) == pytest.approx(base_lr)


def test_supa_multiview_trainer_supa_safe_adam_matches_torch_adam_cpu():
    trainer = _load_trainer_module()
    ours = torch.nn.Parameter(torch.tensor([[0.2, -0.4], [0.6, -0.8]]))
    ref = torch.nn.Parameter(ours.detach().clone())
    grad = torch.tensor([[0.3, -0.1], [0.2, -0.5]])
    ours.grad = grad.clone()
    ref.grad = grad.clone()
    ours_optimizer = trainer._create_adam(
        [{"params": [ours], "lr": 0.01, "name": "test"}],
        eps=1e-8,
    )
    ref_optimizer = torch.optim.Adam(
        [{"params": [ref], "lr": 0.01, "name": "test"}],
        eps=1e-8,
        foreach=False,
    )

    ours_optimizer.step()
    ref_optimizer.step()

    torch.testing.assert_close(ours.detach(), ref.detach())


def test_supa_multiview_trainer_active_sh_degree_matches_upstream_step_timing():
    trainer = _load_trainer_module()
    cfg = trainer.TrainConfig(
        data=Path("unused.npz"),
        result_dir=Path("unused"),
        sh_degree=3,
        sh_degree_interval=5,
    )

    assert trainer._active_sh_degree(1, cfg) == 0
    assert trainer._active_sh_degree(5, cfg) == 0
    assert trainer._active_sh_degree(6, cfg) == 1
    assert trainer._active_sh_degree(11, cfg) == 2
    assert trainer._active_sh_degree(100, cfg) == 3


def test_supa_multiview_trainer_uses_upstream_init_and_lr_config():
    trainer = _load_trainer_module()
    frames = [
        trainer.Frame(
            image=torch.zeros(2, 2, 3),
            viewmat=torch.eye(4),
            K=torch.eye(3),
            mask=None,
            name="frame",
        )
    ]
    data = trainer.MultiviewData(
        train_frames=frames,
        eval_frames=frames,
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        point_colors=torch.full((4, 3), 0.5),
        scene_scale=2.0,
    )
    cfg = trainer.TrainConfig(
        data=Path("unused.npz"),
        result_dir=Path("unused"),
        device="cpu",
        init_num_points=4,
        init_opa=0.25,
        init_scale=0.5,
        lr_scale=2.0,
        means_lr=2e-4,
        scales_lr=6e-3,
        opacities_lr=6e-2,
        quats_lr=2e-3,
        sh0_lr=3e-3,
        shN_lr=3e-4,
    )

    params = trainer.create_model(data, cfg, torch.device("cpu"))
    expected_dist = (torch.tensor([(1.0**2 + 2.0**2 + 3.0**2) / 3.0]).sqrt() * 0.5).log()

    assert torch.sigmoid(params["opacities"]).mean().item() == pytest.approx(0.25)
    assert params["scales"][0, 0].item() == pytest.approx(expected_dist.item())

    optimizers = trainer.create_optimizers(params, data.scene_scale, cfg)
    lrs = {name: optimizer.param_groups[0]["lr"] for name, optimizer in optimizers.items()}

    assert lrs["means"] == pytest.approx(2e-4 * 2.0 * 2.0)
    assert lrs["scales"] == pytest.approx(6e-3 * 2.0)
    assert lrs["opacities"] == pytest.approx(6e-2 * 2.0)
    assert lrs["quats"] == pytest.approx(2e-3 * 2.0)
    assert lrs["sh0"] == pytest.approx(3e-3 * 2.0)
    assert lrs["shN"] == pytest.approx(3e-4 * 2.0)


def test_supa_multiview_trainer_scale_init_matches_full_knn():
    trainer = _load_trainer_module()
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.4],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    scales = trainer._estimate_scales(points, scene_scale=1.0, init_scale=0.75)
    nearest = torch.cdist(points, points).topk(k=4, largest=False).values[:, 1:]
    expected = (
        nearest.square()
        .mean(dim=-1)
        .sqrt()
        .clamp_min(1e-7)
        .mul(0.75)
        .log()
        .unsqueeze(-1)
        .repeat(1, 3)
    )

    torch.testing.assert_close(scales, expected)


def test_supa_multiview_trainer_scale_init_uses_full_cloud_for_large_inputs():
    trainer = _load_trainer_module()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(32, dtype=torch.float32),
        torch.arange(32, dtype=torch.float32),
        indexing="ij",
    )
    centers = (
        torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1) * 0.03
    )
    offsets = torch.tensor([0.0, 0.001, 0.002, 0.003], dtype=torch.float32)
    points = torch.zeros(centers.shape[0], offsets.shape[0], 3)
    points[:, :, :2] = centers.unsqueeze(1)
    points[:, :, 2] = offsets
    points = points.reshape(-1, 3).contiguous()

    scales = trainer._estimate_scales(points, scene_scale=1.0, init_scale=1.0).exp()

    assert scales.max().item() < 0.003
    assert scales.min().item() > 0.0


def test_supa_multiview_trainer_masked_render_loss_uses_upstream_zeroed_ssim():
    trainer = _load_trainer_module()
    render = torch.zeros(6, 6, 3)
    target = torch.zeros(6, 6, 3)
    mask = torch.zeros(6, 6, dtype=torch.bool)
    mask[:3, :3] = True
    target[mask] = 1.0

    _, _, masked_ssim = trainer._render_loss(render, target, mask, 1.0)
    mask_float = mask.to(dtype=render.dtype).unsqueeze(-1)
    upstream_style = trainer._ssim_loss(render * mask_float, target * mask_float, None)
    locally_normalized = trainer._ssim_loss(render, target, mask)

    assert masked_ssim.item() == pytest.approx(upstream_style.item(), abs=1e-6)
    assert masked_ssim.item() != pytest.approx(locally_normalized.item(), abs=1e-6)


def test_supa_multiview_trainer_ssim_loss_supa_matches_cpu_reference():
    _require_supa_devices(1)
    trainer = _load_trainer_module()
    device = torch.device("supa:0")
    torch.supa.set_device(device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(991)
    render_cpu = torch.rand((18, 17, 3), generator=generator, requires_grad=True)
    target_cpu = torch.rand((18, 17, 3), generator=generator)

    cpu_loss = trainer._ssim_loss(render_cpu, target_cpu, None)
    cpu_loss.backward()

    render_supa = render_cpu.detach().to(device).requires_grad_(True)
    target_supa = target_cpu.to(device)
    supa_loss = trainer._ssim_loss(render_supa, target_supa, None)
    supa_loss.backward()
    torch.supa.synchronize()

    torch.testing.assert_close(
        supa_loss.detach().cpu(),
        cpu_loss.detach(),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        render_supa.grad.detach().cpu(),
        render_cpu.grad,
        atol=2e-4,
        rtol=2e-3,
    )


def test_supa_multiview_trainer_random_background_uses_alpha():
    trainer = _load_trainer_module()
    render = torch.zeros(2, 3, 3)

    torch.manual_seed(7)
    opaque = trainer._apply_random_background(render, torch.ones(2, 3, 1))
    torch.testing.assert_close(opaque, render)

    torch.manual_seed(7)
    transparent = trainer._apply_random_background(render, torch.zeros(2, 3, 1))
    assert not torch.allclose(transparent, render)
    torch.testing.assert_close(transparent, transparent[0, 0].expand_as(transparent))


def test_supa_multiview_trainer_loads_legacy_rgb_checkpoint_with_sh_tail():
    trainer = _load_trainer_module()
    splats = {
        "means": torch.zeros(2, 3),
        "log_scales": torch.zeros(2, 3),
        "quats": torch.zeros(2, 4),
        "opacities": torch.zeros(2),
        "colors": torch.zeros(2, 3),
    }

    params = trainer._model_from_splats(splats, torch.device("cpu"), sh_degree=3)

    assert params["scales"].shape == (2, 3)
    assert params["sh0"].shape == (2, 1, 3)
    assert torch.count_nonzero(params["sh0"]).item() == 0
    assert params["shN"].shape == (2, 15, 3)
    assert trainer._sh_param_to_3d(params["shN"]).shape == (2, 15, 3)
    assert torch.count_nonzero(params["shN"]).item() == 0


def test_supa_multiview_trainer_sh_checkpoint_keeps_public_shape():
    trainer = _load_trainer_module()
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(2, 3)),
            "scales": torch.nn.Parameter(torch.zeros(2, 3)),
            "quats": torch.nn.Parameter(torch.zeros(2, 4)),
            "opacities": torch.nn.Parameter(torch.zeros(2)),
            "sh0": torch.nn.Parameter(torch.zeros(2, 1, 3)),
            "shN": torch.nn.Parameter(torch.arange(90, dtype=torch.float32).reshape(2, 15, 3)),
        }
    )

    splats = trainer._params_splats_for_checkpoint(params)

    assert params["shN"].shape == (2, 15, 3)
    assert splats["sh0"].shape == (2, 1, 3)
    assert splats["shN"].shape == (2, 15, 3)
    torch.testing.assert_close(splats["shN"], params["shN"].detach())


def test_supa_multiview_trainer_refinement_resizes_optimizer_state():
    trainer = _load_trainer_module()
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(2, 3)),
            "scales": torch.nn.Parameter(torch.zeros(2, 3)),
            "quats": torch.nn.Parameter(
                torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
            ),
            "opacities": torch.nn.Parameter(torch.zeros(2)),
            "sh0": torch.nn.Parameter(torch.zeros(2, 1, 3)),
            "shN": torch.nn.Parameter(torch.zeros(2, 3, 3)),
        }
    )
    cfg = trainer.TrainConfig(
        data=Path("unused.npz"),
        result_dir=Path("unused"),
        sh_degree=1,
    )
    optimizer = trainer.create_optimizers(params, scene_scale=1.0, cfg=cfg)
    loss = sum(param.square().sum() for param in params.values())
    loss.backward()
    trainer._optimizer_step(optimizer)
    trainer._optimizer_zero_grad(optimizer)

    duplicated = trainer._duplicate_gaussians(
        params, optimizer, torch.tensor([True, False])
    )
    assert duplicated == 1
    assert params["means"].shape == (3, 3)

    split = trainer._split_gaussians(
        params, optimizer, torch.tensor([False, True, False])
    )
    assert split == 1
    assert params["means"].shape == (4, 3)

    params["opacities"].data.fill_(10.0)
    trainer._reset_opacities(params, optimizer, 0.01)
    assert torch.sigmoid(params["opacities"]).max().item() <= 0.010001

    pruned = trainer._prune_gaussians(
        params, optimizer, torch.tensor([False, True, False, False])
    )
    assert pruned == 1
    assert params["means"].shape == (3, 3)

    for name, item in optimizer.items():
        if name not in params:
            continue
        param = params[name]
        state = item.state.get(param, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                assert state[key].shape == param.shape


def test_supa_multiview_trainer_adam_sanitizes_nonfinite_gradients():
    trainer = _load_trainer_module()
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0, -4.0]))
    optimizer = trainer._create_adam(
        [{"params": [param], "lr": 1e-2, "name": "test"}]
    )

    param.grad = torch.tensor([float("inf"), float("-inf"), float("nan"), 1e30])
    optimizer.step()

    assert torch.isfinite(param).all()
    state = optimizer.state[param]
    assert torch.isfinite(state["exp_avg"]).all()
    assert torch.isfinite(state["exp_avg_sq"]).all()
    assert torch.isfinite(state["step"]).all()


def test_supa_multiview_trainer_refinement_resizes_optimizer_dict_state():
    trainer = _load_trainer_module()
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(2, 3)),
            "scales": torch.nn.Parameter(torch.zeros(2, 3)),
            "quats": torch.nn.Parameter(
                torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
            ),
            "opacities": torch.nn.Parameter(torch.zeros(2)),
            "sh0": torch.nn.Parameter(torch.zeros(2, 1, 3)),
            "shN": torch.nn.Parameter(torch.zeros(2, 3, 3)),
        }
    )
    cfg = trainer.TrainConfig(
        data=Path("unused.npz"),
        result_dir=Path("unused"),
        sh_degree=1,
    )
    optimizers = trainer.create_optimizers(params, scene_scale=1.0, cfg=cfg)
    loss = sum(param.square().sum() for param in params.values())
    loss.backward()
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    duplicated = trainer._duplicate_gaussians(
        params, optimizers, torch.tensor([True, False])
    )
    assert duplicated == 1

    split = trainer._split_gaussians(
        params, optimizers, torch.tensor([False, True, False])
    )
    assert split == 1

    pruned = trainer._prune_gaussians(
        params, optimizers, torch.tensor([False, True, False, False])
    )
    assert pruned == 1

    for name, optimizer in optimizers.items():
        if name not in params:
            continue
        param = params[name]
        state = optimizer.state.get(param, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                assert state[key].shape == param.shape


def test_supa_multiview_trainer_refinement_resizes_running_state():
    trainer = _load_trainer_module()
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(3, 3)),
            "scales": torch.nn.Parameter(torch.zeros(3, 3)),
            "quats": torch.nn.Parameter(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                )
            ),
            "opacities": torch.nn.Parameter(torch.zeros(3)),
            "sh0": torch.nn.Parameter(torch.zeros(3, 1, 3)),
        }
    )
    cfg = trainer.TrainConfig(data=Path("unused.npz"), result_dir=Path("unused"))
    optimizers = trainer.create_optimizers(params, scene_scale=1.0, cfg=cfg)
    state = {
        "grad2d": torch.tensor([1.0, 2.0, 3.0]),
        "count": torch.tensor([4.0, 5.0, 6.0]),
        "radii": torch.tensor([0.1, 0.2, 0.3]),
    }

    duplicated = trainer._duplicate_gaussians(
        params,
        optimizers,
        torch.tensor([True, False, False]),
        state=state,
    )
    assert duplicated == 1
    torch.testing.assert_close(state["grad2d"], torch.tensor([1.0, 2.0, 3.0, 1.0]))
    torch.testing.assert_close(state["count"], torch.tensor([4.0, 5.0, 6.0, 4.0]))
    torch.testing.assert_close(state["radii"], torch.tensor([0.1, 0.2, 0.3, 0.1]))

    split = trainer._split_gaussians(
        params,
        optimizers,
        torch.tensor([False, True, False, False]),
        state=state,
    )
    assert split == 1
    torch.testing.assert_close(
        state["grad2d"],
        torch.tensor([1.0, 3.0, 1.0, 2.0, 2.0]),
    )
    torch.testing.assert_close(
        state["count"],
        torch.tensor([4.0, 6.0, 4.0, 5.0, 5.0]),
    )
    torch.testing.assert_close(
        state["radii"],
        torch.tensor([0.1, 0.3, 0.1, 0.2, 0.2]),
    )

    pruned = trainer._prune_gaussians(
        params,
        optimizers,
        torch.tensor([False, True, False, False, True]),
        state=state,
    )
    assert pruned == 2
    torch.testing.assert_close(state["grad2d"], torch.tensor([1.0, 1.0, 2.0]))
    torch.testing.assert_close(state["count"], torch.tensor([4.0, 4.0, 5.0]))
    torch.testing.assert_close(state["radii"], torch.tensor([0.1, 0.1, 0.2]))


def test_supa_multiview_trainer_refinement_growth_cap_counts_split_overlap():
    trainer = _load_trainer_module()
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(4, 3)),
            "scales": torch.nn.Parameter(torch.zeros(4, 3)),
            "quats": torch.nn.Parameter(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                )
            ),
            "opacities": torch.nn.Parameter(torch.zeros(4)),
            "sh0": torch.nn.Parameter(torch.zeros(4, 1, 3)),
        }
    )
    state = {
        "grad2d": torch.ones(4),
        "count": torch.ones(4),
        "radii": torch.ones(4),
    }
    cfg = trainer.TrainConfig(
        data=Path("unused.npz"),
        result_dir=Path("unused"),
        refine_start_iter=0,
        refine_every=1,
        refine_stop_iter=10,
        reset_every=0,
        grow_grad2d=0.0,
        grow_scale3d=1.0,
        grow_scale2d=0.5,
        refine_scale2d_stop_iter=10,
        max_gaussians=6,
    )
    optimizer = trainer.create_optimizers(params, scene_scale=1000.0, cfg=cfg)

    stats = trainer._apply_refinement(
        params,
        optimizer,
        state,
        cfg,
        step=1,
        scene_scale=1000.0,
    )

    assert stats["candidate_duplicate"] == 0
    assert stats["candidate_split"] == 4
    assert stats["split"] == 2
    assert stats["num_gaussians"] == 6
    assert stats["growth_capacity"] == 2
    assert stats["growth_selected_split"] == 2
    assert stats["growth_dropped_split"] == 2
    assert stats["growth_score_candidate"]["finite"] == 4
    assert stats["growth_score_selected"]["finite"] == 2
    assert stats["growth_score_dropped"]["finite"] == 2
    assert params["means"].shape == (6, 3)


def test_supa_multiview_trainer_refinement_ignores_nonfinite_growth_signals():
    trainer = _load_trainer_module()
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(3, 3)),
            "scales": torch.nn.Parameter(torch.zeros(3, 3)),
            "quats": torch.nn.Parameter(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                )
            ),
            "opacities": torch.nn.Parameter(torch.zeros(3)),
            "sh0": torch.nn.Parameter(torch.zeros(3, 1, 3)),
        }
    )
    state = {
        "grad2d": torch.tensor([float("inf"), float("nan"), 1.0]),
        "count": torch.ones(3),
        "radii": torch.tensor([float("inf"), float("nan"), 0.1]),
    }
    cfg = trainer.TrainConfig(
        data=Path("unused.npz"),
        result_dir=Path("unused"),
        refine_start_iter=0,
        refine_every=1,
        refine_stop_iter=10,
        reset_every=0,
        grow_grad2d=0.5,
        grow_scale3d=1.0,
        max_gaussians=10,
    )
    optimizer = trainer.create_optimizers(params, scene_scale=1000.0, cfg=cfg)

    stats = trainer._apply_refinement(
        params,
        optimizer,
        state,
        cfg,
        step=1,
        scene_scale=1000.0,
    )

    assert stats["candidate_duplicate"] == 1
    assert stats["duplicated"] == 1
    assert stats["growth_score_candidate"]["finite"] == 1
    assert stats["grad2d_max"] == pytest.approx(1.0)
    assert torch.isfinite(state["grad2d"]).all()
    assert torch.isfinite(state["radii"]).all()


def test_supa_multiview_trainer_refine_profile_effective_scale2d():
    trainer = _load_trainer_module()

    manual = trainer.TrainConfig(
        data=Path("unused.npz"),
        result_dir=Path("unused"),
        train_crop_size=96,
        refine_profile="manual",
        refine_scale2d_stop_iter=7,
        refine_stop_iter=20,
    )
    crop_profile = trainer.TrainConfig(
        data=Path("unused.npz"),
        result_dir=Path("unused"),
        train_crop_size=96,
        refine_profile="crop-balanced",
        refine_scale2d_stop_iter=0,
        refine_stop_iter=20,
    )
    full_profile = trainer.TrainConfig(
        data=Path("unused.npz"),
        result_dir=Path("unused"),
        train_crop_size=96,
        refine_profile="full-stable",
        refine_scale2d_stop_iter=7,
        refine_stop_iter=20,
    )

    assert trainer._effective_refine_scale2d_stop_iter(manual) == 7
    assert trainer._effective_refine_scale2d_stop_iter(crop_profile) == 20
    assert trainer._effective_refine_scale2d_stop_iter(full_profile) == 0


def test_supa_multiview_trainer_refinement_smoke_cpu(tmp_path):
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pass

    trainer = _load_trainer_module()
    data_path = tmp_path / "tiny_refine.npz"
    result_dir = tmp_path / "refine_results"
    _write_tiny_npz(data_path, num_frames=3)

    cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=result_dir,
        data_type="npz",
        device="cpu",
        iterations=2,
        init_num_points=4,
        save_every=10,
        seed=29,
        save_ply=False,
        eval_max_frames=0,
        ssim_lambda=0.0,
        sh_degree=0,
        refine_start_iter=0,
        refine_every=1,
        refine_stop_iter=10,
        reset_every=0,
        grow_grad2d=-1.0,
        grow_scale3d=1e9,
        max_gaussians=8,
    )
    stats = trainer.train_multiview(cfg)
    metrics = json.loads(Path(stats["metrics"]).read_text(encoding="utf-8"))

    assert stats["num_gaussians"] == 8
    assert metrics["train"]["num_gaussians"][0] == 8
    assert metrics["train"]["last_num_gaussians"] == 8
    assert metrics["train"]["refinement_events"][0]["duplicated"] == 4


def test_supa_multiview_trainer_exposure_refinement_checkpoint_cpu(tmp_path):
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pass

    trainer = _load_trainer_module()
    data_path = tmp_path / "tiny_exposure.npz"
    result_dir = tmp_path / "exposure_results"
    _write_tiny_npz(data_path, num_frames=3)

    cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=result_dir,
        data_type="npz",
        device="cpu",
        iterations=2,
        init_num_points=4,
        save_every=1,
        seed=41,
        save_ply=False,
        refine_exposure=True,
        exposure_lr=1e-2,
        exposure_l2=1e-4,
        ssim_lambda=0.0,
        sh_degree=0,
    )
    stats = trainer.train_multiview(cfg)
    metrics = json.loads(Path(stats["metrics"]).read_text(encoding="utf-8"))
    ckpt = torch.load(stats["checkpoint"], map_location="cpu")

    assert metrics["config"]["refine_exposure"] is True
    assert metrics["train"]["exposure_reg_losses"] == [pytest.approx(0.0), pytest.approx(0.0)]
    assert metrics["diagnostics"]["exposure"] is not None
    assert metrics["diagnostics"]["exposure"]["log_scale"]["count"] == 9
    assert "exposure" in ckpt
    assert ckpt["exposure"]["log_scale"].shape == (3, 3)
    assert ckpt["exposure"]["bias"].shape == (3, 3)

    resume_cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=tmp_path / "exposure_resumed",
        data_type="npz",
        device="cpu",
        iterations=3,
        init_num_points=8,
        save_every=1,
        seed=43,
        save_ply=False,
        refine_exposure=True,
        resume=Path(stats["checkpoint"]),
        ssim_lambda=0.0,
        sh_degree=0,
    )
    resumed = trainer.train_multiview(resume_cfg)
    resumed_ckpt = torch.load(resumed["checkpoint"], map_location="cpu")
    assert resumed["start_step"] == 3
    assert "exposure" in resumed_ckpt
    assert resumed_ckpt["exposure"]["log_scale"].shape == (3, 3)


def test_supa_multiview_trainer_refinement_supa_smoke(tmp_path):
    _require_supa_devices(1, require_brcc=True)
    trainer = _load_trainer_module()
    data_path = tmp_path / "tiny_refine_supa.npz"
    result_dir = tmp_path / "refine_supa_results"
    _write_tiny_npz(data_path, num_frames=3)

    cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=result_dir,
        data_type="npz",
        device="supa:0",
        iterations=1,
        init_num_points=4,
        save_every=10,
        seed=31,
        save_ply=False,
        eval_max_frames=0,
        ssim_lambda=0.0,
        sh_degree=0,
        refine_start_iter=0,
        refine_every=1,
        refine_stop_iter=10,
        reset_every=0,
        grow_grad2d=-1.0,
        grow_scale3d=1e9,
        max_gaussians=6,
    )
    stats = trainer.train_multiview(cfg)
    metrics = json.loads(Path(stats["metrics"]).read_text(encoding="utf-8"))

    assert stats["num_gaussians"] == 6
    assert metrics["train"]["last_num_gaussians"] == 6
    assert metrics["train"]["refinement_events"][0]["duplicated"] == 2


def _supa_sh_adam_params(trainer, device: torch.device) -> torch.nn.ParameterDict:
    trainer.set_accelerator_device(device)
    sh_coeffs = trainer._sh_coeff_count(3)
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(4, 3, device=device)),
            "scales": torch.nn.Parameter(torch.zeros(4, 3, device=device)),
            "quats": torch.nn.Parameter(torch.zeros(4, 4, device=device)),
            "opacities": torch.nn.Parameter(torch.zeros(4, device=device)),
            "sh0": torch.nn.Parameter(torch.zeros(4, 1, 3, device=device)),
            "shN": torch.nn.Parameter(torch.zeros(4, sh_coeffs, 3, device=device)),
        }
    )


def test_supa_multiview_trainer_sh_param_adam_step_supa():
    _require_supa_devices(1)
    trainer = _load_trainer_module()
    device = torch.device("supa:0")
    params = _supa_sh_adam_params(trainer, device)
    cfg = trainer.TrainConfig(
        data=Path("unused.npz"),
        result_dir=Path("unused"),
        sh_degree=3,
    )
    optimizers = trainer.create_optimizers(params, scene_scale=1.0, cfg=cfg)
    optimizer = optimizers["shN"]

    optimizer.zero_grad(set_to_none=True)
    params["shN"].grad = torch.ones_like(params["shN"])
    assert optimizer.defaults["foreach"] is False
    optimizer.step()
    torch.supa.synchronize()

    assert params["shN"].grad is not None


def test_supa_multiview_trainer_optimizer_dict_adam_step_supa():
    _require_supa_devices(1)
    trainer = _load_trainer_module()
    device = torch.device("supa:0")
    params = _supa_sh_adam_params(trainer, device)
    cfg = trainer.TrainConfig(
        data=Path("unused.npz"),
        result_dir=Path("unused"),
        sh_degree=3,
    )
    optimizers = trainer.create_optimizers(params, scene_scale=1.0, cfg=cfg)

    trainer._optimizer_zero_grad(optimizers)
    for param in params.values():
        param.grad = torch.ones_like(param)
    assert all(optimizer.defaults["foreach"] is False for optimizer in optimizers.values())
    trainer._optimizer_step(optimizers)
    torch.supa.synchronize()

    assert all(param.grad is not None for param in params.values())


def test_supa_multiview_trainer_native_tiled_cli_builds_options(tmp_path):
    trainer = _load_trainer_module()
    cfg = trainer.TrainConfig(
        data=tmp_path / "data.npz",
        result_dir=tmp_path / "results",
        native_tiled="true",
        native_tiled_metadata=True,
        native_tiled_train=True,
        native_alpha_threshold=1.0 / 255.0,
    )

    assert trainer._supa_native_options(cfg) == {
        "native": True,
        "native_tiled": "true",
        "native_tiled_metadata": True,
        "native_tiled_train": True,
        "native_tiled_backward_mode": "parallel",
        "native_alpha_threshold": 1.0 / 255.0,
    }


def test_supa_multiview_trainer_native_flag_builds_full_chain_options(tmp_path):
    trainer = _load_trainer_module()
    cfg = trainer.TrainConfig(
        data=tmp_path / "data.npz",
        result_dir=tmp_path / "results",
        native=True,
    )

    assert trainer._supa_native_options(cfg) == {
        "native": True,
        "native_tiled": "auto",
        "native_tiled_metadata": True,
        "native_tiled_train": True,
        "native_tiled_backward_mode": "parallel",
        "native_alpha_threshold": 1.0 / 255.0,
    }


def test_supa_multiview_trainer_native_exact_alpha_overrides_profile(tmp_path):
    trainer = _load_trainer_module()
    cfg = trainer.TrainConfig(
        data=tmp_path / "data.npz",
        result_dir=tmp_path / "results",
        native=True,
        native_alpha_threshold=0.0,
    )

    assert trainer._supa_native_options(cfg) == {
        "native": True,
        "native_tiled": "auto",
        "native_tiled_metadata": True,
        "native_tiled_train": True,
        "native_tiled_backward_mode": "auto",
        "native_alpha_threshold": 0.0,
    }


def test_supa_multiview_trainer_native_threshold_builds_tiled_options(tmp_path):
    trainer = _load_trainer_module()
    cfg = trainer.TrainConfig(
        data=tmp_path / "data.npz",
        result_dir=tmp_path / "results",
        native_alpha_threshold=1.0 / 255.0,
    )

    assert trainer._supa_native_options(cfg) == {
        "native": True,
        "native_tiled": "auto",
        "native_tiled_metadata": True,
        "native_tiled_train": True,
        "native_tiled_backward_mode": "parallel",
        "native_alpha_threshold": 1.0 / 255.0,
    }


def test_supa_multiview_trainer_native_parallel_backward_mode_opts_in(tmp_path):
    trainer = _load_trainer_module()
    cfg = trainer.TrainConfig(
        data=tmp_path / "data.npz",
        result_dir=tmp_path / "results",
        native_tiled_backward_mode="parallel",
    )

    assert trainer._supa_native_options(cfg) == {
        "native": True,
        "native_tiled": "auto",
        "native_tiled_metadata": True,
        "native_tiled_train": True,
        "native_tiled_backward_mode": "parallel",
        "native_alpha_threshold": 1.0 / 255.0,
    }


def test_training_quality_gate_config_matches_trainer(tmp_path):
    trainer = _load_trainer_module()
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "training_quality_gate.py"
    )
    spec = importlib.util.spec_from_file_location("training_quality_gate_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    config = {
        **module.COMMON_CONFIG,
        **next(iter(module.CASES.values())),
        "data": tmp_path / "data.npz",
        "result_dir": tmp_path / "results",
    }
    cfg = trainer.TrainConfig(**config)

    assert module.GATE_NAME == "quality-gate-v2"
    assert cfg.data_type == "colmap"
    assert cfg.data_factor == 4
    assert cfg.iterations == 100
    assert cfg.init_num_points == 65535
    assert cfg.train_crop_size == 256
    assert cfg.native is True
    assert cfg.native_tiled == "auto"
    assert cfg.save_ply is False
    assert cfg.eval_max_frames == 8
    assert cfg.print_all_step_info is False
    assert not hasattr(cfg, "show_all_info")

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "step": 3,
                "config": {"device": "supa:0"},
                "train": {
                    "last_loss": 0.25,
                    "refinement_events": [],
                    "step_time_summary_s": {"median": 1.25, "mean": 1.5},
                },
                "eval": {},
                "diagnostics": {"params": {}},
            }
        ),
        encoding="utf-8",
    )
    summary = module._metric_summary(metrics_path, label="tiny")
    assert summary["median_step_time_s"] == 1.25
    assert summary["mean_step_time_s"] == 1.5


def test_supa_multiview_trainer_print_all_step_info_logs_every_step(tmp_path, capsys):
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pass

    trainer = _load_trainer_module()
    data_path = tmp_path / "tiny_multiview.npz"
    result_dir = tmp_path / "results"
    _write_tiny_npz(data_path, num_frames=3)

    cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=result_dir,
        data_type="npz",
        device="cpu",
        iterations=3,
        init_num_points=4,
        save_every=10,
        seed=13,
        save_ply=False,
        eval_max_frames=0,
        print_all_step_info=True,
    )
    trainer.train_multiview(cfg)

    output = capsys.readouterr().out
    assert "step 0001/0003" in output
    assert "step 0002/0003" in output
    assert "step 0003/0003" in output
    for step in range(1, 4):
        assert re.search(rf"step {step:04d}/0003 .* psnr=\d+\.\d{{3}}\b", output)
    assert (result_dir / "ckpts" / "step_0001.pt").is_file()
    assert not (result_dir / "ckpts" / "step_0002.pt").exists()
    assert (result_dir / "ckpts" / "step_0003.pt").is_file()
    assert not (result_dir / "ply" / "step_0001_points.ply").exists()
    assert not (result_dir / "ply" / "step_0003_points.ply").exists()


def test_supa_multiview_trainer_print_debug_info_logs_operations(tmp_path, capsys):
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pass

    trainer = _load_trainer_module()
    data_path = tmp_path / "tiny_multiview.npz"
    _write_tiny_npz(data_path, num_frames=3)

    cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=tmp_path / "results",
        data_type="npz",
        device="cpu",
        iterations=1,
        init_num_points=4,
        save_every=10,
        seed=13,
        save_ply=False,
        eval_max_frames=0,
        ssim_lambda=0.0,
        sh_degree=0,
        print_debug_info=True,
    )
    trainer.train_multiview(cfg)

    output = capsys.readouterr().out
    timed = (
        r"\[debug\] "
        r"wall=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}[+-]\d{2}:\d{2} "
        r"elapsed_s=\d+\.\d{6} "
        r"delta_s=\d+\.\d{6} "
    )
    for event in (
        "init.begin",
        "init.load_data.done",
        "train.step.begin",
        "train.rasterize.begin",
        "train.render_loss.done",
        "train.backward.done",
        "train.optimizer_step.done",
        "eval.begin",
        "final.diagnostics.done",
    ):
        assert re.search(timed + re.escape(event), output)
    assert re.search(
        timed
        + r"train\.step_metrics\.done .*psnr="
        + r"\d+\.\d+",
        output,
    )


def test_supa_multiview_trainer_train_crop_logs_and_saves_crop(tmp_path, capsys):
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pass

    trainer = _load_trainer_module()
    data_path = tmp_path / "tiny_multiview.npz"
    result_dir = tmp_path / "results"
    _write_tiny_npz(data_path, num_frames=3)

    cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=result_dir,
        data_type="npz",
        device="cpu",
        iterations=1,
        init_num_points=4,
        save_every=10,
        seed=17,
        save_ply=False,
        eval_max_frames=0,
        print_all_step_info=True,
        train_crop_size=4,
        near_plane=0.02,
        far_plane=1000.0,
        radius_clip=0.5,
        log_step_time=True,
    )
    trainer.train_multiview(cfg)

    output = capsys.readouterr().out
    assert "time=" in output
    assert "crop=4x4+" in output

    with Image.open(result_dir / "renders" / "train_step_0001.png") as image:
        assert image.size == (4, 4)

    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["train_crop_size"] == 4
    assert metrics["config"]["near_plane"] == 0.02
    assert metrics["config"]["far_plane"] == 1000.0
    assert metrics["config"]["radius_clip"] == 0.5
    assert metrics["config"]["log_step_time"] is True
    assert len(metrics["train"]["step_times_s"]) == 1
    assert metrics["train"]["last_step_time_s"] >= 0.0
    summary = metrics["train"]["step_time_summary_s"]
    assert summary["count"] == 1
    assert summary["median"] == pytest.approx(metrics["train"]["last_step_time_s"])


def test_supa_multiview_trainer_native_tiled_train_smoke(tmp_path, capsys):
    _require_supa_devices(1, require_brcc=True)
    trainer = _load_trainer_module()

    data_path = tmp_path / "tiny_multiview_supa.npz"
    result_dir = tmp_path / "native_tiled_results"
    _write_tiny_npz(data_path, num_frames=3)

    cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=result_dir,
        data_type="npz",
        device="supa:0",
        iterations=1,
        init_num_points=4,
        save_every=10,
        seed=19,
        native=True,
        save_ply=False,
        eval_max_frames=0,
        print_all_step_info=True,
    )
    stats = trainer.train_multiview(cfg)

    output = capsys.readouterr().out
    assert "variant=tiled" in output
    metrics = json.loads(Path(stats["metrics"]).read_text(encoding="utf-8"))
    assert metrics["train"]["last_projection_backend"] == "supa_native"
    assert metrics["train"]["last_composite_backend"] == "supa_native"
    assert metrics["train"]["last_composite_variant"] == "tiled"
    assert metrics["train"]["last_tile_count_backend"] == "supa_native"
    assert (
        metrics["train"]["last_tiled_metadata_backend"]
        == "supa_native_alpha_threshold"
    )
    assert metrics["composite_variant"] == "full_scan"
    assert metrics["tiled_metadata_backend"] == "supa_native_alpha_threshold"
    assert stats["tiled_metadata_backend"] == "supa_native_alpha_threshold"
    assert Path(stats["final_render"]).is_file()


def test_supa_multiview_trainer_distributed_supa_smoke(tmp_path):
    _require_supa_devices(2, require_brcc=True)
    trainer = _load_trainer_module()
    data_path = tmp_path / "tiny_multiview_dist.npz"
    result_dir = tmp_path / "distributed_results"
    _write_tiny_npz(data_path, num_frames=5)

    single_seed_cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=tmp_path / "single_seed",
        data_type="npz",
        device="cpu",
        iterations=1,
        init_num_points=4,
        save_every=1,
        seed=37,
        eval_max_frames=3,
    )
    single_seed_stats = trainer.train_multiview(single_seed_cfg)
    single_seed_ckpt = torch.load(single_seed_stats["checkpoint"], map_location="cpu")
    assert single_seed_ckpt["step"] == 1
    assert single_seed_ckpt["splats"]["means"].shape == (4, 3)
    assert "optimizer" in single_seed_ckpt

    cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=result_dir,
        data_type="npz",
        device="supa:0",
        iterations=1,
        init_num_points=4,
        save_every=1,
        seed=23,
        distributed=True,
        distributed_world_size=2,
        eval_max_frames=3,
    )
    stats = trainer.train_multiview(cfg)

    assert stats["num_gaussians"] == 4
    assert stats["step"] == 1
    assert len(stats["losses"]) == 1
    assert stats["start_step"] == 1
    assert Path(stats["final_render"]) == result_dir / "renders" / "final.png"
    assert Path(stats["checkpoint"]) == result_dir / "ckpts" / "final.pt"
    assert Path(stats["ply"]) == result_dir / "ply" / "final_points.ply"
    assert Path(stats["final_render"]).is_file()
    assert Path(stats["checkpoint"]).is_file()
    assert Path(stats["ply"]).is_file()
    assert Path(stats["metrics"]).is_file()
    global_ckpt = torch.load(stats["checkpoint"], map_location="cpu")
    assert global_ckpt["step"] == 1
    assert global_ckpt["distributed"] is True
    assert global_ckpt["world_size"] == 2
    assert global_ckpt["splats"]["means"].shape == (4, 3)
    assert len(global_ckpt["rank_checkpoints"]) == 2
    assert len(global_ckpt["rank_shards"]) == 2
    assert "optimizer" in global_ckpt
    assert (result_dir / "ckpts" / "step_0001.pt").is_file()
    assert (result_dir / "ply" / "step_0001_points.ply").is_file()
    _assert_two_rank_train_step_grid(result_dir, 1)
    metrics = json.loads(Path(stats["metrics"]).read_text(encoding="utf-8"))
    assert metrics["distributed"] is True
    assert metrics["distributed_backend"] == "supa_gather"
    assert metrics["world_size"] == 2
    assert metrics["num_gaussians"] == 4
    assert metrics["train"]["num_steps"] == 1
    _assert_three_frame_distributed_eval(result_dir, metrics)
    assert metrics["final_render"] == str(result_dir / "renders" / "final.png")
    assert metrics["checkpoint"] == str(result_dir / "ckpts" / "final.pt")
    assert metrics["ply"] == str(result_dir / "ply" / "final_points.ply")
    assert len(metrics["ranks"]) == 2
    for rank, summary in enumerate(metrics["ranks"]):
        assert summary["world_rank"] == rank
        assert summary["local_num_gaussians"] == 2
        assert Path(summary["metrics"]).is_file()
        assert Path(summary["final_render"]).is_file()
        assert Path(summary["checkpoint"]).is_file()
        assert Path(summary["ply"]).is_file()
        assert (
            result_dir
            / f"rank_{rank:04d}"
            / "ply"
            / "step_0001_points.ply"
        ).is_file()
        ckpt = torch.load(summary["checkpoint"], map_location="cpu")
        assert ckpt["step"] == 1
        assert ckpt["splats"]["means"].shape == (2, 3)

    single_resume_cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=tmp_path / "single_from_distributed",
        data_type="npz",
        device="cpu",
        iterations=2,
        init_num_points=8,
        save_every=1,
        seed=101,
        resume=Path(stats["checkpoint"]),
    )
    single_resume_stats = trainer.train_multiview(single_resume_cfg)
    assert single_resume_stats["start_step"] == 2
    assert single_resume_stats["step"] == 2
    assert single_resume_stats["num_gaussians"] == 4
    single_resume_ckpt = torch.load(single_resume_stats["checkpoint"], map_location="cpu")
    assert single_resume_ckpt["step"] == 2
    assert single_resume_ckpt["splats"]["means"].shape == (4, 3)

    single_to_dist_dir = tmp_path / "distributed_from_single"
    single_to_dist_cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=single_to_dist_dir,
        data_type="npz",
        device="supa:0",
        iterations=2,
        init_num_points=8,
        save_every=1,
        seed=41,
        distributed=True,
        distributed_world_size=2,
        eval_max_frames=3,
        resume=Path(single_seed_stats["checkpoint"]),
    )
    single_to_dist_stats = trainer.train_multiview(single_to_dist_cfg)

    assert single_to_dist_stats["start_step"] == 2
    assert single_to_dist_stats["step"] == 2
    assert single_to_dist_stats["num_gaussians"] == 4
    assert len(single_to_dist_stats["losses"]) == 1
    assert Path(single_to_dist_stats["checkpoint"]) == single_to_dist_dir / "ckpts" / "final.pt"
    assert Path(single_to_dist_stats["checkpoint"]).is_file()
    assert Path(single_to_dist_stats["ply"]).is_file()
    single_to_dist_ckpt = torch.load(single_to_dist_stats["checkpoint"], map_location="cpu")
    assert single_to_dist_ckpt["step"] == 2
    assert single_to_dist_ckpt["distributed"] is True
    assert single_to_dist_ckpt["world_size"] == 2
    assert single_to_dist_ckpt["splats"]["means"].shape == (4, 3)
    assert "optimizer" in single_to_dist_ckpt
    _assert_two_rank_train_step_grid(single_to_dist_dir, 2)
    assert not torch.allclose(
        single_to_dist_ckpt["splats"]["means"],
        single_seed_ckpt["splats"]["means"],
    )
    single_to_dist_metrics = json.loads(
        Path(single_to_dist_stats["metrics"]).read_text(encoding="utf-8")
    )
    assert single_to_dist_metrics["start_step"] == 2
    assert single_to_dist_metrics["step"] == 2
    assert single_to_dist_metrics["config"]["resume"] == str(single_seed_stats["checkpoint"])
    _assert_three_frame_distributed_eval(single_to_dist_dir, single_to_dist_metrics)
    for summary in single_to_dist_metrics["ranks"]:
        ckpt = torch.load(summary["checkpoint"], map_location="cpu")
        assert ckpt["step"] == 2
        assert ckpt["splats"]["means"].shape == (2, 3)

    resumed_dir = tmp_path / "distributed_resumed"
    resumed_cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=resumed_dir,
        data_type="npz",
        device="supa:0",
        iterations=2,
        init_num_points=8,
        save_every=1,
        seed=99,
        distributed=True,
        distributed_world_size=2,
        eval_max_frames=3,
        resume=Path(stats["checkpoint"]),
    )
    resumed_stats = trainer.train_multiview(resumed_cfg)

    assert resumed_stats["start_step"] == 2
    assert resumed_stats["step"] == 2
    assert resumed_stats["num_gaussians"] == 4
    assert len(resumed_stats["losses"]) == 1
    assert Path(resumed_stats["checkpoint"]) == resumed_dir / "ckpts" / "final.pt"
    assert Path(resumed_stats["checkpoint"]).is_file()
    assert Path(resumed_stats["ply"]).is_file()
    resumed_ckpt = torch.load(resumed_stats["checkpoint"], map_location="cpu")
    assert resumed_ckpt["step"] == 2
    assert resumed_ckpt["distributed"] is True
    assert resumed_ckpt["world_size"] == 2
    assert resumed_ckpt["splats"]["means"].shape == (4, 3)
    assert "optimizer" in resumed_ckpt
    _assert_two_rank_train_step_grid(resumed_dir, 2)
    assert not torch.allclose(
        resumed_ckpt["splats"]["means"],
        global_ckpt["splats"]["means"],
    )
    resumed_metrics = json.loads(Path(resumed_stats["metrics"]).read_text(encoding="utf-8"))
    assert resumed_metrics["start_step"] == 2
    assert resumed_metrics["step"] == 2
    assert resumed_metrics["config"]["resume"] == str(stats["checkpoint"])
    assert resumed_metrics["config"]["init_num_points"] == 8
    _assert_three_frame_distributed_eval(resumed_dir, resumed_metrics)
    for summary in resumed_metrics["ranks"]:
        ckpt = torch.load(summary["checkpoint"], map_location="cpu")
        assert ckpt["step"] == 2
        assert ckpt["splats"]["means"].shape == (2, 3)


def test_supa_multiview_trainer_upsamples_sparse_input_points(tmp_path):
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pass

    trainer = _load_trainer_module()

    images = np.zeros((2, 6, 6, 3), dtype=np.float32)
    viewmats = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    Ks = np.tile(
        np.array([[7.0, 0.0, 3.0], [0.0, 7.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        (2, 1, 1),
    )
    points = np.array(
        [
            [-0.1, -0.1, 2.0],
            [0.1, -0.1, 2.1],
            [-0.1, 0.1, 2.2],
            [0.1, 0.1, 2.3],
        ],
        dtype=np.float32,
    )
    point_colors = np.full((4, 3), 0.5, dtype=np.float32)
    data_path = tmp_path / "sparse_points.npz"
    np.savez(
        data_path,
        images=images,
        viewmats=viewmats,
        Ks=Ks,
        points=points,
        point_colors=point_colors,
        train_indices=np.array([0], dtype=np.int64),
        eval_indices=np.array([1], dtype=np.int64),
        scene_scale=np.array(1.0, dtype=np.float32),
    )

    cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=tmp_path / "results",
        data_type="npz",
        device="cpu",
        iterations=1,
        init_num_points=6,
        save_every=1,
        seed=11,
    )
    stats = trainer.train_multiview(cfg)

    assert stats["num_gaussians"] == 6
    assert Path(stats["metrics"]).is_file()
    ckpt = torch.load(stats["checkpoint"], map_location="cpu")
    assert ckpt["splats"]["means"].shape == (6, 3)
    assert torch.unique(ckpt["splats"]["means"], dim=0).shape[0] > 4


def test_supa_multiview_trainer_npz_masks_drive_metrics_cpu(tmp_path):
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pass

    trainer = _load_trainer_module()

    images = np.zeros((2, 6, 6, 3), dtype=np.float32)
    images[0, :, :, 0] = 0.9
    images[1, :, :, 1] = 0.8
    viewmats = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    Ks = np.tile(
        np.array([[7.0, 0.0, 3.0], [0.0, 7.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        (2, 1, 1),
    )
    points = np.array(
        [
            [-0.1, -0.1, 2.0],
            [0.1, -0.1, 2.1],
            [-0.1, 0.1, 2.2],
            [0.1, 0.1, 2.3],
        ],
        dtype=np.float32,
    )
    masks = np.zeros((2, 6, 6, 1), dtype=np.float32)
    masks[0, :3, :3, 0] = 1.0
    masks[1, 2:5, 1:5, 0] = 1.0
    data_path = tmp_path / "masked_points.npz"
    np.savez(
        data_path,
        images=images,
        viewmats=viewmats,
        Ks=Ks,
        masks=masks,
        points=points,
        point_colors=np.full((4, 3), 0.5, dtype=np.float32),
        train_indices=np.array([0], dtype=np.int64),
        eval_indices=np.array([1], dtype=np.int64),
        scene_scale=np.array(1.0, dtype=np.float32),
    )

    data = trainer.load_multiview_data(
        trainer.TrainConfig(
            data=data_path,
            result_dir=tmp_path / "unused",
            data_type="npz",
            device="cpu",
        )
    )
    assert data.train_frames[0].mask is not None
    assert int(data.train_frames[0].mask.sum().item()) == 9
    assert data.eval_frames[0].mask is not None
    assert int(data.eval_frames[0].mask.sum().item()) == 12

    cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=tmp_path / "results",
        data_type="npz",
        device="cpu",
        iterations=1,
        init_num_points=4,
        save_every=1,
        seed=17,
    )
    stats = trainer.train_multiview(cfg)

    metrics = json.loads(Path(stats["metrics"]).read_text(encoding="utf-8"))
    assert metrics["eval"]["num_frames"] == 1
    assert metrics["eval"]["frames"][0]["mask_pixels"] == 12
    assert metrics["eval"]["l1"] is not None
    assert metrics["eval"]["mse"] is not None


def test_supa_multiview_trainer_npz_split_indices_validate(tmp_path):
    trainer = _load_trainer_module()

    images = np.zeros((2, 6, 6, 3), dtype=np.float32)
    viewmats = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    Ks = np.tile(
        np.array([[7.0, 0.0, 3.0], [0.0, 7.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        (2, 1, 1),
    )
    common = {
        "images": images,
        "viewmats": viewmats,
        "Ks": Ks,
        "points": np.array([[0.0, 0.0, 2.0]], dtype=np.float32),
    }

    out_of_range_path = tmp_path / "bad_index.npz"
    np.savez(
        out_of_range_path,
        **common,
        train_indices=np.array([0, 2], dtype=np.int64),
        eval_indices=np.array([1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="train_indices contains out-of-range"):
        trainer.load_multiview_data(
            trainer.TrainConfig(
                data=out_of_range_path,
                result_dir=tmp_path / "unused",
                data_type="npz",
                device="cpu",
            )
        )

    empty_eval_path = tmp_path / "empty_eval.npz"
    np.savez(
        empty_eval_path,
        **common,
        train_indices=np.array([0], dtype=np.int64),
        eval_indices=np.array([], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="eval_indices must contain"):
        trainer.load_multiview_data(
            trainer.TrainConfig(
                data=empty_eval_path,
                result_dir=tmp_path / "unused",
                data_type="npz",
                device="cpu",
            )
        )


def test_supa_multiview_trainer_rejects_rank_shard_distributed_resume(tmp_path):
    trainer = _load_trainer_module()

    splats = {
        "means": torch.zeros(4, 3),
        "scales": torch.zeros(4, 3),
        "quats": torch.zeros(4, 4),
        "opacities": torch.zeros(4),
        "sh0": torch.zeros(4, 1, 3),
    }

    single_ckpt = tmp_path / "single.pt"
    torch.save({"step": 1, "config": {"distributed": False}, "splats": splats}, single_ckpt)
    assert trainer._load_distributed_resume_checkpoint(single_ckpt)["step"] == 1

    root_ckpt = tmp_path / "distributed_root.pt"
    torch.save(
        {
            "step": 2,
            "config": {"distributed": True},
            "distributed": True,
            "world_size": 2,
            "rank_shards": [
                {"world_rank": 0, "start": 0, "stop": 2},
                {"world_rank": 1, "start": 2, "stop": 4},
            ],
            "splats": splats,
        },
        root_ckpt,
    )
    assert trainer._load_distributed_resume_checkpoint(root_ckpt)["step"] == 2

    rank_shard_ckpt = tmp_path / "rank_shard.pt"
    torch.save(
        {
            "step": 2,
            "config": {"distributed": True},
            "splats": {name: value[:2].clone() for name, value in splats.items()},
        },
        rank_shard_ckpt,
    )
    with pytest.raises(ValueError, match="per-rank distributed shard"):
        trainer._load_distributed_resume_checkpoint(rank_shard_ckpt)

    broken_root_ckpt = tmp_path / "broken_root.pt"
    torch.save(
        {
            "step": 2,
            "config": {"distributed": True},
            "distributed": True,
            "world_size": 2,
            "splats": splats,
        },
        broken_root_ckpt,
    )
    with pytest.raises(ValueError, match="missing root metadata"):
        trainer._load_distributed_resume_checkpoint(broken_root_ckpt)


def test_supa_multiview_trainer_resumes_from_checkpoint(tmp_path):
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pass

    trainer = _load_trainer_module()

    images = np.zeros((3, 6, 6, 3), dtype=np.float32)
    images[0, :, :, 0] = 0.7
    images[1, :, :, 1] = 0.6
    images[2, :, :, 2] = 0.5
    viewmats = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    viewmats[1, 0, 3] = 0.05
    Ks = np.tile(
        np.array([[7.0, 0.0, 3.0], [0.0, 7.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        (3, 1, 1),
    )
    points = np.array(
        [
            [-0.1, -0.1, 2.0],
            [0.1, -0.1, 2.1],
            [-0.1, 0.1, 2.2],
            [0.1, 0.1, 2.3],
        ],
        dtype=np.float32,
    )
    point_colors = np.full((4, 3), 0.5, dtype=np.float32)
    data_path = tmp_path / "resume_points.npz"
    np.savez(
        data_path,
        images=images,
        viewmats=viewmats,
        Ks=Ks,
        points=points,
        point_colors=point_colors,
        train_indices=np.array([0, 1], dtype=np.int64),
        eval_indices=np.array([2], dtype=np.int64),
        scene_scale=np.array(1.0, dtype=np.float32),
    )

    first_cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=tmp_path / "first",
        data_type="npz",
        device="cpu",
        iterations=1,
        init_num_points=4,
        save_every=1,
        seed=13,
    )
    first_stats = trainer.train_multiview(first_cfg)
    first_ckpt = torch.load(first_stats["checkpoint"], map_location="cpu")
    first_means = first_ckpt["splats"]["means"].clone()
    first_step_ckpt = Path(first_cfg.result_dir) / "ckpts" / "step_0001.pt"
    assert first_step_ckpt.is_file()

    resume_cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=tmp_path / "resumed",
        data_type="npz",
        device="cpu",
        iterations=3,
        init_num_points=6,
        save_every=1,
        seed=99,
        resume=Path(first_stats["checkpoint"]),
    )
    resumed_stats = trainer.train_multiview(resume_cfg)

    assert resumed_stats["start_step"] == 2
    assert resumed_stats["step"] == 3
    assert len(resumed_stats["losses"]) == 2
    assert resumed_stats["num_gaussians"] == 4
    metrics = json.loads(Path(resumed_stats["metrics"]).read_text(encoding="utf-8"))
    assert metrics["step"] == 3
    assert metrics["start_step"] == 2
    assert metrics["config"]["resume"] == str(first_stats["checkpoint"])
    assert metrics["config"]["init_num_points"] == 6
    assert metrics["train"]["num_steps"] == 2
    assert metrics["eval"]["num_frames"] == 1
    assert metrics["eval"]["grid"] is not None
    assert Path(metrics["eval"]["grid"]).is_file()
    resumed_ckpt = torch.load(resumed_stats["checkpoint"], map_location="cpu")
    assert resumed_ckpt["step"] == 3
    assert resumed_ckpt["splats"]["means"].shape == (4, 3)
    assert "optimizer" in resumed_ckpt
    assert not torch.allclose(resumed_ckpt["splats"]["means"], first_means)

    periodic_resume_cfg = trainer.TrainConfig(
        data=data_path,
        result_dir=tmp_path / "resumed_from_periodic",
        data_type="npz",
        device="cpu",
        iterations=3,
        init_num_points=8,
        save_every=1,
        seed=101,
        resume=first_step_ckpt,
    )
    periodic_stats = trainer.train_multiview(periodic_resume_cfg)
    assert periodic_stats["start_step"] == 2
    assert periodic_stats["step"] == 3
    assert periodic_stats["num_gaussians"] == 4
