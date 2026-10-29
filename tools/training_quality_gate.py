#!/usr/bin/env /usr/bin/python3
"""Run or summarize the fixed training-quality gate.

This is intentionally narrow. It defines one versioned quality gate that can be
re-run after training/rendering changes without repeatedly inventing new grids.
Exploratory sweeps should go in separate experiment directories.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gsplat._logging import configure_logging, format_table, get_logger, log_info  # noqa: E402


LOGGER = get_logger(__name__)
GATE_NAME = "quality-gate-v2"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".agents" / "output" / "train" / GATE_NAME

COMMON_CONFIG: dict[str, Any] = {
    "data_type": "colmap",
    "device": "supa:0",
    "iterations": 100,
    "init_num_points": 65535,
    "data_factor": 4,
    "save_every": 50,
    "seed": 42,
    "native": True,
    "native_tiled": "auto",
    "save_ply": False,
    "eval_max_frames": 8,
    "print_all_step_info": False,
    "train_crop_size": 256,
    "radius_clip": 0.0,
    "ssim_lambda": 0.2,
    "shuffle_train": True,
    "means_lr_final_factor": 0.01,
    "sh_degree": 3,
    "refine_profile": "crop-balanced",
    "refine_gaussians": True,
    "refine_start_iter": 25,
    "refine_stop_iter": 100,
    "refine_every": 25,
    "reset_every": 0,
    "pause_refine_after_reset": 0,
    "prune_opa": 0.005,
    "grow_grad2d": 0.0002,
    "grow_scale3d": 0.01,
    "grow_scale2d": 0.05,
    "prune_scale3d": 0.1,
    "prune_scale2d": 0.15,
    "max_gaussians": 65535,
}

CASES: dict[str, dict[str, Any]] = {
    "minecraft_undistorted_crop256_native_i100": {
        # The sibling "improved" directory in the current checkout contains
        # only sparse COLMAP files. This gate points at the complete image +
        # sparse dataset so --run is immediately reproducible.
        "data": REPO_ROOT / ".agents/dataset/minecraft_dungle_pyramid_undistorted",
    }
}

def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _add_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help: str,
) -> None:
    parser.add_argument(
        name,
        type=_parse_bool,
        default=default,
        metavar="true|false",
        help=f"{help} Defaults to {str(default).lower()}.",
    )


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _load_trainer_module():
    path = REPO_ROOT / "applications" / "supa_multiview_trainer.py"
    spec = importlib.util.spec_from_file_location("supa_multiview_trainer_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load trainer module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metric_summary(metrics_path: Path, *, label: str) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = metrics.get("config", {})
    train = metrics.get("train", {})
    eval_metrics = metrics.get("eval", {})
    diagnostics = metrics.get("diagnostics", {})
    params = diagnostics.get("params", {})
    step_time_summary = train.get("step_time_summary_s") or {}
    return {
        "label": label,
        "metrics": str(metrics_path),
        "step": metrics.get("step"),
        "data": cfg.get("data"),
        "device": cfg.get("device"),
        "iterations": cfg.get("iterations"),
        "train_crop_size": cfg.get("train_crop_size"),
        "refine_scale2d_stop_iter": cfg.get("refine_scale2d_stop_iter"),
        "grow_scale2d": cfg.get("grow_scale2d"),
        "max_gaussians": cfg.get("max_gaussians"),
        "num_gaussians": metrics.get("num_gaussians"),
        "eval_psnr": eval_metrics.get("psnr"),
        "eval_l1": eval_metrics.get("l1"),
        "eval_mse": eval_metrics.get("mse"),
        "last_loss": train.get("last_loss"),
        "median_step_time_s": step_time_summary.get("median"),
        "mean_step_time_s": step_time_summary.get("mean"),
        "refinement_events": len(train.get("refinement_events", [])),
        "last_refinement": train.get("last_refinement"),
        "projection_backend": metrics.get("projection_backend"),
        "composite_backend": metrics.get("composite_backend"),
        "composite_variant": metrics.get("composite_variant"),
        "tile_count_backend": metrics.get("tile_count_backend"),
        "tiled_metadata_backend": metrics.get("tiled_metadata_backend"),
        "opacity": params.get("opacity"),
        "max_scale_scene_fraction": params.get("max_scale_scene_fraction"),
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "label",
        "step",
        "crop",
        "scale2d_stop",
        "gaussians",
        "eval_psnr",
        "eval_l1",
        "last_loss",
        "median_step_s",
        "events",
        "variant",
    ]
    values = []
    for row in rows:
        values.append(
            [
                row["label"],
                row.get("step"),
                row.get("train_crop_size"),
                row.get("refine_scale2d_stop_iter"),
                row.get("num_gaussians"),
                None
                if row.get("eval_psnr") is None
                else f"{float(row['eval_psnr']):.4f}",
                None if row.get("eval_l1") is None else f"{float(row['eval_l1']):.4f}",
                None
                if row.get("last_loss") is None
                else f"{float(row['last_loss']):.4f}",
                None
                if row.get("median_step_time_s") is None
                else f"{float(row['median_step_time_s']):.3f}",
                row.get("refinement_events"),
                row.get("composite_variant"),
            ]
        )
    log_info(LOGGER, format_table("Training Quality Gate", headers, values))


def _run_case(name: str, output_root: Path, *, force: bool) -> dict[str, Any]:
    trainer = _load_trainer_module()
    case_dir = output_root / name
    metrics_path = case_dir / "metrics.json"
    if metrics_path.is_file() and not force:
        return _metric_summary(metrics_path, label=name)
    if case_dir.exists() and force:
        shutil.rmtree(case_dir)

    config = {**COMMON_CONFIG, **CASES[name], "result_dir": case_dir}
    missing = [str(value) for key, value in config.items() if key == "data" and not value.exists()]
    if missing:
        raise FileNotFoundError(f"missing dataset(s): {', '.join(missing)}")
    data_path = Path(config["data"])
    if config.get("data_type") == "colmap":
        image_root = data_path / "images"
        if not image_root.is_dir():
            raise FileNotFoundError(
                f"COLMAP quality gate data must include an images/ folder: {image_root}"
            )

    cfg = trainer.TrainConfig(**config)
    started = time.time()
    stats = trainer.train_multiview(cfg)
    summary = _metric_summary(Path(stats["metrics"]), label=name)
    summary["elapsed_s"] = time.time() - started
    return summary


def _write_manifest(path: Path, revision: str) -> None:
    payload = {
        "gate": GATE_NAME,
        "revision": revision,
        "purpose": (
            "Fixed canonical COLMAP quality gate for training/rendering changes. "
            "Do not add cases here unless the gate version is bumped."
        ),
        "rerun_policy": [
            "Run after code changes in applications/supa_multiview_trainer.py, gsplat/rendering.py, gsplat/_torch_fallback.py, gsplat/_device.py, or gsplat/supa/.",
            "Run before changing trainer quality defaults.",
            "Do not use this gate for open-ended hyperparameter search; create a separate .agents/output/train experiment directory instead.",
        ],
        "common_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in COMMON_CONFIG.items()
        },
        "cases": {
            name: {
                key: str(value) if isinstance(value, Path) else value
                for key, value in case.items()
            }
            for name, case in CASES.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    _add_bool_arg(
        parser,
        "--run",
        default=False,
        help="Run gate cases. Without this flag, existing current-version case metrics are summarized.",
    )
    parser.add_argument(
        "--case",
        choices=sorted(CASES),
        action="append",
        default=None,
        help="Run/summarize selected case(s). May be passed multiple times.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to .agents/output/train/quality-gate-v2/<rev>.",
    )
    _add_bool_arg(
        parser,
        "--force",
        default=False,
        help="Replace existing case outputs.",
    )
    args = parser.parse_args()

    revision = _git_revision()
    output_root = args.output_root or DEFAULT_OUTPUT_ROOT / revision
    output_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(output_root / "manifest.json", revision)

    selected = args.case or sorted(CASES)
    rows = []
    if args.run:
        for name in selected:
            log_info(LOGGER, f"running {name}")
            rows.append(_run_case(name, output_root, force=args.force))
    else:
        for name in selected:
            metrics_path = output_root / name / "metrics.json"
            if metrics_path.is_file():
                rows.append(_metric_summary(metrics_path, label=name))

    summary = {
        "gate": GATE_NAME,
        "revision": revision,
        "mode": "run" if args.run else "current-summary",
        "output_root": str(output_root),
        "rows": rows,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_table(rows)
    log_info(LOGGER, f"wrote {summary_path}")


if __name__ == "__main__":
    main()
