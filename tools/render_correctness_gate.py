#!/usr/bin/env /usr/bin/python3
"""Run a small, fixed SUPA rendering correctness gate.

The goal is to produce a durable parity artifact without expanding the test
surface every time training quality is discussed. Full pytest coverage remains
in tests/test_supa_fallback.py; this script selects a few representative
forward/backward comparisons and writes numeric errors to .agents/output/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gsplat._logging import configure_logging, get_logger, log_info  # noqa: E402

import tests.test_supa_fallback as fallback_tests  # noqa: E402


LOGGER = get_logger(__name__)

_NATIVE_CORE_OPTIONS = {
    "native": True,
    "native_tiled": "false",
    "native_tiled_metadata": False,
    "native_tiled_train": False,
    "native_tiled_backward_mode": "auto",
    "native_alpha_threshold": 0.0,
}

_NATIVE_TILED_TRAIN_OPTIONS = {
    "native": True,
    "native_tiled": "false",
    "native_tiled_metadata": False,
    "native_tiled_train": True,
    "native_tiled_backward_mode": "auto",
    "native_alpha_threshold": 0.0,
}


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _require_supa_device() -> torch.device:
    try:
        import torch_br  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError as exc:
        raise RuntimeError("torch_br is not installed") from exc
    if not hasattr(torch, "supa") or not torch.supa.is_available():
        raise RuntimeError("SUPA device is not available")
    return torch.device("supa:0")


def _max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    if lhs.numel() == 0:
        return 0.0
    return float((lhs.detach().cpu() - rhs.detach().cpu()).abs().max().item())


def _grad_errors(
    cpu_scene: dict[str, torch.Tensor],
    supa_scene: dict[str, torch.Tensor],
    names: tuple[str, ...],
) -> dict[str, float]:
    out = {}
    for name in names:
        cpu_grad = cpu_scene[name].grad
        supa_grad = supa_scene[name].grad
        if cpu_grad is None or supa_grad is None:
            raise AssertionError(f"missing gradient for {name}")
        out[name] = _max_abs(supa_grad, cpu_grad)
    return out


def _case_fallback_basic(device: torch.device) -> dict[str, Any]:
    cpu_scene = fallback_tests._make_basic_scene("cpu")
    supa_scene = fallback_tests._make_basic_scene(device)

    cpu_renders, cpu_alphas, _ = fallback_tests._call_reference_fallback(
        cpu_scene, chunk_size=2
    )
    supa_renders, supa_alphas, meta = fallback_tests._call_public_rasterization(
        supa_scene, render_mode="RGB+D", channel_chunk=2
    )
    if meta["backend"] != "torch_fallback":
        raise AssertionError(f"expected torch_fallback backend, got {meta['backend']}")

    (cpu_renders.sum() + cpu_alphas.sum()).backward()
    (supa_renders.sum() + supa_alphas.sum()).backward()
    return {
        "name": "supa_fallback_basic_rgb_depth",
        "status": "passed",
        "meta": {
            "backend": meta.get("backend"),
            "projection_backend": meta.get("projection_backend"),
            "composite_backend": meta.get("composite_backend"),
            "composite_variant": meta.get("composite_variant"),
        },
        "errors": {
            "render_max_abs": _max_abs(supa_renders, cpu_renders),
            "alpha_max_abs": _max_abs(supa_alphas, cpu_alphas),
            "grad_max_abs": _grad_errors(
                cpu_scene,
                supa_scene,
                ("means", "quats", "scales", "opacities", "colors"),
            ),
        },
    }


def _case_fallback_sh(device: torch.device) -> dict[str, Any]:
    cpu_scene = fallback_tests._make_basic_scene("cpu")
    supa_scene = fallback_tests._make_basic_scene(device)
    cpu_colors = fallback_tests._make_sh_colors(cpu_scene)
    supa_colors = fallback_tests._make_sh_colors(supa_scene)

    cpu_renders, cpu_alphas, _ = fallback_tests._call_reference_fallback(
        cpu_scene,
        colors=cpu_colors,
        sh_degree=1,
        render_mode="RGB",
        chunk_size=2,
    )
    supa_renders, supa_alphas, meta = fallback_tests._call_public_rasterization(
        supa_scene,
        colors=supa_colors,
        sh_degree=1,
        render_mode="RGB",
        channel_chunk=2,
    )
    if meta["backend"] != "torch_fallback":
        raise AssertionError(f"expected torch_fallback backend, got {meta['backend']}")

    (cpu_renders.sum() + cpu_alphas.sum()).backward()
    (supa_renders.sum() + supa_alphas.sum()).backward()
    grad_errors = _grad_errors(
        cpu_scene,
        supa_scene,
        ("means", "quats", "scales", "opacities"),
    )
    grad_errors["sh_colors"] = _max_abs(supa_colors.grad, cpu_colors.grad)
    return {
        "name": "supa_fallback_sh_rgb",
        "status": "passed",
        "meta": {
            "backend": meta.get("backend"),
            "projection_backend": meta.get("projection_backend"),
            "composite_backend": meta.get("composite_backend"),
            "composite_variant": meta.get("composite_variant"),
        },
        "errors": {
            "render_max_abs": _max_abs(supa_renders, cpu_renders),
            "alpha_max_abs": _max_abs(supa_alphas, cpu_alphas),
            "grad_max_abs": grad_errors,
        },
    }


def _case_native_two_camera(device: torch.device) -> dict[str, Any]:
    cpu_scene = fallback_tests._make_two_camera_scene("cpu")
    supa_scene = fallback_tests._make_two_camera_scene(device)
    cpu_extra = fallback_tests._make_extra_signals(cpu_scene)
    supa_extra = fallback_tests._make_extra_signals(supa_scene)
    backgrounds = torch.tensor(
        [[0.05, 0.10, 0.15], [0.15, 0.08, 0.03]], dtype=torch.float32
    )

    cpu_renders, cpu_alphas, cpu_meta = fallback_tests._call_reference_fallback(
        cpu_scene,
        render_mode="RGB+ED",
        extra_signals=cpu_extra,
        backgrounds=backgrounds,
        chunk_size=2,
    )
    supa_renders, supa_alphas, meta = fallback_tests._call_public_rasterization(
        supa_scene,
        render_mode="RGB+ED",
        extra_signals=supa_extra,
        backgrounds=backgrounds.to(device),
        channel_chunk=2,
        supa_native_options=_NATIVE_CORE_OPTIONS,
    )
    if meta.get("composite_backend") != "supa_native":
        return {
            "name": "supa_native_two_camera_extra_expected_depth",
            "status": "skipped",
            "reason": "SUPA native composite is not available",
            "meta": {
                "projection_backend": meta.get("projection_backend"),
                "composite_backend": meta.get("composite_backend"),
                "composite_variant": meta.get("composite_variant"),
            },
        }

    cpu_loss = (
        cpu_renders.sum()
        + cpu_alphas.sum()
        + cpu_meta["render_extra_signals"].sum()
    )
    supa_loss = (
        supa_renders.sum()
        + supa_alphas.sum()
        + meta["render_extra_signals"].sum()
    )
    cpu_loss.backward()
    supa_loss.backward()
    grad_errors = _grad_errors(
        cpu_scene,
        supa_scene,
        ("means", "quats", "scales", "opacities", "colors"),
    )
    grad_errors["extra_signals"] = _max_abs(supa_extra.grad, cpu_extra.grad)
    return {
        "name": "supa_native_two_camera_extra_expected_depth",
        "status": "passed",
        "meta": {
            "projection_backend": meta.get("projection_backend"),
            "composite_backend": meta.get("composite_backend"),
            "composite_variant": meta.get("composite_variant"),
            "tile_count_backend": meta.get("tile_count_backend"),
            "tiled_metadata_backend": meta.get("tiled_metadata_backend"),
        },
        "errors": {
            "render_max_abs": _max_abs(supa_renders, cpu_renders),
            "alpha_max_abs": _max_abs(supa_alphas, cpu_alphas),
            "extra_max_abs": _max_abs(
                meta["render_extra_signals"], cpu_meta["render_extra_signals"]
            ),
            "grad_max_abs": grad_errors,
        },
    }


def _case_native_tiled_train(device: torch.device) -> dict[str, Any]:
    cpu_scene = fallback_tests._make_two_camera_scene("cpu")
    supa_scene = fallback_tests._make_two_camera_scene(device)
    cpu_extra = fallback_tests._make_extra_signals(cpu_scene)
    supa_extra = fallback_tests._make_extra_signals(supa_scene)
    backgrounds = torch.tensor(
        [[0.05, 0.10, 0.15], [0.15, 0.08, 0.03]], dtype=torch.float32
    )

    cpu_renders, cpu_alphas, cpu_meta = fallback_tests._call_reference_fallback(
        cpu_scene,
        render_mode="RGB+ED",
        extra_signals=cpu_extra,
        backgrounds=backgrounds,
        chunk_size=2,
    )
    supa_renders, supa_alphas, meta = fallback_tests._call_public_rasterization(
        supa_scene,
        render_mode="RGB+ED",
        extra_signals=supa_extra,
        backgrounds=backgrounds.to(device),
        channel_chunk=2,
        supa_native_options=_NATIVE_TILED_TRAIN_OPTIONS,
    )
    result = {
        "name": "supa_native_tiled_train_two_camera",
        "status": "skipped",
        "reason": "SUPA native tiled training composite is not available",
        "meta": {
            "projection_backend": meta.get("projection_backend"),
            "composite_backend": meta.get("composite_backend"),
            "composite_variant": meta.get("composite_variant"),
        },
    }
    if meta.get("composite_backend") == "supa_native":
        cpu_loss = (
            cpu_renders.sum()
            + cpu_alphas.sum()
            + cpu_meta["render_extra_signals"].sum()
        )
        supa_loss = (
            supa_renders.sum()
            + supa_alphas.sum()
            + meta["render_extra_signals"].sum()
        )
        cpu_loss.backward()
        supa_loss.backward()
        grad_errors = _grad_errors(
            cpu_scene,
            supa_scene,
            ("means", "quats", "scales", "opacities", "colors"),
        )
        grad_errors["extra_signals"] = _max_abs(supa_extra.grad, cpu_extra.grad)
        result = {
            "name": "supa_native_tiled_train_two_camera",
            "status": "passed",
            "meta": {
                "projection_backend": meta.get("projection_backend"),
                "composite_backend": meta.get("composite_backend"),
                "composite_variant": meta.get("composite_variant"),
                "tile_count_backend": meta.get("tile_count_backend"),
                "tiled_metadata_backend": meta.get("tiled_metadata_backend"),
            },
            "errors": {
                "render_max_abs": _max_abs(supa_renders, cpu_renders),
                "alpha_max_abs": _max_abs(supa_alphas, cpu_alphas),
                "extra_max_abs": _max_abs(
                    meta["render_extra_signals"], cpu_meta["render_extra_signals"]
                ),
                "grad_max_abs": grad_errors,
            },
        }
    result["name"] = "supa_native_tiled_train_two_camera"
    if result["status"] == "passed" and result["meta"].get("composite_variant") != "tiled":
        result = {
            "name": "supa_native_tiled_train_two_camera",
            "status": "skipped",
            "reason": "SUPA native tiled training composite is not available",
            "meta": result["meta"],
        }
    return result


def _case_garden_native_subset(device: torch.device) -> dict[str, Any]:
    from gsplat._torch_fallback import _tiles_per_gauss

    cpu_scene = fallback_tests._make_garden_subset_scene("cpu")
    supa_scene = fallback_tests._make_garden_subset_scene(device)
    backgrounds = torch.tensor(
        [[0.03, 0.04, 0.05], [0.07, 0.02, 0.01]], dtype=torch.float32
    )

    cpu_renders, cpu_alphas, _ = fallback_tests._call_reference_fallback(
        cpu_scene,
        backgrounds=backgrounds,
        render_mode="RGB+D",
        chunk_size=8,
        width=cpu_scene["width"],
        height=cpu_scene["height"],
    )
    supa_renders, supa_alphas, meta = fallback_tests._call_public_rasterization(
        supa_scene,
        backgrounds=backgrounds.to(device),
        render_mode="RGB+D",
        channel_chunk=8,
        width=supa_scene["width"],
        height=supa_scene["height"],
        supa_native_options=_NATIVE_CORE_OPTIONS,
    )
    if meta.get("composite_backend") != "supa_native":
        return {
            "name": "supa_native_garden_asset_subset",
            "status": "skipped",
            "reason": "SUPA native composite is not available",
            "meta": {
                "projection_backend": meta.get("projection_backend"),
                "composite_backend": meta.get("composite_backend"),
                "composite_variant": meta.get("composite_variant"),
            },
        }

    ref_tiles = _tiles_per_gauss(
        meta["means2d"],
        meta["radii"],
        meta["tile_size"],
        meta["tile_width"],
        meta["tile_height"],
    )
    cpu_loss = cpu_renders.square().mean() + 0.3 * cpu_alphas.square().mean()
    supa_loss = supa_renders.square().mean() + 0.3 * supa_alphas.square().mean()
    cpu_loss.backward()
    supa_loss.backward()
    return {
        "name": "supa_native_garden_asset_subset",
        "status": "passed",
        "meta": {
            "projection_backend": meta.get("projection_backend"),
            "composite_backend": meta.get("composite_backend"),
            "composite_variant": meta.get("composite_variant"),
            "tile_count_backend": meta.get("tile_count_backend"),
            "tiled_metadata_backend": meta.get("tiled_metadata_backend"),
        },
        "errors": {
            "render_max_abs": _max_abs(supa_renders, cpu_renders),
            "alpha_max_abs": _max_abs(supa_alphas, cpu_alphas),
            "tiles_per_gauss_max_abs": _max_abs(meta["tiles_per_gauss"], ref_tiles),
            "grad_max_abs": _grad_errors(
                cpu_scene,
                supa_scene,
                ("means", "quats", "scales", "opacities", "colors"),
            ),
        },
    }


CASES: tuple[tuple[str, Callable[[torch.device], dict[str, Any]]], ...] = (
    ("fallback_basic", _case_fallback_basic),
    ("fallback_sh", _case_fallback_sh),
    ("native_two_camera", _case_native_two_camera),
    ("native_tiled_train", _case_native_tiled_train),
    ("garden_native_subset", _case_garden_native_subset),
)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path; defaults to .agents/output/correctness/render_gate_<rev>.json",
    )
    parser.add_argument(
        "--case",
        choices=[name for name, _ in CASES],
        action="append",
        default=None,
        help="Run only selected case(s). May be passed multiple times.",
    )
    args = parser.parse_args()

    rev = _git_revision()
    output = (
        args.output
        or REPO_ROOT / ".agents" / "output" / "correctness" / f"render_gate_{rev}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = set(args.case) if args.case else None
    device = _require_supa_device()
    started = time.time()
    results = []
    for name, func in CASES:
        if selected is not None and name not in selected:
            continue
        case_started = time.time()
        try:
            result = func(device)
            result["elapsed_s"] = time.time() - case_started
        except pytest.skip.Exception as exc:
            result = {
                "name": name,
                "status": "skipped",
                "reason": str(exc),
                "elapsed_s": time.time() - case_started,
            }
        except Exception as exc:
            result = {
                "name": name,
                "status": "failed",
                "error": repr(exc),
                "elapsed_s": time.time() - case_started,
            }
            results.append(result)
            report = {
                "revision": rev,
                "status": "failed",
                "elapsed_s": time.time() - started,
                "cases": results,
            }
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            raise
        results.append(result)
        log_info(LOGGER, f"{result['name']}: {result['status']}")

    status = "passed"
    if any(item["status"] == "failed" for item in results):
        status = "failed"
    report = {
        "revision": rev,
        "status": status,
        "elapsed_s": time.time() - started,
        "cases": results,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_info(LOGGER, f"wrote {output}")


if __name__ == "__main__":
    main()
