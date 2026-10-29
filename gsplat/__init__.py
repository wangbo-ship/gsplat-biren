# SPDX-License-Identifier: Apache-2.0

"""SUPA-only public surface for gsplat-biren."""

from __future__ import annotations

from ._math import _quat_scale_to_covar_preci as quat_scale_to_covar_preci
from ._torch_impl import (
    _fully_fused_projection as fully_fused_projection,
    _isect_offset_encode as isect_offset_encode,
    _isect_tiles as isect_tiles,
    _spherical_harmonics as spherical_harmonics,
    _world_to_cam as world_to_cam,
)
from ._types import (
    BivariateWindshieldModelParameters,
    CameraModel,
    ExternalDistortionModelMeta,
    ExternalDistortionModelParameters,
    ExternalDistortionReferencePolynomial,
    FThetaCameraDistortionParameters,
    FThetaPolynomialType,
    LidarTiling,
    RollingShutterType,
    RowOffsetStructuredSpinningLidarModelParameters,
    RowOffsetStructuredSpinningLidarModelParametersExt,
    SpinningDirection,
    UnscentedTransformParameters,
)
from .color_correct import color_correct_affine, color_correct_quadratic
from .compression import PngCompression
from .exporter import export_splats
from .rendering import (
    RasterizeMode,
    RenderMode,
    rasterization,
    rasterization_2dgs,
    rasterization_2dgs_inria_wrapper,
    rasterization_inria_wrapper,
)
from .strategy import DefaultStrategy, MCMCStrategy, Strategy
from .version import __version__


def has_2dgs() -> bool:
    return False


def has_3dgs() -> bool:
    return True


def has_3dgut() -> bool:
    return False


def has_adam() -> bool:
    return False


def has_camera_wrappers() -> bool:
    return False


def has_reloc() -> bool:
    return False


__all__ = [
    "CameraModel",
    "DefaultStrategy",
    "BivariateWindshieldModelParameters",
    "ExternalDistortionModelMeta",
    "ExternalDistortionModelParameters",
    "ExternalDistortionReferencePolynomial",
    "FThetaCameraDistortionParameters",
    "FThetaPolynomialType",
    "LidarTiling",
    "MCMCStrategy",
    "PngCompression",
    "RasterizeMode",
    "RenderMode",
    "RollingShutterType",
    "RowOffsetStructuredSpinningLidarModelParameters",
    "RowOffsetStructuredSpinningLidarModelParametersExt",
    "SpinningDirection",
    "Strategy",
    "UnscentedTransformParameters",
    "__version__",
    "color_correct_affine",
    "color_correct_quadratic",
    "export_splats",
    "fully_fused_projection",
    "has_2dgs",
    "has_3dgs",
    "has_3dgut",
    "has_adam",
    "has_camera_wrappers",
    "has_reloc",
    "isect_offset_encode",
    "isect_tiles",
    "quat_scale_to_covar_preci",
    "rasterization",
    "rasterization_2dgs",
    "rasterization_2dgs_inria_wrapper",
    "rasterization_inria_wrapper",
    "spherical_harmonics",
    "world_to_cam",
]
