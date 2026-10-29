# SPDX-License-Identifier: Apache-2.0

"""Public types for the SUPA-focused gsplat-biren API."""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Any

from typing_extensions import Literal


CameraModel = Literal["pinhole", "ortho", "fisheye"]
ExternalDistortionModelMeta = Literal["bivariate-windshield"]


class RollingShutterType(IntEnum):
    """Kept for signature compatibility; only GLOBAL is supported."""

    ROLLING_TOP_TO_BOTTOM = 0
    ROLLING_LEFT_TO_RIGHT = 1
    ROLLING_BOTTOM_TO_TOP = 2
    ROLLING_RIGHT_TO_LEFT = 3
    GLOBAL = 4


class FThetaPolynomialType(IntEnum):
    """Compatibility enum for the removed f-theta camera parameter surface."""

    PIXELDIST_TO_ANGLE = 0
    ANGLE_TO_PIXELDIST = 1


class ExternalDistortionReferencePolynomial(IntEnum):
    """Compatibility enum for the removed external distortion surface."""

    FORWARD = 1
    BACKWARD = 2


class SpinningDirection(Enum):
    """Compatibility enum for the removed lidar parameter surface."""

    CLOCKWISE = 0
    COUNTER_CLOCKWISE = 1


class ExternalDistortionModelParameters:
    """Marker base class for removed external distortion parameter objects."""


class UnsupportedSurfaceParameter:
    """Placeholder for frozen upstream-only parameter objects."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "This gsplat-biren build is SUPA-only and does not expose the "
            "upstream extended camera/distortion parameter classes."
        )


UnscentedTransformParameters = UnsupportedSurfaceParameter
FThetaCameraDistortionParameters = UnsupportedSurfaceParameter
RowOffsetStructuredSpinningLidarModelParameters = UnsupportedSurfaceParameter
RowOffsetStructuredSpinningLidarModelParametersExt = UnsupportedSurfaceParameter


class BivariateWindshieldModelParameters(
    UnsupportedSurfaceParameter,
    ExternalDistortionModelParameters,
):
    """Placeholder for the removed bivariate windshield parameter object."""

    MAX_ORDER: int = 5
    MAX_COEFFS: int = 21


class LidarTiling(UnsupportedSurfaceParameter):
    """Placeholder for the removed lidar tiling helper."""
