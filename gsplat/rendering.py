# SPDX-License-Identifier: Apache-2.0

"""SUPA-focused public rasterization API."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from ._device import torch_fallback_enabled
from ._types import (
    CameraModel,
    ExternalDistortionModelParameters,
    FThetaCameraDistortionParameters,
    RollingShutterType,
    RowOffsetStructuredSpinningLidarModelParametersExt,
    UnscentedTransformParameters,
)
from .distributed import all_gather_int32, all_gather_tensor_list


# Gaussian depth modes (D/ED) use projection depth. Hit-distance modes are
# intentionally outside the frozen SUPA surface because they require eval3d.
RenderMode = Literal["RGB", "d", "Ed", "D", "ED", "RGB-d", "RGB-Ed", "RGB+D", "RGB+ED"]
RasterizeMode = Literal["classic", "antialiased"]


def render_mode_has_color(mode: RenderMode) -> bool:
    return mode in {"RGB", "RGB-d", "RGB-Ed", "RGB+D", "RGB+ED"}


def render_mode_has_hit_distance(mode: RenderMode) -> bool:
    return mode in {"d", "Ed", "RGB-d", "RGB-Ed"}


def render_mode_has_depth(mode: RenderMode) -> bool:
    return mode in {"D", "ED", "RGB+D", "RGB+ED"}


def render_mode_has_expected_depth(mode: RenderMode) -> bool:
    return mode in {"Ed", "ED", "RGB-Ed", "RGB+ED"}


def render_mode_has_depth_channel(mode: RenderMode) -> bool:
    return render_mode_has_depth(mode) or render_mode_has_hit_distance(mode)


def render_mode_has_only_depth_channel(mode: RenderMode) -> bool:
    return render_mode_has_depth_channel(mode) and not render_mode_has_color(mode)


def render_mode_has_only_color(mode: RenderMode) -> bool:
    return not render_mode_has_depth_channel(mode) and render_mode_has_color(mode)


def _gather_supa_distributed_tensor(tensor: Tensor, sizes: list[int]) -> Tensor:
    """Gather first-dimension SUPA shards with padding."""
    world_size = len(sizes)
    max_size = max(sizes)
    if max_size == 0:
        return tensor.new_empty((0, *tensor.shape[1:]))

    if tensor.shape[0] < max_size:
        padding = tensor.new_zeros((max_size - tensor.shape[0], *tensor.shape[1:]))
        tensor = torch.cat((tensor, padding), dim=0)
    elif tensor.shape[0] > max_size:
        raise ValueError(
            f"local shard has {tensor.shape[0]} rows but max distributed size is {max_size}"
        )

    (gathered,) = all_gather_tensor_list(world_size, [tensor.contiguous()])
    chunks = gathered.split(max_size, dim=0)
    return torch.cat(
        [chunk[: int(size)] for chunk, size in zip(chunks, sizes)], dim=0
    )


def _validate_basic_surface(
    *,
    means: Tensor,
    colors: Optional[Tensor],
    viewmats: Tensor,
    width: int,
    height: int,
    render_mode: RenderMode,
    packed: bool,
    sparse_grad: bool,
    absgrad: bool,
    segmented: bool,
    camera_model: CameraModel,
    with_ut: bool,
    with_eval3d: bool,
    return_normals: bool,
    global_z_order: bool,
    rays: Optional[Tensor],
    radial_coeffs: Optional[Tensor],
    tangential_coeffs: Optional[Tensor],
    thin_prism_coeffs: Optional[Tensor],
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters],
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt],
    external_distortion_coeffs: Optional[ExternalDistortionModelParameters],
    rolling_shutter: RollingShutterType,
    viewmats_rs: Optional[Tensor],
    distributed: bool,
) -> None:
    unsupported: list[str] = []
    if packed:
        unsupported.append("packed=True")
    if sparse_grad:
        unsupported.append("sparse_grad=True")
    if absgrad:
        unsupported.append("absgrad=True")
    if segmented:
        unsupported.append("segmented=True")
    if camera_model not in ("pinhole", "ortho", "fisheye"):
        unsupported.append(f"camera_model={camera_model!r}")
    if with_ut:
        unsupported.append("with_ut=True")
    if with_eval3d:
        unsupported.append("with_eval3d=True")
    if return_normals:
        unsupported.append("return_normals=True")
    if not global_z_order:
        unsupported.append("global_z_order=False")
    if rays is not None:
        unsupported.append("rays")
    if any(
        item is not None
        for item in (radial_coeffs, tangential_coeffs, thin_prism_coeffs, ftheta_coeffs)
    ):
        unsupported.append("camera distortion")
    if lidar_coeffs is not None:
        unsupported.append("lidar_coeffs")
    if external_distortion_coeffs is not None:
        unsupported.append("external_distortion_coeffs")
    if rolling_shutter != RollingShutterType.GLOBAL:
        unsupported.append("rolling_shutter")
    if viewmats_rs is not None:
        unsupported.append("viewmats_rs")
    if render_mode_has_hit_distance(render_mode):
        unsupported.append(f"render_mode={render_mode!r}")
    if distributed and means.device.type != "supa":
        unsupported.append("distributed=True on non-SUPA tensors")
    if distributed and means.shape[:-2] != ():
        unsupported.append("distributed=True with batch dimensions")
    if unsupported:
        raise NotImplementedError(
            "gsplat-biren's frozen public surface supports basic unpacked "
            f"3DGS rasterization only. Unsupported options: {', '.join(unsupported)}."
        )

    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}.")
    if colors is None and render_mode_has_color(render_mode):
        raise ValueError(
            f"colors must be provided when render_mode={render_mode!r} includes RGB."
        )
    if colors is not None and render_mode_has_only_depth_channel(render_mode):
        # Depth-only modes ignore colors in the fallback. Keep this allowed for
        # compatibility with existing call sites that pass colors unconditionally.
        return
    if not (render_mode_has_color(render_mode) or render_mode_has_depth(render_mode)):
        raise ValueError(f"unsupported render_mode={render_mode!r}.")


def _expect_shape(name: str, tensor: Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}.")


def _validate_feature_shape(
    name: str,
    features: Tensor,
    *,
    batch_dims: tuple[int, ...],
    n_cameras: int,
    n_gaussians: int,
    sh_degree: Optional[int],
    require_rgb_sh: bool,
) -> None:
    valid_prefixes = (
        batch_dims + (n_gaussians,),
        batch_dims + (n_cameras, n_gaussians),
    )
    if sh_degree is None:
        if features.ndim not in (len(batch_dims) + 2, len(batch_dims) + 3):
            raise ValueError(
                f"{name} must have shape [..., N, D] or [..., C, N, D], "
                f"got {tuple(features.shape)}."
            )
        if features.shape[-1] <= 0:
            raise ValueError(f"{name} must have at least one channel.")
        if tuple(features.shape[:-1]) not in valid_prefixes:
            raise ValueError(
                f"{name} must align with batch/camera/Gaussian dimensions, "
                f"got {tuple(features.shape)} for batch={batch_dims}, "
                f"C={n_cameras}, N={n_gaussians}."
            )
        return

    if int(sh_degree) < 0:
        raise ValueError(f"{name}_sh_degree must be non-negative, got {sh_degree}.")
    if features.ndim not in (len(batch_dims) + 3, len(batch_dims) + 4):
        raise ValueError(
            f"{name} must have SH shape [..., N, K, D] or [..., C, N, K, D], "
            f"got {tuple(features.shape)}."
        )
    if features.shape[-1] <= 0:
        raise ValueError(f"{name} must have at least one channel.")
    if tuple(features.shape[:-2]) not in valid_prefixes:
        raise ValueError(
            f"{name} must align with batch/camera/Gaussian dimensions, "
            f"got {tuple(features.shape)} for batch={batch_dims}, "
            f"C={n_cameras}, N={n_gaussians}."
        )
    required_coeffs = (int(sh_degree) + 1) ** 2
    if features.shape[-2] < required_coeffs:
        raise ValueError(
            f"{name} must include at least {required_coeffs} SH coefficients "
            f"for degree {sh_degree}, got {tuple(features.shape)}."
        )
    if require_rgb_sh and features.shape[-1] != 3:
        raise ValueError(
            f"{name} SH coefficients must have 3 color channels, got {tuple(features.shape)}."
        )


def _validate_tensor_contract(
    *,
    means: Tensor,
    quats: Optional[Tensor],
    scales: Optional[Tensor],
    opacities: Tensor,
    colors: Optional[Tensor],
    viewmats: Tensor,
    Ks: Tensor,
    covars: Optional[Tensor],
    backgrounds: Optional[Tensor],
    extra_signals: Optional[Tensor],
    sh_degree: Optional[int],
    extra_signals_sh_degree: Optional[int],
    rasterize_mode: RasterizeMode,
    tile_size: int,
    channel_chunk: int,
) -> tuple[tuple[int, ...], int, int]:
    if means.ndim < 2 or means.shape[-1] != 3:
        raise ValueError(f"means must have shape [..., N, 3], got {tuple(means.shape)}.")
    if viewmats.ndim < 3 or viewmats.shape[-2:] != (4, 4):
        raise ValueError(
            f"viewmats must have shape [..., C, 4, 4], got {tuple(viewmats.shape)}."
        )
    if Ks.ndim < 3 or Ks.shape[-2:] != (3, 3):
        raise ValueError(f"Ks must have shape [..., C, 3, 3], got {tuple(Ks.shape)}.")

    batch_dims = tuple(means.shape[:-2])
    n_gaussians = int(means.shape[-2])
    n_cameras = int(viewmats.shape[-3])
    _expect_shape("viewmats", viewmats, batch_dims + (n_cameras, 4, 4))
    _expect_shape("Ks", Ks, batch_dims + (n_cameras, 3, 3))
    _expect_shape("opacities", opacities, batch_dims + (n_gaussians,))

    if covars is None:
        if quats is None or scales is None:
            raise ValueError("quats and scales are required when covars is not provided.")
        _expect_shape("quats", quats, batch_dims + (n_gaussians, 4))
        _expect_shape("scales", scales, batch_dims + (n_gaussians, 3))
    elif tuple(covars.shape) not in (
        batch_dims + (n_gaussians, 3, 3),
        batch_dims + (n_gaussians, 6),
    ):
        raise ValueError(
            "covars must have shape "
            f"{batch_dims + (n_gaussians, 3, 3)} or {batch_dims + (n_gaussians, 6)}, "
            f"got {tuple(covars.shape)}."
        )

    if colors is not None:
        _validate_feature_shape(
            "colors",
            colors,
            batch_dims=batch_dims,
            n_cameras=n_cameras,
            n_gaussians=n_gaussians,
            sh_degree=sh_degree,
            require_rgb_sh=True,
        )

    if backgrounds is not None:
        expected_prefix = batch_dims + (n_cameras,)
        if backgrounds.ndim != len(expected_prefix) + 1:
            raise ValueError(
                f"backgrounds must have shape {expected_prefix + ('D',)}, "
                f"got {tuple(backgrounds.shape)}."
            )
        if tuple(backgrounds.shape[:-1]) != expected_prefix:
            raise ValueError(
                "backgrounds must have shape "
                f"{expected_prefix + (backgrounds.shape[-1],)}, "
                f"got {tuple(backgrounds.shape)}."
            )
        if backgrounds.shape[-1] <= 0:
            raise ValueError("backgrounds must have at least one channel.")
    if extra_signals is not None:
        _validate_feature_shape(
            "extra_signals",
            extra_signals,
            batch_dims=batch_dims,
            n_cameras=n_cameras,
            n_gaussians=n_gaussians,
            sh_degree=extra_signals_sh_degree,
            require_rgb_sh=False,
        )

    if rasterize_mode not in ("classic", "antialiased"):
        raise ValueError(
            f"rasterize_mode must be 'classic' or 'antialiased', got {rasterize_mode!r}."
        )
    if int(tile_size) <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}.")
    if int(channel_chunk) <= 0:
        raise ValueError(f"channel_chunk must be positive, got {channel_chunk}.")
    return batch_dims, n_gaussians, n_cameras


def rasterization(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Optional[
        Tensor
    ],  # [..., (C,) N, D] or [..., (C,) N, K, 3]; None for depth-only modes
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = False,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: RenderMode = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: RasterizeMode = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: CameraModel = "pinhole",
    segmented: bool = False,
    covars: Optional[Tensor] = None,
    with_ut: bool = False,
    with_eval3d: bool = False,
    return_normals: bool = False,
    global_z_order: bool = True,
    rays: Optional[Tensor] = None,
    radial_coeffs: Optional[Tensor] = None,
    tangential_coeffs: Optional[Tensor] = None,
    thin_prism_coeffs: Optional[Tensor] = None,
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    external_distortion_coeffs: Optional[ExternalDistortionModelParameters] = None,
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,
    ut_params: Optional[UnscentedTransformParameters] = None,
    extra_signals: Optional[Tensor] = None,
    extra_signals_sh_degree: Optional[int] = None,
    supa_native_options: Optional[Dict[str, object]] = None,
    supa_use_torch_fallback: Optional[bool] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize basic 3D Gaussians on the SUPA-focused gsplat-biren surface.

    The maintained public path is unpacked basic 3DGS rasterization through the
    differentiable PyTorch fallback, with optional native SUPA acceleration
    selected by ``supa_native_options``. ``packed`` defaults to ``False`` in this
    fork because the SUPA fallback does not implement packed intermediates.
    """
    _ = ut_params  # Signature compatibility; with_ut is rejected above.
    _validate_basic_surface(
        means=means,
        colors=colors,
        viewmats=viewmats,
        width=width,
        height=height,
        render_mode=render_mode,
        packed=packed,
        sparse_grad=sparse_grad,
        absgrad=absgrad,
        segmented=segmented,
        camera_model=camera_model,
        with_ut=with_ut,
        with_eval3d=with_eval3d,
        return_normals=return_normals,
        global_z_order=global_z_order,
        rays=rays,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
        ftheta_coeffs=ftheta_coeffs,
        lidar_coeffs=lidar_coeffs,
        external_distortion_coeffs=external_distortion_coeffs,
        rolling_shutter=rolling_shutter,
        viewmats_rs=viewmats_rs,
        distributed=distributed,
    )

    batch_dims, N, C = _validate_tensor_contract(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        covars=covars,
        backgrounds=backgrounds,
        extra_signals=extra_signals,
        sh_degree=sh_degree,
        extra_signals_sh_degree=extra_signals_sh_degree,
        rasterize_mode=rasterize_mode,
        tile_size=tile_size,
        channel_chunk=channel_chunk,
    )
    if extra_signals is not None and distributed:
        raise NotImplementedError("distributed=True with extra_signals is not supported.")

    use_torch_fallback = (
        torch_fallback_enabled(means.device)
        if supa_use_torch_fallback is None
        else bool(supa_use_torch_fallback)
    )
    if not use_torch_fallback:
        raise NotImplementedError(
            "The upstream rasterization path has been removed from gsplat-biren. "
            "Use the SUPA PyTorch fallback with optional native SUPA acceleration."
        )

    from ._torch_fallback import rasterization as torch_rasterization

    if distributed:
        world_size = torch.distributed.get_world_size()
        N_world = all_gather_int32(world_size, N, device=means.device)
        means = _gather_supa_distributed_tensor(means, N_world)
        opacities = _gather_supa_distributed_tensor(opacities, N_world)
        if covars is None:
            quats = _gather_supa_distributed_tensor(quats, N_world)
            scales = _gather_supa_distributed_tensor(scales, N_world)
        else:
            covars = _gather_supa_distributed_tensor(covars, N_world)
        if colors is not None:
            colors = _gather_supa_distributed_tensor(colors, N_world)

    render_colors, render_alphas, meta = torch_rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        eps2d=eps2d,
        sh_degree=sh_degree,
        tile_size=tile_size,
        backgrounds=backgrounds,
        render_mode=render_mode,
        rasterize_mode=rasterize_mode,
        camera_model=camera_model,
        covars=covars,
        extra_signals=extra_signals,
        extra_signals_sh_degree=extra_signals_sh_degree,
        chunk_size=max(1, int(channel_chunk)),
        supa_native_options=supa_native_options,
    )
    if distributed:
        meta.update(
            {
                "distributed": True,
                "distributed_backend": "supa_gather",
                "N_world": N_world,
            }
        )
    return render_colors, render_alphas, meta


def _unsupported_removed_surface(name: str):
    def _raise(*args, **kwargs):
        raise NotImplementedError(
            f"{name} has been removed from gsplat-biren's SUPA-only public surface."
        )

    _raise.__name__ = name
    return _raise


rasterization_2dgs = _unsupported_removed_surface("rasterization_2dgs")
rasterization_inria_wrapper = _unsupported_removed_surface("rasterization_inria_wrapper")
rasterization_2dgs_inria_wrapper = _unsupported_removed_surface(
    "rasterization_2dgs_inria_wrapper"
)
