# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autograd wrappers for optional native SUPA kernels."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from gsplat._device import is_supa_device
from gsplat._device import synchronize

from ._backend import load_supa_backend


_TILED_BWD_AUTO_TILE_REDUCE_MAX_RANGE = 768


def _materialize_grad(grad: Tensor, *, force_copy: bool = False) -> Tensor:
    if grad.is_contiguous() and not force_copy:
        return grad
    materialized = torch.empty(grad.shape, dtype=grad.dtype, device=grad.device)
    materialized.copy_(grad)
    return materialized


def _materialize_projection_grad(grad: Tensor) -> Tensor:
    # BRPyTorch custom-autograd gradients can look contiguous in Python while
    # exposing incomplete data to the native pointer path.
    if grad.device.type == "supa":
        return grad.cpu().to(grad.device).contiguous()
    return _materialize_grad(grad)


def _materialize_projection_saved(tensor: Optional[Tensor]) -> Optional[Tensor]:
    # Saved forward tensors need the same fresh SUPA allocation before the
    # projection backward launch; otherwise only part of some tensors is read.
    if tensor is not None and tensor.device.type == "supa":
        return tensor.cpu().to(tensor.device).contiguous()
    return tensor


def _make_composite_sorted_ids(depths: Tensor, radii: Tensor) -> Tensor:
    target_device = depths.device
    # Build non-differentiable ordering metadata on CPU. Index tensors produced
    # by SUPA PyTorch ops can expose incomplete data to native pointer kernels.
    depths = depths.detach().cpu()
    radii = radii.detach().cpu()
    valid = (radii > 0).all(dim=-1) & torch.isfinite(depths)
    masked_depths = torch.where(valid, depths, torch.full_like(depths, float("inf")))
    sorted_ids = torch.argsort(masked_depths, dim=-1)
    sorted_valid = torch.gather(valid, dim=-1, index=sorted_ids)
    sorted_ids = torch.where(sorted_valid, sorted_ids, torch.full_like(sorted_ids, -1))
    return sorted_ids.to(torch.int32).to(target_device).contiguous()


class _QuatScaleToCovarPreci(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        quats: Tensor,
        scales: Tensor,
        compute_covar: bool,
        compute_preci: bool,
        triu: bool,
    ) -> Tuple[Tensor, Tensor]:
        backend = load_supa_backend()
        if backend is None:
            raise RuntimeError("SUPA native backend is unavailable")
        covars, precis = backend.quat_scale_to_covar_preci_fwd(
            quats, scales, compute_covar, compute_preci, triu
        )
        ctx.save_for_backward(quats, scales)
        ctx.compute_covar = compute_covar
        ctx.compute_preci = compute_preci
        ctx.triu = triu
        return covars, precis

    @staticmethod
    def backward(ctx, v_covars: Tensor, v_precis: Tensor):
        backend = load_supa_backend()
        if backend is None:
            raise RuntimeError("SUPA native backend is unavailable")
        quats, scales = ctx.saved_tensors
        grad_covars = _materialize_grad(v_covars) if ctx.compute_covar else None
        grad_precis = _materialize_grad(v_precis) if ctx.compute_preci else None
        v_quats, v_scales = backend.quat_scale_to_covar_preci_bwd(
            quats,
            scales,
            ctx.triu,
            grad_covars,
            grad_precis,
        )
        synchronize(quats.device)
        return v_quats, v_scales, None, None, None


class _ProjectionEWA3DGSPinhole(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means: Tensor,
        covars: Optional[Tensor],
        quats: Optional[Tensor],
        scales: Optional[Tensor],
        viewmats: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        calc_compensations: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        backend = load_supa_backend()
        if backend is None:
            raise RuntimeError("SUPA native backend is unavailable")
        radii, means2d, depths, conics, compensations = backend.projection_ewa_3dgs_pinhole_fwd(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            calc_compensations,
        )
        ctx.mark_non_differentiable(radii)
        ctx.has_covars = covars is not None
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.calc_compensations = calc_compensations
        if ctx.has_covars:
            ctx.save_for_backward(means, covars, viewmats, Ks, radii, conics, compensations)
        else:
            ctx.save_for_backward(means, quats, scales, viewmats, Ks, radii, conics, compensations)
        return radii, means2d, depths, conics, compensations

    @staticmethod
    def backward(
        ctx,
        _v_radii: Optional[Tensor],
        v_means2d: Optional[Tensor],
        v_depths: Optional[Tensor],
        v_conics: Optional[Tensor],
        v_compensations: Optional[Tensor],
    ):
        if ctx.has_covars:
            means, covars, viewmats, Ks, radii, conics, compensations = ctx.saved_tensors
            quats = None
            scales = None
        else:
            means, quats, scales, viewmats, Ks, radii, conics, compensations = ctx.saved_tensors
            covars = None

        need_v_means = bool(ctx.needs_input_grad[0])
        need_v_covars = bool(ctx.has_covars and ctx.needs_input_grad[1])
        need_v_quats = bool((not ctx.has_covars) and ctx.needs_input_grad[2])
        need_v_scales = bool((not ctx.has_covars) and ctx.needs_input_grad[3])
        need_v_viewmats = bool(ctx.needs_input_grad[4])

        means = _materialize_projection_saved(means)
        covars = _materialize_projection_saved(covars)
        quats = _materialize_projection_saved(quats)
        scales = _materialize_projection_saved(scales)
        viewmats = _materialize_projection_saved(viewmats)
        Ks = _materialize_projection_saved(Ks)
        radii = _materialize_projection_saved(radii)
        conics = _materialize_projection_saved(conics)
        compensations = _materialize_projection_saved(compensations)

        if v_means2d is None:
            v_means2d = torch.zeros(radii.shape, dtype=means.dtype, device=means.device)
        else:
            v_means2d = _materialize_projection_grad(v_means2d)
        if v_depths is None:
            v_depths_arg: Optional[Tensor] = None
        else:
            v_depths_arg = _materialize_projection_grad(v_depths)
        if v_conics is None:
            v_conics = torch.zeros(conics.shape, dtype=means.dtype, device=means.device)
        else:
            v_conics = _materialize_projection_grad(v_conics)
        if ctx.calc_compensations:
            if v_compensations is None:
                v_compensations = torch.zeros(
                    radii.shape[:-1], dtype=means.dtype, device=means.device
                )
            else:
                v_compensations = _materialize_projection_grad(v_compensations)
        else:
            v_compensations = None

        out = projection_ewa_3dgs_pinhole_bwd(
            means,
            covars,
            viewmats,
            Ks,
            ctx.width,
            ctx.height,
            ctx.eps2d,
            radii,
            conics,
            compensations if ctx.calc_compensations else None,
            v_means2d,
            v_depths_arg,
            v_conics,
            v_compensations,
            quats=quats,
            scales=scales,
            compute_v_means=need_v_means,
            compute_v_covars=need_v_covars,
            compute_v_quats=need_v_quats,
            compute_v_scales=need_v_scales,
            compute_v_viewmats=need_v_viewmats,
            synchronize_launch=False,
        )
        if out is None:
            raise RuntimeError("SUPA native projection backward is unavailable")
        v_means, v_covars, v_quats, v_scales, v_viewmats = out
        if ctx.has_covars:
            v_quats = None
            v_scales = None
        else:
            v_covars = None
        if not need_v_means:
            v_means = None
        if not need_v_covars:
            v_covars = None
        if not need_v_quats:
            v_quats = None
        if not need_v_scales:
            v_scales = None
        if not need_v_viewmats:
            v_viewmats = None
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _SphericalHarmonics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, degree: int, dirs: Tensor, coeffs: Tensor) -> Tensor:
        backend = load_supa_backend()
        if backend is None:
            raise RuntimeError("SUPA native backend is unavailable")
        colors = backend.spherical_harmonics_fwd(int(degree), dirs, coeffs)
        ctx.degree = int(degree)
        ctx.save_for_backward(dirs, coeffs)
        return colors

    @staticmethod
    def backward(ctx, grad_colors: Tensor):
        backend = load_supa_backend()
        if backend is None:
            raise RuntimeError("SUPA native backend is unavailable")
        dirs, coeffs = ctx.saved_tensors
        grad_colors = _materialize_projection_grad(grad_colors)
        grad_dirs, grad_coeffs = backend.spherical_harmonics_bwd(
            int(ctx.degree),
            dirs,
            coeffs,
            grad_colors,
            bool(ctx.needs_input_grad[1]),
        )
        synchronize(dirs.device)
        return None, grad_dirs if ctx.needs_input_grad[1] else None, grad_coeffs


def _composite_backward_leaf(tensor: Tensor, requires_grad: bool) -> Tensor:
    if tensor.device.type == "supa":
        tensor = tensor.detach().cpu().to(tensor.device).contiguous()
    else:
        tensor = tensor.detach().contiguous()
    tensor.requires_grad_(requires_grad)
    return tensor


class _AlphaComposite(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        conics: Tensor,
        features: Tensor,
        opacities: Tensor,
        depths: Tensor,
        radii: Tensor,
        backgrounds: Optional[Tensor],
        width: int,
        height: int,
    ) -> Tuple[Tensor, Tensor]:
        backend = load_supa_backend()
        if backend is None:
            raise RuntimeError("SUPA native backend is unavailable")
        sorted_ids = _make_composite_sorted_ids(depths, radii)
        render, alpha = backend.alpha_composite_fwd(
            means2d,
            conics,
            features,
            opacities,
            depths,
            radii,
            sorted_ids,
            backgrounds,
            width,
            height,
        )
        ctx.width = width
        ctx.height = height
        ctx.has_backgrounds = backgrounds is not None
        if backgrounds is None:
            ctx.save_for_backward(means2d, conics, features, opacities, depths, radii, sorted_ids)
        else:
            ctx.save_for_backward(
                means2d, conics, features, opacities, depths, radii, sorted_ids, backgrounds
            )
        return render, alpha

    @staticmethod
    def backward(ctx, v_render: Optional[Tensor], v_alpha: Optional[Tensor]):
        if ctx.has_backgrounds:
            means2d, conics, features, opacities, depths, radii, sorted_ids, backgrounds = ctx.saved_tensors
        else:
            means2d, conics, features, opacities, depths, radii, sorted_ids = ctx.saved_tensors
            backgrounds = None

        native_grads = alpha_composite_bwd(
            means2d,
            conics,
            features,
            opacities,
            depths,
            radii,
            sorted_ids,
            ctx.width,
            ctx.height,
            backgrounds,
            v_render,
            v_alpha,
        )
        if native_grads is not None:
            (
                grad_means2d,
                grad_conics,
                grad_features,
                grad_opacities,
                grad_depths,
                grad_backgrounds,
            ) = native_grads
            needs = ctx.needs_input_grad
            return (
                grad_means2d if needs[0] else None,
                grad_conics if needs[1] else None,
                grad_features if needs[2] else None,
                grad_opacities if needs[3] else None,
                None,
                None,
                grad_backgrounds if needs[6] and backgrounds is not None else None,
                None,
                None,
            )

        from gsplat._torch_fallback import _alpha_composite

        needs = ctx.needs_input_grad
        with torch.enable_grad():
            grad_inputs = [
                _composite_backward_leaf(means2d, needs[0]),
                _composite_backward_leaf(conics, needs[1]),
                _composite_backward_leaf(features, needs[2]),
                _composite_backward_leaf(opacities, needs[3]),
                _composite_backward_leaf(depths, needs[4]),
            ]
            grad_backgrounds = (
                _composite_backward_leaf(backgrounds, needs[6])
                if backgrounds is not None
                else None
            )
            render_ref, alpha_ref = _alpha_composite(
                grad_inputs[0],
                grad_inputs[1],
                grad_inputs[2],
                grad_inputs[3],
                grad_inputs[4],
                radii,
                ctx.width,
                ctx.height,
                grad_backgrounds,
                chunk_size=32,
            )

        outputs = []
        grad_outputs = []
        if v_render is not None and render_ref.requires_grad:
            outputs.append(render_ref)
            grad_outputs.append(_materialize_grad(v_render))
        if v_alpha is not None and alpha_ref.requires_grad:
            outputs.append(alpha_ref)
            grad_outputs.append(_materialize_grad(v_alpha))

        grad_means2d = grad_conics = grad_features = grad_opacities = grad_depths = None
        grad_backgrounds_out = None
        if outputs:
            differentiable_inputs = []
            input_positions = []
            for position, tensor in enumerate(grad_inputs):
                if tensor.requires_grad:
                    differentiable_inputs.append(tensor)
                    input_positions.append(position)
            if grad_backgrounds is not None and grad_backgrounds.requires_grad:
                differentiable_inputs.append(grad_backgrounds)
                input_positions.append(5)

            if differentiable_inputs:
                grads = torch.autograd.grad(
                    outputs,
                    differentiable_inputs,
                    grad_outputs=grad_outputs,
                    allow_unused=True,
                )
                for position, grad in zip(input_positions, grads):
                    if position == 0:
                        grad_means2d = grad
                    elif position == 1:
                        grad_conics = grad
                    elif position == 2:
                        grad_features = grad
                    elif position == 3:
                        grad_opacities = grad
                    elif position == 4:
                        grad_depths = grad
                    else:
                        grad_backgrounds_out = grad

        return (
            grad_means2d,
            grad_conics,
            grad_features,
            grad_opacities,
            grad_depths,
            None,
            grad_backgrounds_out,
            None,
            None,
        )


class _RasterizeToPixelsTiled(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        conics: Tensor,
        features: Tensor,
        opacities: Tensor,
        backgrounds: Optional[Tensor],
        tile_offsets: Tensor,
        flatten_ids: Tensor,
        width: int,
        height: int,
        tile_size: int,
        backward_mode: str,
    ) -> Tuple[Tensor, Tensor]:
        tiled = rasterize_to_pixels_tiled_fwd(
            means2d,
            conics,
            features,
            opacities,
            width,
            height,
            tile_size,
            tile_offsets,
            flatten_ids,
            backgrounds,
            synchronize_launch=False,
        )
        if tiled is None:
            raise RuntimeError("SUPA native tiled rasterization forward is unavailable")
        render, alpha = tiled
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.backward_mode = _normalize_tiled_backward_mode(backward_mode)
        ctx.has_backgrounds = backgrounds is not None
        if backgrounds is None:
            ctx.save_for_backward(
                means2d, conics, features, opacities, tile_offsets, flatten_ids, alpha
            )
        else:
            ctx.save_for_backward(
                means2d,
                conics,
                features,
                opacities,
                backgrounds,
                tile_offsets,
                flatten_ids,
                alpha,
            )
        return render, alpha

    @staticmethod
    def backward(ctx, v_render: Optional[Tensor], v_alpha: Optional[Tensor]):
        if ctx.has_backgrounds:
            (
                means2d,
                conics,
                features,
                opacities,
                backgrounds,
                tile_offsets,
                flatten_ids,
                forward_alpha,
            ) = ctx.saved_tensors
        else:
            (
                means2d,
                conics,
                features,
                opacities,
                tile_offsets,
                flatten_ids,
                forward_alpha,
            ) = ctx.saved_tensors
            backgrounds = None

        native = rasterize_to_pixels_tiled_bwd(
            means2d,
            conics,
            features,
            opacities,
            ctx.width,
            ctx.height,
            ctx.tile_size,
            tile_offsets,
            flatten_ids,
            backgrounds,
            v_render,
            v_alpha,
            forward_alpha,
            backward_mode=ctx.backward_mode,
        )
        if native is None:
            raise RuntimeError("SUPA native tiled rasterization backward is unavailable")

        grad_means2d, grad_conics, grad_features, grad_opacities, grad_backgrounds = native
        needs = ctx.needs_input_grad
        return (
            grad_means2d if needs[0] else None,
            grad_conics if needs[1] else None,
            grad_features if needs[2] else None,
            grad_opacities if needs[3] else None,
            grad_backgrounds if needs[4] and backgrounds is not None else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def quat_scale_to_covar_preci(
    quats: Tensor,
    scales: Tensor,
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Optional[Tuple[Optional[Tensor], Optional[Tensor]]]:
    """Run the native SUPA quaternion-scale kernel when supported."""
    if (
        not is_supa_device(quats)
        or quats.dtype != torch.float32
        or scales.dtype != torch.float32
        or not quats.is_contiguous()
        or not scales.is_contiguous()
    ):
        return None
    if quats.shape[:-1] != scales.shape[:-1] or quats.shape[-1] != 4 or scales.shape[-1] != 3:
        return None
    backend = load_supa_backend()
    if backend is None:
        return None

    covars, precis = _QuatScaleToCovarPreci.apply(
        quats, scales, compute_covar, compute_preci, triu
    )
    return (covars if compute_covar else None, precis if compute_preci else None)


def projection_ewa_3dgs_pinhole(
    means: Tensor,
    covars: Optional[Tensor],
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    *,
    quats: Optional[Tensor] = None,
    scales: Optional[Tensor] = None,
    synchronize_launch: bool = True,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]]:
    """Run the native SUPA pinhole projection forward prototype when supported."""
    has_covars = covars is not None
    has_quats_scales = quats is not None and scales is not None
    if has_covars == has_quats_scales:
        return None
    if (
        not is_supa_device(means)
        or means.dtype != torch.float32
        or viewmats.dtype != torch.float32
        or Ks.dtype != torch.float32
        or not means.is_contiguous()
        or not viewmats.is_contiguous()
        or not Ks.is_contiguous()
    ):
        return None
    if has_covars and (covars.dtype != torch.float32 or not covars.is_contiguous()):
        return None
    if has_quats_scales and (
        quats.dtype != torch.float32
        or scales.dtype != torch.float32
        or not quats.is_contiguous()
        or not scales.is_contiguous()
    ):
        return None
    if (
        means.dim() < 2
        or means.shape[-1] != 3
        or viewmats.shape[:-3] != means.shape[:-2]
        or Ks.shape[:-3] != means.shape[:-2]
        or viewmats.shape[-2:] != (4, 4)
        or Ks.shape[-2:] != (3, 3)
        or viewmats.shape[-3] != Ks.shape[-3]
    ):
        return None
    if has_covars and covars.shape != means.shape[:-1] + (3, 3):
        return None
    if has_quats_scales and (
        quats.shape != means.shape[:-1] + (4,)
        or scales.shape != means.shape[:-1] + (3,)
    ):
        return None
    backend = load_supa_backend()
    if backend is None:
        return None

    radii, means2d, depths, conics, compensations = _ProjectionEWA3DGSPinhole.apply(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        calc_compensations,
    )
    if synchronize_launch:
        synchronize(means.device)
    return radii, means2d, depths, conics, compensations if calc_compensations else None


def projection_ewa_3dgs_pinhole_bwd(
    means: Tensor,
    covars: Optional[Tensor],
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    eps2d: float,
    radii: Tensor,
    conics: Tensor,
    compensations: Optional[Tensor],
    v_means2d: Tensor,
    v_depths: Optional[Tensor],
    v_conics: Tensor,
    v_compensations: Optional[Tensor] = None,
    *,
    quats: Optional[Tensor] = None,
    scales: Optional[Tensor] = None,
    compute_v_means: bool = True,
    compute_v_covars: bool = True,
    compute_v_quats: bool = True,
    compute_v_scales: bool = True,
    compute_v_viewmats: bool = True,
    synchronize_launch: bool = True,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """Run the native SUPA pinhole projection backward prototype when supported."""
    has_covars = covars is not None
    has_quats_scales = quats is not None and scales is not None
    if has_covars == has_quats_scales:
        return None
    if (
        not is_supa_device(means)
        or means.dtype != torch.float32
        or viewmats.dtype != torch.float32
        or Ks.dtype != torch.float32
        or radii.dtype != torch.int32
        or conics.dtype != torch.float32
        or v_means2d.dtype != torch.float32
        or (v_depths is not None and v_depths.dtype != torch.float32)
        or v_conics.dtype != torch.float32
        or not means.is_contiguous()
        or not viewmats.is_contiguous()
        or not Ks.is_contiguous()
        or not radii.is_contiguous()
        or not conics.is_contiguous()
        or not v_means2d.is_contiguous()
        or (v_depths is not None and not v_depths.is_contiguous())
        or not v_conics.is_contiguous()
    ):
        return None
    if has_covars and (covars.dtype != torch.float32 or not covars.is_contiguous()):
        return None
    if has_quats_scales and (
        quats.dtype != torch.float32
        or scales.dtype != torch.float32
        or not quats.is_contiguous()
        or not scales.is_contiguous()
    ):
        return None
    if (
        means.dim() < 2
        or means.shape[-1] != 3
        or viewmats.shape[:-3] != means.shape[:-2]
        or Ks.shape[:-3] != means.shape[:-2]
        or viewmats.shape[-2:] != (4, 4)
        or Ks.shape[-2:] != (3, 3)
        or viewmats.shape[-3] != Ks.shape[-3]
        or radii.shape != means.shape[:-2] + (viewmats.shape[-3], means.shape[-2], 2)
        or conics.shape != means.shape[:-2] + (viewmats.shape[-3], means.shape[-2], 3)
        or v_means2d.shape != radii.shape
        or (
            v_depths is not None
            and v_depths.shape != means.shape[:-2] + (viewmats.shape[-3], means.shape[-2])
        )
        or v_conics.shape != conics.shape
    ):
        return None
    if has_covars and covars.shape != means.shape[:-1] + (3, 3):
        return None
    if has_quats_scales and (
        quats.shape != means.shape[:-1] + (4,)
        or scales.shape != means.shape[:-1] + (3,)
    ):
        return None
    scalar_shape = means.shape[:-2] + (viewmats.shape[-3], means.shape[-2])
    if compensations is not None and (
        compensations.dtype != torch.float32
        or not compensations.is_contiguous()
        or compensations.shape != scalar_shape
    ):
        return None
    if v_compensations is not None and (
        v_compensations.dtype != torch.float32
        or not v_compensations.is_contiguous()
        or v_compensations.shape != scalar_shape
    ):
        return None

    backend = load_supa_backend()
    if backend is None:
        return None

    out = backend.projection_ewa_3dgs_pinhole_bwd(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        radii,
        conics,
        compensations,
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
        bool(compute_v_means),
        bool(compute_v_covars),
        bool(compute_v_quats),
        bool(compute_v_scales),
        bool(compute_v_viewmats),
    )
    if synchronize_launch:
        synchronize(means.device)
    return out


def spherical_harmonics(
    degree: int,
    dirs: Tensor,
    coeffs: Tensor,
    synchronize_launch: bool = True,
) -> Optional[Tensor]:
    """Run the native SUPA spherical harmonics operator when supported."""
    degree = int(degree)
    required_k = (degree + 1) * (degree + 1)
    if (
        not is_supa_device(dirs)
        or dirs.dtype != torch.float32
        or coeffs.dtype != torch.float32
        or not dirs.is_contiguous()
        or not coeffs.is_contiguous()
        or degree < 0
        or degree > 4
        or dirs.dim() < 1
        or dirs.shape[-1] != 3
        or coeffs.dim() != dirs.dim() + 1
        or coeffs.shape[:-2] != dirs.shape[:-1]
        or coeffs.shape[-2] < required_k
        or coeffs.shape[-1] != 3
    ):
        return None
    backend = load_supa_backend()
    if backend is None or not hasattr(backend, "spherical_harmonics_fwd"):
        return None
    if not hasattr(backend, "spherical_harmonics_bwd"):
        return None

    out = _SphericalHarmonics.apply(degree, dirs, coeffs)
    if synchronize_launch:
        synchronize(dirs.device)
    return out


class _CompactSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: Tensor, compact_ids: Tensor, dim: int) -> Tensor:
        backend = load_supa_backend()
        if backend is None:
            raise RuntimeError("SUPA native backend is unavailable")
        dim = int(dim)
        if dim < 0:
            dim += tensor.dim()
        if dim < 0 or dim >= tensor.dim():
            raise IndexError("compact select dimension is out of range")

        input_shape = tuple(tensor.shape)
        outer = math.prod(input_shape[:dim])
        n_gaussians = input_shape[dim]
        row_size = math.prod(input_shape[dim + 1 :])
        flat_input = tensor.reshape(outer, n_gaussians, row_size).contiguous()
        out = backend.compact_select_fwd(flat_input, compact_ids)
        ctx.dim = dim
        ctx.input_shape = input_shape
        ctx.save_for_backward(compact_ids)
        output_shape = input_shape[:dim] + (compact_ids.numel(),) + input_shape[dim + 1 :]
        return out.reshape(output_shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        backend = load_supa_backend()
        if backend is None:
            raise RuntimeError("SUPA native backend is unavailable")
        (compact_ids,) = ctx.saved_tensors
        dim = ctx.dim
        input_shape = ctx.input_shape
        outer = math.prod(input_shape[:dim])
        n_gaussians = input_shape[dim]
        row_size = math.prod(input_shape[dim + 1 :])
        flat_grad = _materialize_projection_grad(
            grad_output.reshape(outer, compact_ids.numel(), row_size).contiguous()
        )
        grad_input = backend.compact_select_bwd(flat_grad, compact_ids, int(n_gaussians))
        synchronize(grad_output.device)
        return grad_input.reshape(input_shape), None, None


def visible_compact_ids(
    radii: Tensor,
    depths: Tensor,
    synchronize_launch: bool = True,
) -> Optional[Tuple[Tensor, int]]:
    """Build unique visible Gaussian ids with native SUPA mark/compact kernels."""
    if (
        not is_supa_device(depths)
        or not is_supa_device(radii)
        or radii.dtype != torch.int32
        or depths.dtype != torch.float32
        or not radii.is_contiguous()
        or not depths.is_contiguous()
        or radii.dim() < 2
        or radii.shape[-1] != 2
        or depths.shape != radii.shape[:-1]
    ):
        return None
    n_gaussians = radii.shape[-2]
    if n_gaussians > np.iinfo(np.int32).max:
        return None
    backend = load_supa_backend()
    if backend is None or not hasattr(backend, "visible_compact_ids"):
        return None

    flat_radii = radii.reshape(-1, n_gaussians, 2).contiguous()
    flat_depths = depths.reshape(-1, n_gaussians).contiguous()
    compact_ids, visible_count = backend.visible_compact_ids(flat_radii, flat_depths)
    if synchronize_launch:
        synchronize(depths.device)
    count = int(visible_count.cpu().item())
    return compact_ids[:count].contiguous(), count


def compact_select(tensor: Tensor, compact_ids: Tensor, dim: int) -> Optional[Tensor]:
    """Select unique compact ids along one dimension with native SUPA backward."""
    if (
        not is_supa_device(tensor)
        or not is_supa_device(compact_ids)
        or tensor.dtype != torch.float32
        or compact_ids.dtype != torch.int32
        or not tensor.is_contiguous()
        or not compact_ids.is_contiguous()
        or compact_ids.dim() != 1
        or tensor.dim() == 0
    ):
        return None
    dim = int(dim)
    if dim < 0:
        dim += tensor.dim()
    if dim < 0 or dim >= tensor.dim():
        return None
    if tensor.shape[dim] > np.iinfo(np.int32).max:
        return None
    backend = load_supa_backend()
    if backend is None or not hasattr(backend, "compact_select_fwd"):
        return None
    if not hasattr(backend, "compact_select_bwd"):
        return None

    out = _CompactSelect.apply(tensor, compact_ids, dim)
    synchronize(tensor.device)
    return out


def tile_count_unpacked(
    means2d: Tensor,
    radii: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    synchronize_launch: bool = True,
) -> Optional[Tensor]:
    """Run the native SUPA unpacked tile-count prototype when supported."""
    if (
        not is_supa_device(means2d)
        or means2d.dtype != torch.float32
        or radii.dtype != torch.int32
        or not means2d.is_contiguous()
        or not radii.is_contiguous()
        or means2d.dim() != 3
        or means2d.shape[-1] != 2
        or radii.shape != means2d.shape
    ):
        return None
    backend = load_supa_backend()
    if backend is None or not hasattr(backend, "tile_count_unpacked"):
        return None

    out = backend.tile_count_unpacked(
        means2d,
        radii,
        int(tile_size),
        int(tile_width),
        int(tile_height),
    )
    if synchronize_launch:
        synchronize(means2d.device)
    return out


def tile_count_alpha_unpacked(
    means2d: Tensor,
    radii: Tensor,
    conics: Tensor,
    opacities: Tensor,
    alpha_threshold: float,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    synchronize_launch: bool = True,
) -> Optional[Tensor]:
    """Run native SUPA tile counts after applying an alpha-threshold footprint."""
    if (
        not is_supa_device(means2d)
        or means2d.dtype != torch.float32
        or radii.dtype != torch.int32
        or conics.dtype != torch.float32
        or opacities.dtype != torch.float32
        or not means2d.is_contiguous()
        or not radii.is_contiguous()
        or not conics.is_contiguous()
        or not opacities.is_contiguous()
        or means2d.dim() != 3
        or means2d.shape[-1] != 2
        or radii.shape != means2d.shape
        or conics.shape != means2d.shape[:2] + (3,)
        or opacities.shape != means2d.shape[:2]
        or not 0.0 < float(alpha_threshold) < 1.0
    ):
        return None
    backend = load_supa_backend()
    if backend is None or not hasattr(backend, "tile_count_alpha_unpacked"):
        return None

    out = backend.tile_count_alpha_unpacked(
        means2d,
        radii,
        conics,
        opacities,
        float(alpha_threshold),
        int(tile_size),
        int(tile_width),
        int(tile_height),
    )
    if synchronize_launch:
        synchronize(means2d.device)
    return out


def _isect_tiles_unpacked_lanes_cpu_starts(
    backend,
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tiles_per_gauss: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
):
    tiles_per_gauss_cpu = tiles_per_gauss.detach().cpu().contiguous()
    flat_counts = tiles_per_gauss_cpu.reshape(-1).to(torch.int64)
    starts_cpu = torch.cumsum(flat_counts, dim=0) - flat_counts
    if starts_cpu.numel() > 0 and int(starts_cpu[-1].item() + flat_counts[-1].item()) > np.iinfo(np.int32).max:
        return None
    # Keep int32 metadata lanes flat: 2D int32 tensors copied from CPU have
    # exposed incomplete data to native pointer kernels on BRPyTorch.
    starts = starts_cpu.to(torch.int32).to(means2d.device).contiguous()
    native_tiles_per_gauss = flat_counts.to(torch.int32).to(means2d.device).contiguous()
    return backend.isect_tiles_unpacked_lanes(
        means2d,
        radii,
        depths,
        native_tiles_per_gauss,
        starts,
        int(tile_size),
        int(tile_width),
        int(tile_height),
    )


def isect_tiles_unpacked_lanes(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tiles_per_gauss: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    synchronize_launch: bool = True,
    return_cpu: bool = True,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Expand unpacked tile intersections into int32 lanes without packed int64 IDs."""
    if (
        not is_supa_device(means2d)
        or means2d.dtype != torch.float32
        or radii.dtype != torch.int32
        or depths.dtype != torch.float32
        or tiles_per_gauss.dtype != torch.int32
        or not means2d.is_contiguous()
        or not radii.is_contiguous()
        or not depths.is_contiguous()
        or not tiles_per_gauss.is_contiguous()
        or means2d.dim() != 3
        or means2d.shape[-1] != 2
        or radii.shape != means2d.shape
        or depths.shape != means2d.shape[:2]
        or tiles_per_gauss.shape != depths.shape
        or tile_size <= 0
        or tile_width <= 0
        or tile_height <= 0
    ):
        return None
    backend = load_supa_backend()
    if backend is None or not hasattr(backend, "isect_tiles_unpacked_lanes"):
        return None

    if hasattr(backend, "isect_tiles_unpacked_lanes_from_counts"):
        out = backend.isect_tiles_unpacked_lanes_from_counts(
            means2d,
            radii,
            depths,
            tiles_per_gauss,
            int(tile_size),
            int(tile_width),
            int(tile_height),
        )
    else:
        out = _isect_tiles_unpacked_lanes_cpu_starts(
            backend,
            means2d,
            radii,
            depths,
            tiles_per_gauss,
            tile_size,
            tile_width,
            tile_height,
        )
    if out is None:
        return None
    if synchronize_launch:
        synchronize(means2d.device)
    if return_cpu:
        return tuple(tensor.detach().cpu().contiguous() for tensor in out)
    return tuple(tensor.contiguous() for tensor in out)


def isect_tiles_alpha_unpacked_lanes(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    conics: Tensor,
    opacities: Tensor,
    tiles_per_gauss: Tensor,
    alpha_threshold: float,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    synchronize_launch: bool = True,
    return_cpu: bool = True,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Expand alpha-thresholded tile intersections into int32 lanes on SUPA."""
    if (
        not is_supa_device(means2d)
        or means2d.dtype != torch.float32
        or radii.dtype != torch.int32
        or depths.dtype != torch.float32
        or conics.dtype != torch.float32
        or opacities.dtype != torch.float32
        or tiles_per_gauss.dtype != torch.int32
        or not means2d.is_contiguous()
        or not radii.is_contiguous()
        or not depths.is_contiguous()
        or not conics.is_contiguous()
        or not opacities.is_contiguous()
        or not tiles_per_gauss.is_contiguous()
        or means2d.dim() != 3
        or means2d.shape[-1] != 2
        or radii.shape != means2d.shape
        or depths.shape != means2d.shape[:2]
        or conics.shape != means2d.shape[:2] + (3,)
        or opacities.shape != means2d.shape[:2]
        or tiles_per_gauss.shape != depths.shape
        or tile_size <= 0
        or tile_width <= 0
        or tile_height <= 0
        or not 0.0 < float(alpha_threshold) < 1.0
    ):
        return None
    backend = load_supa_backend()
    if backend is None or not hasattr(backend, "isect_tiles_alpha_unpacked_lanes_from_counts"):
        return None

    out = backend.isect_tiles_alpha_unpacked_lanes_from_counts(
        means2d,
        radii,
        depths,
        conics,
        opacities,
        tiles_per_gauss,
        float(alpha_threshold),
        int(tile_size),
        int(tile_width),
        int(tile_height),
    )
    if synchronize_launch:
        synchronize(means2d.device)
    if return_cpu:
        return tuple(tensor.detach().cpu().contiguous() for tensor in out)
    return tuple(tensor.contiguous() for tensor in out)


def tiled_offsets_from_lanes(
    dense_tile_ids: Tensor,
    flatten_ids: Tensor,
    isect_ids_lo: Tensor,
    n_images: int,
    tile_width: int,
    tile_height: int,
    synchronize_launch: bool = True,
) -> Optional[Tuple[Tensor, Tensor]]:
    """Build tile offsets and depth-sorted flatten ids from native int32 lanes."""
    if (
        not is_supa_device(dense_tile_ids)
        or not is_supa_device(flatten_ids)
        or not is_supa_device(isect_ids_lo)
        or dense_tile_ids.dtype != torch.int32
        or flatten_ids.dtype != torch.int32
        or isect_ids_lo.dtype != torch.int32
        or not dense_tile_ids.is_contiguous()
        or not flatten_ids.is_contiguous()
        or not isect_ids_lo.is_contiguous()
        or dense_tile_ids.dim() != 1
        or flatten_ids.shape != dense_tile_ids.shape
        or isect_ids_lo.shape != dense_tile_ids.shape
        or n_images < 0
        or tile_width < 0
        or tile_height < 0
    ):
        return None
    backend = load_supa_backend()
    if backend is None or not hasattr(backend, "tiled_offsets_from_lanes"):
        return None

    out = backend.tiled_offsets_from_lanes(
        dense_tile_ids,
        flatten_ids,
        isect_ids_lo,
        int(n_images),
        int(tile_width),
        int(tile_height),
    )
    if synchronize_launch:
        synchronize(dense_tile_ids.device)
    return out


def _max_tiled_range(tile_offsets: Tensor, n_isects: int) -> Optional[int]:
    if tile_offsets.numel() == 0:
        return 0
    try:
        flat_offsets = tile_offsets.reshape(-1)
        final_offset = torch.full(
            (1,),
            int(n_isects),
            dtype=flat_offsets.dtype,
            device=flat_offsets.device,
        )
        range_ends = torch.cat((flat_offsets[1:], final_offset), dim=0)
        return int((range_ends - flat_offsets).max().detach().cpu().item())
    except (NotImplementedError, RuntimeError):
        return None


def _normalize_tiled_backward_mode(mode: str) -> str:
    value = str(mode).strip().lower().replace("-", "_")
    if value in {"auto", "tile_reduce", "pixel", "parallel"}:
        return value
    return "auto"


def _tiled_offsets_from_lanes_torch_sort(
    dense_tile_ids: Tensor,
    flatten_ids: Tensor,
    isect_ids_lo: Tensor,
    n_images: int,
    tile_width: int,
    tile_height: int,
) -> Optional[Tuple[Tensor, Tensor]]:
    """Build tiled offsets on SUPA without the CPU round-trip.

    The old custom native path grouped intersections by tile and then ran an
    insertion sort inside every tile. Large scenes can put hundreds of
    intersections in one tile, so that O(k^2) sort dominates metadata time.
    BRPyTorch's device-side stable argsort is much faster for this int32
    metadata, and the final flatten IDs are still kept on SUPA for the tiled
    raster kernels.
    """
    if (
        not is_supa_device(dense_tile_ids)
        or not is_supa_device(flatten_ids)
        or not is_supa_device(isect_ids_lo)
        or dense_tile_ids.dtype != torch.int32
        or flatten_ids.dtype != torch.int32
        or isect_ids_lo.dtype != torch.int32
        or not dense_tile_ids.is_contiguous()
        or not flatten_ids.is_contiguous()
        or not isect_ids_lo.is_contiguous()
        or dense_tile_ids.dim() != 1
        or flatten_ids.shape != dense_tile_ids.shape
        or isect_ids_lo.shape != dense_tile_ids.shape
        or n_images < 0
        or tile_width < 0
        or tile_height < 0
    ):
        return None

    n_tiles = int(n_images) * int(tile_width) * int(tile_height)
    tile_offsets_shape = (int(n_images), int(tile_height), int(tile_width))
    if n_tiles == 0 or dense_tile_ids.numel() == 0:
        return (
            torch.zeros(
                tile_offsets_shape,
                dtype=torch.int32,
                device=dense_tile_ids.device,
            ),
            flatten_ids.new_empty((0,)),
        )

    try:
        # Stable depth sort followed by stable tile sort is equivalent to a
        # stable sort by the packed (tile, depth) key used by the CPU reference.
        order_depth = torch.argsort(isect_ids_lo, stable=True)
        depth_sorted_dense = dense_tile_ids[order_depth]
        depth_sorted_flatten = flatten_ids[order_depth]
        order_tile = torch.argsort(depth_sorted_dense, stable=True)
        sorted_dense = depth_sorted_dense[order_tile]
        sorted_flatten_ids = depth_sorted_flatten[order_tile].contiguous()

        tile_counts = torch.bincount(
            sorted_dense.to(torch.long),
            minlength=n_tiles,
        ).to(torch.int32)
        tile_offsets = (
            torch.cumsum(tile_counts, dim=0).to(torch.int32) - tile_counts
        ).reshape(tile_offsets_shape)
    except (NotImplementedError, RuntimeError):
        return None
    return tile_offsets.contiguous(), sorted_flatten_ids


def _isect_tiles_unpacked_metadata_np(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tiles_per_gauss: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    (
        I,
        n_isects,
        flatten_ids_np,
        dense_tile_ids,
        isect_ids_hi,
        isect_ids_lo,
    ) = _expand_isect_tiles_unpacked_np(
        means2d,
        radii,
        depths,
        tiles_per_gauss,
        tile_size,
        tile_width,
        tile_height,
    )
    if n_isects == 0:
        empty_isects = np.empty(0, dtype=np.int64)
        offsets = np.zeros((I, tile_height, tile_width), dtype=np.int32) if sort else None
        return empty_isects, flatten_ids_np, offsets

    isect_ids_np = _pack_isect_ids_np(isect_ids_hi, isect_ids_lo)
    if sort:
        order = np.argsort(isect_ids_np, kind="stable")
        isect_ids_np = isect_ids_np[order]
        flatten_ids_np = flatten_ids_np[order]

    offsets = None
    if sort:
        offsets = _tile_offsets_from_dense_ids_np(
            dense_tile_ids,
            I,
            tile_width,
            tile_height,
        )
    return isect_ids_np, flatten_ids_np, offsets


def _expand_isect_tiles_unpacked_np(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tiles_per_gauss: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means2d_cpu = means2d.detach().cpu().numpy()
    radii_cpu = radii.detach().cpu().numpy()
    depths_cpu = depths.detach().cpu().numpy()
    tiles_per_gauss_cpu = tiles_per_gauss.detach().cpu().numpy().astype(np.int64, copy=False)
    I, N = means2d_cpu.shape[:2]
    n_isects = int(tiles_per_gauss_cpu.sum())
    if n_isects == 0:
        empty_flatten = np.empty(0, dtype=np.int32)
        empty_dense = np.empty(0, dtype=np.int32)
        empty_hi = np.empty(0, dtype=np.int32)
        empty_lo = np.empty(0, dtype=np.int32)
        return I, n_isects, empty_flatten, empty_dense, empty_hi, empty_lo

    tile_n_bits = (tile_width * tile_height).bit_length()

    inv_tile_size = 1.0 / float(tile_size)
    tile_means2d = means2d_cpu * inv_tile_size
    tile_radii = radii_cpu.astype(np.float32, copy=False) * inv_tile_size
    tile_mins = np.floor(tile_means2d - tile_radii).astype(np.int32)
    tile_maxs = np.ceil(tile_means2d + tile_radii).astype(np.int32)
    tile_mins[..., 0] = np.clip(tile_mins[..., 0], 0, tile_width)
    tile_mins[..., 1] = np.clip(tile_mins[..., 1], 0, tile_height)
    tile_maxs[..., 0] = np.clip(tile_maxs[..., 0], 0, tile_width)
    tile_maxs[..., 1] = np.clip(tile_maxs[..., 1], 0, tile_height)

    flat_counts = tiles_per_gauss_cpu.reshape(-1).astype(np.int32, copy=False)
    valid_flat_ids = np.flatnonzero(flat_counts > 0).astype(np.int32)
    valid_counts = flat_counts[valid_flat_ids]
    flatten_ids_np = np.repeat(valid_flat_ids, valid_counts)

    starts = np.cumsum(valid_counts, dtype=np.int32) - valid_counts
    local_ids = np.arange(n_isects, dtype=np.int32) - np.repeat(starts, valid_counts)

    flat_tile_mins = tile_mins.reshape(I * N, 2)
    flat_tile_maxs = tile_maxs.reshape(I * N, 2)
    valid_min_x = flat_tile_mins[valid_flat_ids, 0]
    valid_min_y = flat_tile_mins[valid_flat_ids, 1]
    tile_widths = flat_tile_maxs[valid_flat_ids, 0] - valid_min_x
    repeated_widths = np.repeat(tile_widths, valid_counts)
    local_y, local_x = np.divmod(local_ids, repeated_widths)
    tile_x = np.repeat(valid_min_x, valid_counts) + local_x
    tile_y = np.repeat(valid_min_y, valid_counts) + local_y
    tile_ids = tile_y * tile_width + tile_x

    image_ids = flatten_ids_np // N
    dense_tile_ids = image_ids * (tile_width * tile_height) + tile_ids
    isect_ids_hi = ((image_ids << tile_n_bits) | tile_ids).astype(np.int32)
    depth_ids = depths_cpu.astype(np.float32, copy=False).view(np.int32).reshape(-1)
    isect_ids_lo = depth_ids[flatten_ids_np]
    return I, n_isects, flatten_ids_np, dense_tile_ids, isect_ids_hi, isect_ids_lo


def _pack_isect_ids_np(isect_ids_hi: np.ndarray, isect_ids_lo: np.ndarray) -> np.ndarray:
    isect_ids_np = (isect_ids_hi.astype(np.int64) << np.int64(32)) | (
        isect_ids_lo.astype(np.int64) & np.int64(0xFFFFFFFF)
    )
    return isect_ids_np


def _tile_offsets_from_dense_ids_np(
    dense_tile_ids: np.ndarray,
    I: int,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    counts = np.bincount(
        dense_tile_ids,
        minlength=I * tile_height * tile_width,
    ).astype(np.int32, copy=False)
    return (np.cumsum(counts, dtype=np.int32) - counts).reshape(
        I, tile_height, tile_width
    )


def _isect_tiles_unpacked_offsets_np(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tiles_per_gauss: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    (
        I,
        n_isects,
        flatten_ids_np,
        dense_tile_ids,
        isect_ids_hi,
        isect_ids_lo,
    ) = _expand_isect_tiles_unpacked_np(
        means2d,
        radii,
        depths,
        tiles_per_gauss,
        tile_size,
        tile_width,
        tile_height,
    )

    tile_offsets_np = _tile_offsets_from_dense_ids_np(
        dense_tile_ids,
        I,
        tile_width,
        tile_height,
    )
    if n_isects > 0:
        isect_ids_np = _pack_isect_ids_np(isect_ids_hi, isect_ids_lo)
        order = np.argsort(isect_ids_np, kind="stable")
        flatten_ids_np = flatten_ids_np[order]
    return flatten_ids_np, tile_offsets_np


def _isect_tiles_unpacked_metadata_cpu(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tiles_per_gauss: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool,
) -> Tuple[Tensor, Tensor]:
    isect_ids_np, flatten_ids_np, _ = _isect_tiles_unpacked_metadata_np(
        means2d,
        radii,
        depths,
        tiles_per_gauss,
        tile_size,
        tile_width,
        tile_height,
        sort,
    )
    isect_ids = torch.from_numpy(np.ascontiguousarray(isect_ids_np))
    flatten_ids = torch.from_numpy(np.ascontiguousarray(flatten_ids_np.astype(np.int32)))
    return isect_ids.contiguous(), flatten_ids.contiguous()


def _isect_tiles_unpacked_offsets_cpu(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tiles_per_gauss: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> Tuple[Tensor, Tensor]:
    flatten_ids_np, tile_offsets_np = _isect_tiles_unpacked_offsets_np(
        means2d,
        radii,
        depths,
        tiles_per_gauss,
        tile_size,
        tile_width,
        tile_height,
    )
    tile_offsets = torch.from_numpy(np.ascontiguousarray(tile_offsets_np))
    flatten_ids = torch.from_numpy(np.ascontiguousarray(flatten_ids_np.astype(np.int32)))
    return tile_offsets.contiguous(), flatten_ids.contiguous()


def _isect_tiles_unpacked_offsets_alpha_cpu(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    conics: Tensor,
    opacities: Tensor,
    alpha_threshold: float,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    means2d_cpu = means2d.detach().cpu().numpy().astype(np.float32, copy=False)
    radii_cpu = radii.detach().cpu().numpy().astype(np.int32, copy=False)
    depths_cpu = depths.detach().cpu().numpy().astype(np.float32, copy=False)
    conics_cpu = conics.detach().cpu().numpy().astype(np.float32, copy=False)
    opacities_cpu = opacities.detach().cpu().numpy().astype(np.float32, copy=False)
    I, N = means2d_cpu.shape[:2]
    threshold = float(alpha_threshold)

    valid = (radii_cpu > 0).all(axis=-1) & np.isfinite(opacities_cpu) & (
        opacities_cpu > threshold
    )
    extent = radii_cpu.astype(np.float32, copy=False)

    A = conics_cpu[..., 0]
    B = conics_cpu[..., 1]
    C = conics_cpu[..., 2]
    disc = B * B - A * C
    with np.errstate(divide="ignore", invalid="ignore"):
        level = np.minimum(9.0, 2.0 * np.log(opacities_cpu / threshold))
        neg_level_over_disc = -level / disc
        x_extent = np.sqrt(neg_level_over_disc * C)
        y_extent = np.sqrt(neg_level_over_disc * A)
    ellipse_valid = (
        valid
        & np.isfinite(A)
        & np.isfinite(B)
        & np.isfinite(C)
        & (A > 0.0)
        & (C > 0.0)
        & (disc < -1.0e-20)
        & np.isfinite(level)
        & (level > 0.0)
        & np.isfinite(x_extent)
        & np.isfinite(y_extent)
    )
    extent[..., 0] = np.where(ellipse_valid, x_extent, extent[..., 0])
    extent[..., 1] = np.where(ellipse_valid, y_extent, extent[..., 1])

    inv_tile_size = 1.0 / float(tile_size)
    tile_means2d = means2d_cpu * inv_tile_size
    tile_extents = extent * inv_tile_size
    tile_mins = np.floor(tile_means2d - tile_extents).astype(np.int32)
    tile_maxs = np.ceil(tile_means2d + tile_extents).astype(np.int32)
    tile_mins[..., 0] = np.clip(tile_mins[..., 0], 0, tile_width)
    tile_mins[..., 1] = np.clip(tile_mins[..., 1], 0, tile_height)
    tile_maxs[..., 0] = np.clip(tile_maxs[..., 0], 0, tile_width)
    tile_maxs[..., 1] = np.clip(tile_maxs[..., 1], 0, tile_height)

    spans = np.maximum(tile_maxs - tile_mins, 0)
    counts = np.where(valid, spans[..., 0] * spans[..., 1], 0).astype(np.int32)
    n_isects = int(counts.sum())
    if n_isects == 0:
        tiles_per_gauss = torch.from_numpy(np.ascontiguousarray(counts)).to(
            means2d.device
        )
        tile_offsets = torch.zeros((I, tile_height, tile_width), dtype=torch.int32)
        flatten_ids = torch.empty((0,), dtype=torch.int32, device=means2d.device)
        return tiles_per_gauss.contiguous(), tile_offsets.contiguous(), flatten_ids

    flat_counts = counts.reshape(-1)
    valid_flat_ids = np.flatnonzero(flat_counts > 0).astype(np.int32)
    valid_counts = flat_counts[valid_flat_ids].astype(np.int32, copy=False)
    flatten_ids_np = np.repeat(valid_flat_ids, valid_counts).astype(np.int32, copy=False)

    starts = np.cumsum(valid_counts, dtype=np.int64) - valid_counts.astype(np.int64)
    local_ids = np.arange(n_isects, dtype=np.int64) - np.repeat(starts, valid_counts)

    flat_tile_mins = tile_mins.reshape(I * N, 2)
    flat_tile_maxs = tile_maxs.reshape(I * N, 2)
    valid_min_x = flat_tile_mins[valid_flat_ids, 0]
    valid_min_y = flat_tile_mins[valid_flat_ids, 1]
    tile_widths = flat_tile_maxs[valid_flat_ids, 0] - valid_min_x
    repeated_widths = np.repeat(tile_widths, valid_counts).astype(np.int64, copy=False)
    local_y, local_x = np.divmod(local_ids, repeated_widths)
    tile_x = np.repeat(valid_min_x, valid_counts).astype(np.int64, copy=False) + local_x
    tile_y = np.repeat(valid_min_y, valid_counts).astype(np.int64, copy=False) + local_y
    tile_ids = (tile_y * tile_width + tile_x).astype(np.int32, copy=False)

    image_ids = flatten_ids_np // N
    dense_tile_ids = image_ids * (tile_width * tile_height) + tile_ids
    tile_offsets_np = _tile_offsets_from_dense_ids_np(
        dense_tile_ids,
        I,
        tile_width,
        tile_height,
    )
    tile_n_bits = (tile_width * tile_height).bit_length()
    isect_ids_hi = ((image_ids << tile_n_bits) | tile_ids).astype(np.int32)
    depth_ids = depths_cpu.astype(np.float32, copy=False).view(np.int32).reshape(-1)
    isect_ids_lo = depth_ids[flatten_ids_np]
    isect_ids_np = _pack_isect_ids_np(isect_ids_hi, isect_ids_lo)
    order = np.argsort(isect_ids_np, kind="stable")
    flatten_ids_np = flatten_ids_np[order]

    tiles_per_gauss = torch.from_numpy(np.ascontiguousarray(counts)).to(means2d.device)
    tile_offsets = torch.from_numpy(np.ascontiguousarray(tile_offsets_np))
    flatten_ids = torch.from_numpy(np.ascontiguousarray(flatten_ids_np.astype(np.int32)))
    return tiles_per_gauss.contiguous(), tile_offsets.contiguous(), flatten_ids.contiguous()


def _isect_tiles_unpacked_offsets_lanes_cpu(
    dense_tile_ids: Tensor,
    flatten_ids: Tensor,
    isect_ids_hi: Tensor,
    isect_ids_lo: Tensor,
    n_images: int,
    tile_width: int,
    tile_height: int,
) -> Tuple[Tensor, Tensor]:
    dense_tile_ids_np = dense_tile_ids.detach().cpu().numpy().astype(np.int32, copy=False)
    flatten_ids_np = flatten_ids.detach().cpu().numpy().astype(np.int32, copy=False)
    isect_ids_hi_np = isect_ids_hi.detach().cpu().numpy().astype(np.int32, copy=False)
    isect_ids_lo_np = isect_ids_lo.detach().cpu().numpy().astype(np.int32, copy=False)
    tile_offsets_np = _tile_offsets_from_dense_ids_np(
        dense_tile_ids_np,
        n_images,
        tile_width,
        tile_height,
    )
    if flatten_ids_np.size > 0:
        isect_ids_np = _pack_isect_ids_np(isect_ids_hi_np, isect_ids_lo_np)
        order = np.argsort(isect_ids_np, kind="stable")
        flatten_ids_np = flatten_ids_np[order]
    tile_offsets = torch.from_numpy(np.ascontiguousarray(tile_offsets_np))
    sorted_flatten_ids = torch.from_numpy(np.ascontiguousarray(flatten_ids_np))
    return tile_offsets.contiguous(), sorted_flatten_ids.contiguous()


def isect_tiles_unpacked(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    synchronize_launch: bool = True,
) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
    """Run the SUPA unpacked AABB tile-intersection prototype.

    ``tiles_per_gauss`` is computed by the native SUPA count kernel. The packed
    ``isect_ids`` and ``flatten_ids`` metadata is returned on CPU for now because
    BRPyTorch currently drops the high 32 bits of int64 tensors copied to SUPA.
    """
    if (
        not is_supa_device(means2d)
        or means2d.dtype != torch.float32
        or radii.dtype != torch.int32
        or depths.dtype != torch.float32
        or not means2d.is_contiguous()
        or not radii.is_contiguous()
        or not depths.is_contiguous()
        or means2d.dim() != 3
        or means2d.shape[-1] != 2
        or radii.shape != means2d.shape
        or depths.shape != means2d.shape[:2]
        or tile_size <= 0
        or tile_width < 0
        or tile_height < 0
    ):
        return None
    total = math.prod(means2d.shape[:2])
    if total == 0:
        empty_ids = torch.empty((0,), dtype=torch.int64)
        empty_flatten = torch.empty((0,), dtype=torch.int32)
        tiles_per_gauss = torch.empty(means2d.shape[:2], dtype=torch.int32, device=means2d.device)
        return tiles_per_gauss, empty_ids, empty_flatten

    tiles_per_gauss = tile_count_unpacked(
        means2d,
        radii,
        tile_size,
        tile_width,
        tile_height,
        synchronize_launch=synchronize_launch,
    )
    if tiles_per_gauss is None:
        return None
    isect_ids, flatten_ids = _isect_tiles_unpacked_metadata_cpu(
        means2d,
        radii,
        depths,
        tiles_per_gauss,
        tile_size,
        tile_width,
        tile_height,
        sort,
    )
    if synchronize_launch:
        synchronize(means2d.device)
    return tiles_per_gauss, isect_ids, flatten_ids


def isect_tiles_unpacked_offsets(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    synchronize_launch: bool = True,
    return_backend: bool = False,
    use_native_offsets: Optional[bool] = None,
    conics: Optional[Tensor] = None,
    opacities: Optional[Tensor] = None,
    alpha_threshold: float = 0.0,
) -> Optional[Tuple[Tensor, ...]]:
    """Build sorted tiled raster metadata without materializing packed IDs."""
    if use_native_offsets is None:
        use_native_offsets = False
    else:
        use_native_offsets = bool(use_native_offsets)

    if (
        not is_supa_device(means2d)
        or means2d.dtype != torch.float32
        or radii.dtype != torch.int32
        or depths.dtype != torch.float32
        or not means2d.is_contiguous()
        or not radii.is_contiguous()
        or not depths.is_contiguous()
        or means2d.dim() != 3
        or means2d.shape[-1] != 2
        or radii.shape != means2d.shape
        or depths.shape != means2d.shape[:2]
        or tile_size <= 0
        or tile_width < 0
        or tile_height < 0
    ):
        return None
    total = math.prod(means2d.shape[:2])
    if total == 0:
        tiles_per_gauss = torch.empty(means2d.shape[:2], dtype=torch.int32, device=means2d.device)
        tile_offsets = torch.zeros(
            means2d.shape[:1] + (tile_height, tile_width),
            dtype=torch.int32,
            device=means2d.device,
        )
        flatten_ids = torch.empty((0,), dtype=torch.int32, device=means2d.device)
        if return_backend:
            return tiles_per_gauss, tile_offsets, flatten_ids, "empty"
        return tiles_per_gauss, tile_offsets, flatten_ids

    threshold = float(alpha_threshold)
    if threshold > 0.0:
        if (
            conics is None
            or opacities is None
            or conics.dtype != torch.float32
            or opacities.dtype != torch.float32
            or not conics.is_contiguous()
            or not opacities.is_contiguous()
            or conics.shape != means2d.shape[:2] + (3,)
            or opacities.shape != means2d.shape[:2]
        ):
            return None
        if use_native_offsets:
            tiles_per_gauss = tile_count_alpha_unpacked(
                means2d,
                radii,
                conics,
                opacities,
                threshold,
                tile_size,
                tile_width,
                tile_height,
                synchronize_launch=False,
            )
            if tiles_per_gauss is not None:
                lanes = isect_tiles_alpha_unpacked_lanes(
                    means2d,
                    radii,
                    depths,
                    conics,
                    opacities,
                    tiles_per_gauss,
                    threshold,
                    tile_size,
                    tile_width,
                    tile_height,
                    synchronize_launch=False,
                    return_cpu=False,
                )
                if lanes is not None:
                    native_offsets = _tiled_offsets_from_lanes_torch_sort(
                        lanes[0],
                        lanes[1],
                        lanes[3],
                        int(means2d.shape[0]),
                        tile_width,
                        tile_height,
                    )
                    if native_offsets is not None:
                        tile_offsets, flatten_ids = native_offsets
                        if synchronize_launch:
                            synchronize(means2d.device)
                        if return_backend:
                            return (
                                tiles_per_gauss,
                                tile_offsets,
                                flatten_ids,
                                "supa_native_alpha_threshold",
                            )
                        return tiles_per_gauss, tile_offsets, flatten_ids

        tiles_per_gauss, tile_offsets, flatten_ids = _isect_tiles_unpacked_offsets_alpha_cpu(
            means2d,
            radii,
            depths,
            conics,
            opacities,
            threshold,
            tile_size,
            tile_width,
            tile_height,
        )
        if synchronize_launch:
            synchronize(means2d.device)
        if return_backend:
            return tiles_per_gauss, tile_offsets, flatten_ids, "cpu_numpy_alpha_threshold"
        return tiles_per_gauss, tile_offsets, flatten_ids

    tiles_per_gauss = tile_count_unpacked(
        means2d,
        radii,
        tile_size,
        tile_width,
        tile_height,
        synchronize_launch=synchronize_launch,
    )
    if tiles_per_gauss is None:
        return None
    metadata_backend = "cpu_numpy"
    if use_native_offsets:
        lanes = isect_tiles_unpacked_lanes(
            means2d,
            radii,
            depths,
            tiles_per_gauss,
            tile_size,
            tile_width,
            tile_height,
            synchronize_launch=synchronize_launch,
            return_cpu=False,
        )
        if lanes is not None:
            native_offsets = _tiled_offsets_from_lanes_torch_sort(
                lanes[0],
                lanes[1],
                lanes[3],
                int(means2d.shape[0]),
                tile_width,
                tile_height,
            )
            if synchronize_launch and native_offsets is not None:
                synchronize(means2d.device)
            if native_offsets is not None:
                tile_offsets, flatten_ids = native_offsets
                metadata_backend = "supa_native_offsets"
            else:
                native_offsets = tiled_offsets_from_lanes(
                    lanes[0],
                    lanes[1],
                    lanes[3],
                    int(means2d.shape[0]),
                    tile_width,
                    tile_height,
                    synchronize_launch=synchronize_launch,
                )
                if native_offsets is not None:
                    tile_offsets, flatten_ids = native_offsets
                    metadata_backend = "supa_native_offsets"
                else:
                    tile_offsets, flatten_ids = _isect_tiles_unpacked_offsets_lanes_cpu(
                        *lanes,
                        int(means2d.shape[0]),
                        tile_width,
                        tile_height,
                    )
                    metadata_backend = "supa_native_lanes_cpu_offsets"
        else:
            tile_offsets, flatten_ids = _isect_tiles_unpacked_offsets_cpu(
                means2d,
                radii,
                depths,
                tiles_per_gauss,
                tile_size,
                tile_width,
                tile_height,
            )
    else:
        tile_offsets, flatten_ids = _isect_tiles_unpacked_offsets_cpu(
            means2d,
            radii,
            depths,
            tiles_per_gauss,
            tile_size,
            tile_width,
            tile_height,
        )
    if synchronize_launch:
        synchronize(means2d.device)
    if return_backend:
        return tiles_per_gauss, tile_offsets, flatten_ids, metadata_backend
    return tiles_per_gauss, tile_offsets, flatten_ids


def isect_offset_encode(
    isect_ids: Tensor,
    n_images: int,
    tile_width: int,
    tile_height: int,
) -> Optional[Tensor]:
    """Encode sorted intersection IDs to public tile offsets on CPU."""
    if (
        isect_ids.dtype != torch.int64
        or isect_ids.dim() != 1
        or n_images < 0
        or tile_width < 0
        or tile_height < 0
    ):
        return None
    isect_ids_cpu = isect_ids.detach().cpu().contiguous()
    tile_n_bits = (tile_width * tile_height).bit_length()
    tile_counts = torch.zeros((n_images, tile_height, tile_width), dtype=torch.int64)
    if isect_ids_cpu.numel() > 0:
        isect_ids_uq, counts = torch.unique_consecutive(
            isect_ids_cpu >> 32, return_counts=True
        )
        image_ids_uq = isect_ids_uq >> tile_n_bits
        tile_ids_uq = isect_ids_uq & ((1 << tile_n_bits) - 1)
        tile_ids_x_uq = tile_ids_uq % tile_width
        tile_ids_y_uq = tile_ids_uq // tile_width
        tile_counts[image_ids_uq, tile_ids_y_uq, tile_ids_x_uq] = counts

    cum_tile_counts = torch.cumsum(tile_counts.flatten(), dim=0).reshape_as(tile_counts)
    return (cum_tile_counts - tile_counts).int().contiguous()


def rasterize_to_pixels_tiled_fwd(
    means2d: Tensor,
    conics: Tensor,
    features: Tensor,
    opacities: Tensor,
    width: int,
    height: int,
    tile_size: int,
    tile_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    synchronize_launch: bool = True,
) -> Optional[Tuple[Tensor, Tensor]]:
    """Run the native SUPA forward-only tiled rasterization prototype."""
    if (
        not is_supa_device(means2d)
        or means2d.dtype != torch.float32
        or conics.dtype != torch.float32
        or features.dtype != torch.float32
        or opacities.dtype != torch.float32
        or tile_offsets.dtype != torch.int32
        or flatten_ids.dtype != torch.int32
        or not means2d.is_contiguous()
        or not conics.is_contiguous()
        or not features.is_contiguous()
        or not opacities.is_contiguous()
        or means2d.dim() != 3
        or means2d.shape[-1] != 2
        or conics.shape != means2d.shape[:-1] + (3,)
        or features.dim() != 3
        or features.shape[:2] != means2d.shape[:2]
        or opacities.shape != means2d.shape[:2]
        or tile_offsets.dim() != 3
        or tile_offsets.shape[0] != means2d.shape[0]
        or flatten_ids.dim() != 1
        or width <= 0
        or height <= 0
        or tile_size <= 0
    ):
        return None
    if backgrounds is not None and (
        backgrounds.dtype != torch.float32
        or not backgrounds.is_contiguous()
        or backgrounds.shape != (means2d.shape[0], features.shape[-1])
    ):
        return None
    backend = load_supa_backend()
    if backend is None or not hasattr(backend, "rasterize_to_pixels_tiled_fwd"):
        return None

    tile_offsets_supa = tile_offsets.detach().to(means2d.device).contiguous()
    flatten_ids_supa = flatten_ids.detach().to(means2d.device).contiguous()
    out = backend.rasterize_to_pixels_tiled_fwd(
        means2d,
        conics,
        features,
        opacities,
        backgrounds,
        int(width),
        int(height),
        int(tile_size),
        tile_offsets_supa,
        flatten_ids_supa,
    )
    if synchronize_launch:
        synchronize(means2d.device)
    return out


def rasterize_to_pixels_tiled_bwd(
    means2d: Tensor,
    conics: Tensor,
    features: Tensor,
    opacities: Tensor,
    width: int,
    height: int,
    tile_size: int,
    tile_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    v_render: Optional[Tensor] = None,
    v_alpha: Optional[Tensor] = None,
    forward_alpha: Optional[Tensor] = None,
    synchronize_launch: bool = True,
    backward_mode: str = "auto",
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]]:
    """Run the native SUPA tiled rasterization backward prototype."""
    if v_render is None:
        v_render = torch.zeros(
            (means2d.shape[0], height, width, features.shape[-1]),
            dtype=features.dtype,
            device=features.device,
        )
    else:
        v_render = _materialize_projection_grad(v_render)
    if v_alpha is None:
        v_alpha_arg: Optional[Tensor] = None
    else:
        v_alpha_arg = _materialize_projection_grad(v_alpha)
    if forward_alpha is not None:
        forward_alpha = _materialize_projection_saved(forward_alpha)

    if (
        not is_supa_device(means2d)
        or not is_supa_device(v_render)
        or (v_alpha_arg is not None and not is_supa_device(v_alpha_arg))
        or (forward_alpha is not None and not is_supa_device(forward_alpha))
        or means2d.dtype != torch.float32
        or conics.dtype != torch.float32
        or features.dtype != torch.float32
        or opacities.dtype != torch.float32
        or tile_offsets.dtype != torch.int32
        or flatten_ids.dtype != torch.int32
        or v_render.dtype != torch.float32
        or (v_alpha_arg is not None and v_alpha_arg.dtype != torch.float32)
        or (forward_alpha is not None and forward_alpha.dtype != torch.float32)
        or not means2d.is_contiguous()
        or not conics.is_contiguous()
        or not features.is_contiguous()
        or not opacities.is_contiguous()
        or not v_render.is_contiguous()
        or (v_alpha_arg is not None and not v_alpha_arg.is_contiguous())
        or (forward_alpha is not None and not forward_alpha.is_contiguous())
        or means2d.dim() != 3
        or means2d.shape[-1] != 2
        or conics.shape != means2d.shape[:-1] + (3,)
        or features.dim() != 3
        or features.shape[:2] != means2d.shape[:2]
        or features.shape[-1] > 64
        or opacities.shape != means2d.shape[:2]
        or tile_offsets.dim() != 3
        or tile_offsets.shape[0] != means2d.shape[0]
        or flatten_ids.dim() != 1
        or v_render.shape != (means2d.shape[0], height, width, features.shape[-1])
        or (
            v_alpha_arg is not None
            and v_alpha_arg.shape != (means2d.shape[0], height, width, 1)
        )
        or (
            forward_alpha is not None
            and forward_alpha.shape != (means2d.shape[0], height, width, 1)
        )
        or width <= 0
        or height <= 0
        or tile_size <= 0
    ):
        return None
    if backgrounds is not None and (
        backgrounds.dtype != torch.float32
        or not backgrounds.is_contiguous()
        or backgrounds.shape != (means2d.shape[0], features.shape[-1])
    ):
        return None
    backend = load_supa_backend()
    if backend is None or not hasattr(backend, "rasterize_to_pixels_tiled_bwd"):
        return None

    tile_offsets_supa = tile_offsets.detach().to(means2d.device).contiguous()
    flatten_ids_supa = flatten_ids.detach().to(means2d.device).contiguous()
    max_tile_range = _max_tiled_range(tile_offsets_supa, int(flatten_ids_supa.numel()))
    if max_tile_range is None:
        return None
    mode = _normalize_tiled_backward_mode(backward_mode)
    tile_reduce_supported = (
        features.shape[-1] == 3
        and int(tile_size) == 16
        and hasattr(backend, "rasterize_to_pixels_tiled_bwd_tile_reduce")
    )
    tile_reduce_range_ok = max_tile_range <= _TILED_BWD_AUTO_TILE_REDUCE_MAX_RANGE
    # Auto remains conservative for performance. Explicit "parallel" requests
    # the stable prefix-checkpoint tile-reduce kernel for all supported ranges.
    use_tile_reduce = tile_reduce_supported and (
        mode in {"tile_reduce", "parallel"}
        or (mode == "auto" and tile_reduce_range_ok)
    )
    if use_tile_reduce:
        out = backend.rasterize_to_pixels_tiled_bwd_tile_reduce(
            means2d,
            conics,
            features,
            opacities,
            backgrounds,
            v_render,
            v_alpha_arg,
            forward_alpha,
            int(width),
            int(height),
            int(tile_size),
            tile_offsets_supa,
            flatten_ids_supa,
            int(max_tile_range),
        )
    elif mode == "tile_reduce":
        return None
    else:
        out = backend.rasterize_to_pixels_tiled_bwd(
            means2d,
            conics,
            features,
            opacities,
            backgrounds,
            v_render,
            v_alpha_arg,
            forward_alpha,
            int(width),
            int(height),
            int(tile_size),
            tile_offsets_supa,
            flatten_ids_supa,
            int(max_tile_range),
        )
    if synchronize_launch:
        synchronize(means2d.device)
    grad_backgrounds = out[4] if backgrounds is not None else None
    return out[0], out[1], out[2], out[3], grad_backgrounds


def rasterize_to_pixels_tiled(
    means2d: Tensor,
    conics: Tensor,
    features: Tensor,
    opacities: Tensor,
    width: int,
    height: int,
    tile_size: int,
    tile_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    backward_mode: str = "auto",
) -> Optional[Tuple[Tensor, Tensor]]:
    """Run native SUPA tiled rasterization with autograd when supported."""
    if (
        not is_supa_device(means2d)
        or means2d.dtype != torch.float32
        or conics.dtype != torch.float32
        or features.dtype != torch.float32
        or opacities.dtype != torch.float32
        or tile_offsets.dtype != torch.int32
        or flatten_ids.dtype != torch.int32
        or not means2d.is_contiguous()
        or not conics.is_contiguous()
        or not features.is_contiguous()
        or not opacities.is_contiguous()
        or means2d.dim() != 3
        or means2d.shape[-1] != 2
        or conics.shape != means2d.shape[:-1] + (3,)
        or features.dim() != 3
        or features.shape[:2] != means2d.shape[:2]
        or features.shape[-1] > 64
        or opacities.shape != means2d.shape[:2]
        or tile_offsets.dim() != 3
        or tile_offsets.shape[0] != means2d.shape[0]
        or flatten_ids.dim() != 1
        or width <= 0
        or height <= 0
        or tile_size <= 0
    ):
        return None
    if backgrounds is not None and (
        backgrounds.dtype != torch.float32
        or not backgrounds.is_contiguous()
        or backgrounds.shape != (means2d.shape[0], features.shape[-1])
    ):
        return None
    backend = load_supa_backend()
    if backend is None or not hasattr(backend, "rasterize_to_pixels_tiled_fwd"):
        return None
    if not hasattr(backend, "rasterize_to_pixels_tiled_bwd"):
        return None
    return _RasterizeToPixelsTiled.apply(
        means2d,
        conics,
        features,
        opacities,
        backgrounds,
        tile_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
        _normalize_tiled_backward_mode(backward_mode),
    )


def alpha_composite_fwd(
    means2d: Tensor,
    conics: Tensor,
    features: Tensor,
    opacities: Tensor,
    depths: Tensor,
    radii: Tensor,
    width: int,
    height: int,
    backgrounds: Optional[Tensor] = None,
    synchronize_launch: bool = True,
) -> Optional[Tuple[Tensor, Tensor]]:
    """Run the native SUPA forward-only alpha compositing prototype when supported."""
    if (
        not is_supa_device(means2d)
        or means2d.dtype != torch.float32
        or conics.dtype != torch.float32
        or features.dtype != torch.float32
        or opacities.dtype != torch.float32
        or depths.dtype != torch.float32
        or radii.dtype != torch.int32
        or not means2d.is_contiguous()
        or not conics.is_contiguous()
        or not features.is_contiguous()
        or not opacities.is_contiguous()
        or not depths.is_contiguous()
        or not radii.is_contiguous()
    ):
        return None
    if (
        means2d.dim() != 3
        or means2d.shape[-1] != 2
        or conics.shape != means2d.shape[:-1] + (3,)
        or features.dim() != 3
        or features.shape[:2] != means2d.shape[:2]
        or opacities.shape != means2d.shape[:2]
        or depths.shape != means2d.shape[:2]
        or radii.shape != means2d.shape
    ):
        return None
    if backgrounds is not None and (
        backgrounds.dtype != torch.float32
        or not backgrounds.is_contiguous()
        or backgrounds.shape != (means2d.shape[0], features.shape[-1])
    ):
        return None
    backend = load_supa_backend()
    if backend is None:
        return None

    sorted_ids = _make_composite_sorted_ids(depths, radii)
    out = backend.alpha_composite_fwd(
        means2d,
        conics,
        features,
        opacities,
        depths,
        radii,
        sorted_ids,
        backgrounds,
        width,
        height,
    )
    if synchronize_launch:
        synchronize(means2d.device)
    return out


def alpha_composite(
    means2d: Tensor,
    conics: Tensor,
    features: Tensor,
    opacities: Tensor,
    depths: Tensor,
    radii: Tensor,
    width: int,
    height: int,
    backgrounds: Optional[Tensor] = None,
) -> Optional[Tuple[Tensor, Tensor]]:
    """Run native SUPA compositing forward with a PyTorch recompute backward."""
    if (
        not is_supa_device(means2d)
        or means2d.dtype != torch.float32
        or conics.dtype != torch.float32
        or features.dtype != torch.float32
        or opacities.dtype != torch.float32
        or depths.dtype != torch.float32
        or radii.dtype != torch.int32
        or not means2d.is_contiguous()
        or not conics.is_contiguous()
        or not features.is_contiguous()
        or not opacities.is_contiguous()
        or not depths.is_contiguous()
        or not radii.is_contiguous()
    ):
        return None
    if (
        means2d.dim() != 3
        or means2d.shape[-1] != 2
        or conics.shape != means2d.shape[:-1] + (3,)
        or features.dim() != 3
        or features.shape[:2] != means2d.shape[:2]
        or opacities.shape != means2d.shape[:2]
        or depths.shape != means2d.shape[:2]
        or radii.shape != means2d.shape
    ):
        return None
    if backgrounds is not None and (
        backgrounds.dtype != torch.float32
        or not backgrounds.is_contiguous()
        or backgrounds.shape != (means2d.shape[0], features.shape[-1])
    ):
        return None
    backend = load_supa_backend()
    if backend is None:
        return None

    out = _AlphaComposite.apply(
        means2d,
        conics,
        features,
        opacities,
        depths,
        radii,
        backgrounds,
        width,
        height,
    )
    synchronize(means2d.device)
    return out


def alpha_composite_bwd(
    means2d: Tensor,
    conics: Tensor,
    features: Tensor,
    opacities: Tensor,
    depths: Tensor,
    radii: Tensor,
    sorted_ids: Optional[Tensor],
    width: int,
    height: int,
    backgrounds: Optional[Tensor],
    v_render: Optional[Tensor],
    v_alpha: Optional[Tensor],
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]]:
    """Run the native SUPA alpha compositing backward prototype when supported."""
    if v_render is None:
        v_render = torch.zeros(
            (means2d.shape[0], height, width, features.shape[-1]),
            dtype=features.dtype,
            device=features.device,
        )
    else:
        v_render = _materialize_projection_grad(v_render)
    if v_alpha is None:
        v_alpha = torch.zeros(
            (means2d.shape[0], height, width, 1),
            dtype=features.dtype,
            device=features.device,
        )
    else:
        v_alpha = _materialize_projection_grad(v_alpha)

    if (
        not is_supa_device(means2d)
        or means2d.dtype != torch.float32
        or conics.dtype != torch.float32
        or features.dtype != torch.float32
        or opacities.dtype != torch.float32
        or depths.dtype != torch.float32
        or radii.dtype != torch.int32
        or (sorted_ids is not None and sorted_ids.dtype != torch.int32)
        or v_render.dtype != torch.float32
        or v_alpha.dtype != torch.float32
        or not means2d.is_contiguous()
        or not conics.is_contiguous()
        or not features.is_contiguous()
        or not opacities.is_contiguous()
        or not depths.is_contiguous()
        or not radii.is_contiguous()
        or (sorted_ids is not None and not sorted_ids.is_contiguous())
        or not v_render.is_contiguous()
        or not v_alpha.is_contiguous()
    ):
        return None
    if (
        means2d.dim() != 3
        or means2d.shape[-1] != 2
        or conics.shape != means2d.shape[:-1] + (3,)
        or features.dim() != 3
        or features.shape[:2] != means2d.shape[:2]
        or features.shape[-1] > 64
        or opacities.shape != means2d.shape[:2]
        or depths.shape != means2d.shape[:2]
        or radii.shape != means2d.shape
        or (sorted_ids is not None and sorted_ids.shape != means2d.shape[:2])
        or v_render.shape != (means2d.shape[0], height, width, features.shape[-1])
        or v_alpha.shape != (means2d.shape[0], height, width, 1)
    ):
        return None
    if backgrounds is not None and (
        backgrounds.dtype != torch.float32
        or not backgrounds.is_contiguous()
        or backgrounds.shape != (means2d.shape[0], features.shape[-1])
    ):
        return None

    backend = load_supa_backend()
    if backend is None or not hasattr(backend, "alpha_composite_bwd"):
        return None

    out = backend.alpha_composite_bwd(
        means2d,
        conics,
        features,
        opacities,
        depths,
        radii,
        sorted_ids,
        backgrounds,
        v_render,
        v_alpha,
        width,
        height,
    )
    synchronize(means2d.device)
    grad_backgrounds = out[5] if backgrounds is not None else None
    return out[0], out[1], out[2], out[3], out[4], grad_backgrounds


def projection_ewa_3dgs_pinhole_fwd(
    means: Tensor,
    covars: Optional[Tensor],
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    *,
    quats: Optional[Tensor] = None,
    scales: Optional[Tensor] = None,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]]:
    """Backward-compatible alias for the native SUPA pinhole projection prototype."""
    return projection_ewa_3dgs_pinhole(
        means,
        covars,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        calc_compensations,
        quats=quats,
        scales=scales,
    )
