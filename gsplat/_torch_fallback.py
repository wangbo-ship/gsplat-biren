# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small-scene PyTorch rasterization fallback used while native SUPA kernels land."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from ._constants import MAX_ALPHA
from ._math import _quat_scale_to_covar_preci
from ._torch_impl import _fully_fused_projection, _spherical_harmonics


_SUPA_TILED_AUTO_MIN_GAUSSIANS = 512
_SUPA_TILED_AUTO_MIN_PIXELS = 128 * 96


def _native_options(options: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if options is None:
        return None
    normalized = {
        "native": bool(options.get("native", False)),
        "native_tiled": str(options.get("native_tiled", "false")).strip().lower(),
        "native_tiled_metadata": bool(options.get("native_tiled_metadata", False)),
        "native_tiled_train": bool(options.get("native_tiled_train", False)),
        "native_tiled_backward_mode": _normalize_native_tiled_backward_mode(
            options.get("native_tiled_backward_mode", "auto")
        ),
        "native_alpha_threshold": float(options.get("native_alpha_threshold", 0.0)),
    }
    for feature in (
        "compact",
        "composite",
        "covariance",
        "projection",
        "sh",
        "tile_count",
    ):
        key = f"native_{feature}"
        if key in options:
            normalized[key] = bool(options[key])
    return normalized


def _normalize_native_tiled_backward_mode(value: object) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {"auto", "tile_reduce", "pixel", "parallel"}:
        return mode
    return "auto"


def _native_enabled(feature: str, options: Optional[Dict[str, object]] = None) -> bool:
    if options is not None:
        key = f"native_{feature.lower()}"
        if key in options:
            return bool(options[key])
        return bool(options.get("native", False))
    return False


def _native_tiled_mode(options: Optional[Dict[str, object]] = None) -> str:
    if options is not None:
        value = str(options.get("native_tiled", "false")).strip().lower()
        if value == "true":
            return "force"
        if value == "auto":
            return "auto"
        return "off"
    return "off"


def _native_tiled_train_enabled(options: Optional[Dict[str, object]] = None) -> bool:
    if options is not None:
        return bool(options.get("native_tiled_train", False))
    return False


def _native_tiled_metadata_enabled(options: Optional[Dict[str, object]] = None) -> bool:
    if options is not None:
        return bool(options.get("native_tiled_metadata", False))
    return False


def _native_alpha_threshold(options: Optional[Dict[str, object]] = None) -> float:
    if options is None:
        return 0.0
    return max(float(options.get("native_alpha_threshold", 0.0)), 0.0)


def _native_tiled_backward_mode(options: Optional[Dict[str, object]] = None) -> str:
    if options is None:
        return "auto"
    return _normalize_native_tiled_backward_mode(
        options.get("native_tiled_backward_mode", "auto")
    )


def _native_tiled_enabled(
    *,
    forward_only: bool,
    n_gaussians: int,
    width: int,
    height: int,
    options: Optional[Dict[str, object]] = None,
) -> bool:
    mode = _native_tiled_mode(options)
    if not forward_only:
        return False
    if mode == "force":
        return True
    if mode != "auto":
        return False
    return (
        n_gaussians >= _SUPA_TILED_AUTO_MIN_GAUSSIANS
        and (width * height) >= _SUPA_TILED_AUTO_MIN_PIXELS
    )


def _compute_covar(
    quats: Tensor,
    scales: Tensor,
    native_options: Optional[Dict[str, object]] = None,
) -> Tuple[Tensor, str]:
    if quats.device.type == "supa" and _native_enabled("covariance", native_options):
        try:
            from .supa._wrapper import quat_scale_to_covar_preci

            out = quat_scale_to_covar_preci(
                quats.contiguous(),
                scales.contiguous(),
                compute_covar=True,
                compute_preci=False,
                triu=False,
            )
            if out is not None and out[0] is not None:
                return out[0], "supa_native"
        except Exception as exc:  # pragma: no cover - environment-dependent fallback
            import warnings

            warnings.warn(
                f"SUPA native covariance kernel failed, using PyTorch fallback: {exc}",
                stacklevel=2,
            )
    covars, _ = _quat_scale_to_covar_preci(quats, scales, True, False, triu=False)
    return covars, "torch"


def _try_native_spherical_harmonics(
    sh_degree: Optional[int],
    dirs: Tensor,
    coeffs: Tensor,
    native_options: Optional[Dict[str, object]] = None,
) -> Optional[Tensor]:
    if (
        sh_degree is None
        or dirs.device.type != "supa"
        or not _native_enabled("sh", native_options)
        or sh_degree < 0
        or sh_degree > 4
        or dirs.dtype != torch.float32
        or coeffs.dtype != torch.float32
        or coeffs.shape[-1] != 3
    ):
        return None
    try:
        from .supa._wrapper import spherical_harmonics

        native_dirs = _materialize_native_arg(dirs)
        native_coeffs = _materialize_native_arg(coeffs)
        if native_dirs is None or native_coeffs is None:
            return None
        return spherical_harmonics(
            int(sh_degree),
            native_dirs.contiguous(),
            native_coeffs.contiguous(),
            synchronize_launch=not torch.is_grad_enabled(),
        )
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        import warnings

        warnings.warn(
            f"SUPA native spherical harmonics kernel failed, using PyTorch fallback: {exc}",
            stacklevel=2,
        )
        return None


def _expand_triu_covars(covars: Tensor) -> Tensor:
    full = torch.zeros(covars.shape[:-1] + (3, 3), device=covars.device, dtype=covars.dtype)
    tri = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
    full[..., tri[0], tri[1]] = covars
    full[..., tri[1], tri[0]] = covars
    return full


def _materialize_native_arg(tensor: Optional[Tensor]) -> Optional[Tensor]:
    if tensor is not None and tensor.device.type == "supa":
        # Some BRPyTorch tensors produced by SUPA ops report as contiguous while
        # exposing stale or partial data to native pointer kernels.
        return tensor.cpu().to(tensor.device).contiguous()
    return tensor


def _try_native_projection(
    *,
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
    camera_model: str,
    synchronize_launch: bool = True,
    native_options: Optional[Dict[str, object]] = None,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]]:
    if (
        means.device.type != "supa"
        or camera_model != "pinhole"
        or not _native_enabled("projection", native_options)
        or Ks.requires_grad
    ):
        return None
    if covars is None and (quats is None or scales is None):
        return None

    try:
        from .supa._wrapper import projection_ewa_3dgs_pinhole

        native_means = _materialize_native_arg(means)
        native_covars = _materialize_native_arg(covars)
        native_quats = _materialize_native_arg(quats if covars is None else None)
        native_scales = _materialize_native_arg(scales if covars is None else None)
        native_viewmats = _materialize_native_arg(viewmats)
        native_Ks = _materialize_native_arg(Ks)

        return projection_ewa_3dgs_pinhole(
            native_means.contiguous(),
            native_covars.contiguous() if native_covars is not None else None,
            native_viewmats.contiguous(),
            native_Ks.contiguous(),
            width,
            height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            calc_compensations=calc_compensations,
            quats=native_quats.contiguous() if native_quats is not None else None,
            scales=native_scales.contiguous() if native_scales is not None else None,
            synchronize_launch=synchronize_launch,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        import warnings

        warnings.warn(
            f"SUPA native projection kernel failed, using PyTorch fallback: {exc}",
            stacklevel=2,
        )
        return None


def _evaluate_features(
    features: Tensor,
    sh_degree: Optional[int],
    means: Tensor,
    radii: Tensor,
    viewmats: Tensor,
    batch_dims: tuple,
    C: int,
    N: int,
    clamp_after_sh: bool,
    native_options: Optional[Dict[str, object]] = None,
) -> Tuple[Tensor, Optional[str]]:
    """Normalize color or auxiliary feature layout to [..., C, N, D]."""
    num_batch_dims = len(batch_dims)
    if sh_degree is None:
        if features.dim() == num_batch_dims + 2:
            return torch.broadcast_to(
                features[..., None, :, :], batch_dims + (C, N, features.shape[-1])
            ), None
        return features, None

    R = viewmats[..., :3, :3]
    t = viewmats[..., :3, 3]
    campos = -(R.mT @ t.unsqueeze(-1)).squeeze(-1)
    dirs = means[..., None, :, :] - campos[..., None, :]
    if features.dim() == num_batch_dims + 3:
        coeffs = torch.broadcast_to(
            features[..., None, :, :, :], batch_dims + (C, N, -1, features.shape[-1])
        )
    else:
        coeffs = features
    native = _try_native_spherical_harmonics(sh_degree, dirs, coeffs, native_options)
    if native is not None:
        evaluated = native
        backend = "supa_native"
    else:
        evaluated = _spherical_harmonics(sh_degree, dirs, coeffs)
        backend = "torch"
    evaluated = evaluated + 0.5
    if clamp_after_sh:
        evaluated = torch.clamp_min(evaluated, 0.0)
    return torch.where((radii > 0).all(dim=-1)[..., None], evaluated, 0.0), backend


def _tiles_per_gauss(
    means2d: Tensor,
    radii: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> Tensor:
    """Estimate tile intersections for compatibility metadata."""
    tile_means2d = means2d / tile_size
    tile_radii = radii / tile_size
    tile_mins = torch.floor(tile_means2d - tile_radii).int()
    tile_maxs = torch.ceil(tile_means2d + tile_radii).int()
    tile_mins[..., 0] = torch.clamp(tile_mins[..., 0], 0, tile_width)
    tile_mins[..., 1] = torch.clamp(tile_mins[..., 1], 0, tile_height)
    tile_maxs[..., 0] = torch.clamp(tile_maxs[..., 0], 0, tile_width)
    tile_maxs[..., 1] = torch.clamp(tile_maxs[..., 1], 0, tile_height)
    tiles = (tile_maxs - tile_mins).prod(dim=-1)
    return (tiles * (radii > 0).all(dim=-1)).int()


def _visible_gaussian_ids(radii: Tensor, depths: Tensor, n_gaussians: int) -> Tuple[Tensor, int]:
    device = depths.device
    valid = (radii > 0).all(dim=-1) & torch.isfinite(depths)
    if n_gaussians == 0:
        ids = torch.empty((0,), dtype=torch.long, device=device)
        return ids, 0
    visible_cpu = valid.reshape(-1, n_gaussians).any(dim=0).detach().cpu()
    visible_count = int(visible_cpu.sum().item())
    ids = torch.nonzero(visible_cpu, as_tuple=False).flatten()
    return ids.to(device=device, dtype=torch.long), visible_count


def _try_native_visible_gaussian_ids(
    radii: Tensor,
    depths: Tensor,
    n_gaussians: int,
    native_options: Optional[Dict[str, object]] = None,
) -> Optional[Tuple[Tensor, int]]:
    if (
        depths.device.type != "supa"
        or not _native_enabled("compact", native_options)
        or n_gaussians == 0
    ):
        return None
    try:
        from .supa._wrapper import visible_compact_ids

        native_radii = (
            _materialize_native_arg(radii).reshape(-1, n_gaussians, 2).contiguous()
        )
        native_depths = (
            _materialize_native_arg(depths).reshape(-1, n_gaussians).contiguous()
        )
        out = visible_compact_ids(native_radii, native_depths)
        if out is None:
            return None
        return out
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        import warnings

        warnings.warn(
            f"SUPA native visible compact kernel failed, using PyTorch fallback: {exc}",
            stacklevel=2,
        )
        return None


class _SupaCompactSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: Tensor, indices: Tensor, dim: int) -> Tensor:
        dim = int(dim)
        if dim < 0:
            dim += tensor.dim()
        ctx.dim = dim
        ctx.input_shape = tuple(tensor.shape)
        long_indices = indices.detach().to(torch.long).contiguous()
        ctx.save_for_backward(long_indices)
        return tensor.index_select(dim, long_indices)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (indices,) = ctx.saved_tensors
        grad_input = torch.zeros(
            ctx.input_shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        if indices.numel() > 0:
            # SUPA index_select backward uses index_add_, which can fail on
            # large compacted tensors. compact_ids are unique, so index_copy_
            # restores the dense gradient without that problematic accumulation.
            grad_input.index_copy_(ctx.dim, indices, grad_output.contiguous())
        return grad_input, None, None


def _compact_select(
    tensor: Tensor,
    indices: Tensor,
    dim: int,
    native_options: Optional[Dict[str, object]] = None,
) -> Tensor:
    if tensor.device.type == "supa" and tensor.requires_grad:
        if _native_enabled("compact", native_options) and tensor.dtype == torch.float32:
            try:
                from .supa._wrapper import compact_select

                native_tensor = _materialize_native_arg(tensor).contiguous()
                native_indices = indices.to(
                    device=tensor.device, dtype=torch.int32
                ).contiguous()
                out = compact_select(native_tensor, native_indices, dim)
                if out is not None:
                    return out
            except Exception as exc:  # pragma: no cover - environment-dependent fallback
                import warnings

                warnings.warn(
                    f"SUPA native compact select failed, using PyTorch fallback: {exc}",
                    stacklevel=2,
                )
        return _SupaCompactSelect.apply(tensor, indices, dim)
    return tensor.index_select(dim, indices.to(dtype=torch.long))


def _expand_compacted_tiles(
    tiles_per_gauss: Tensor,
    compact_ids: Optional[Tensor],
    n_gaussians: int,
) -> Tensor:
    if compact_ids is None:
        return tiles_per_gauss
    expanded = tiles_per_gauss.new_zeros(tiles_per_gauss.shape[:-1] + (n_gaussians,))
    if compact_ids.numel() > 0:
        expanded.index_copy_(
            -1,
            compact_ids.to(device=tiles_per_gauss.device, dtype=torch.long),
            tiles_per_gauss,
        )
    return expanded


def _try_native_tiles_per_gauss(
    means2d: Tensor,
    radii: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    projection_backend: str,
    synchronize_launch: bool = True,
    native_options: Optional[Dict[str, object]] = None,
) -> Optional[Tensor]:
    if (
        projection_backend != "supa_native"
        or means2d.device.type != "supa"
        or not _native_enabled("tile_count", native_options)
        or tile_size <= 0
    ):
        return None

    try:
        from .supa._wrapper import tile_count_unpacked

        N = means2d.shape[-2]
        native_means2d = _materialize_native_arg(means2d)
        native_radii = _materialize_native_arg(radii)
        tiles = tile_count_unpacked(
            native_means2d.reshape(-1, N, 2).contiguous(),
            native_radii.reshape(-1, N, 2).contiguous(),
            tile_size,
            tile_width,
            tile_height,
            synchronize_launch=synchronize_launch,
        )
        if tiles is not None:
            return tiles.reshape(means2d.shape[:-1])
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        import warnings

        warnings.warn(
            f"SUPA native tile-count metadata failed, using PyTorch fallback: {exc}",
            stacklevel=2,
        )
    return None


def _alpha_composite(
    means2d: Tensor,
    conics: Tensor,
    features: Tensor,
    opacities: Tensor,
    depths: Tensor,
    radii: Tensor,
    width: int,
    height: int,
    backgrounds: Optional[Tensor],
    chunk_size: int,
) -> Tuple[Tensor, Tensor]:
    """Naively alpha-composite sorted Gaussians with PyTorch tensor ops."""
    image_dims = means2d.shape[:-2]
    I = math.prod(image_dims)
    N = means2d.shape[-2]
    channels = features.shape[-1]
    device = means2d.device
    dtype = means2d.dtype

    means2d = means2d.reshape(I, N, 2)
    conics = conics.reshape(I, N, 3)
    features = features.reshape(I, N, channels)
    opacities = opacities.reshape(I, N)
    depths = depths.reshape(I, N)
    radii = radii.reshape(I, N, 2)
    if backgrounds is not None:
        backgrounds = backgrounds.reshape(I, channels)

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack((xs, ys), dim=-1)

    render_images = []
    alpha_images = []
    for image_id in range(I):
        valid = (radii[image_id] > 0).all(dim=-1) & torch.isfinite(depths[image_id])
        sorted_ids = torch.argsort(depths[image_id])
        sorted_ids = sorted_ids[valid[sorted_ids]]

        render = torch.zeros((height, width, channels), device=device, dtype=dtype)
        alpha_acc = torch.zeros((height, width, 1), device=device, dtype=dtype)
        transmittance = torch.ones((height, width), device=device, dtype=dtype)

        for start in range(0, sorted_ids.numel(), chunk_size):
            ids = sorted_ids[start : start + chunk_size]
            delta = pixels.unsqueeze(0) - means2d[image_id, ids, None, None, :]
            c = conics[image_id, ids]
            sigma = (
                0.5
                * (
                    c[:, None, None, 0] * delta[..., 0].square()
                    + c[:, None, None, 2] * delta[..., 1].square()
                )
                + c[:, None, None, 1] * delta[..., 0] * delta[..., 1]
            )
            alpha = opacities[image_id, ids, None, None] * torch.exp(-sigma)
            alpha = torch.clamp(alpha, min=0.0, max=MAX_ALPHA)
            feature_chunk = features[image_id, ids]
            for local_id in range(ids.numel()):
                weight = transmittance * alpha[local_id]
                render = render + weight[..., None] * feature_chunk[local_id]
                alpha_acc = alpha_acc + weight[..., None]
                transmittance = transmittance * (1.0 - alpha[local_id])

        if backgrounds is not None:
            render = render + transmittance[..., None] * backgrounds[image_id]
        render_images.append(render)
        alpha_images.append(alpha_acc)

    render_images = torch.stack(render_images, dim=0).reshape(
        image_dims + (height, width, channels)
    )
    alpha_images = torch.stack(alpha_images, dim=0).reshape(
        image_dims + (height, width, 1)
    )
    return render_images, alpha_images


def _try_native_alpha_composite(
    means2d: Tensor,
    conics: Tensor,
    features: Tensor,
    opacities: Tensor,
    depths: Tensor,
    radii: Tensor,
    width: int,
    height: int,
    tile_size: int,
    backgrounds: Optional[Tensor],
    trust_forward_inputs: bool = False,
    materialize_forward_features: bool = False,
    native_options: Optional[Dict[str, object]] = None,
) -> Optional[Tuple[Tensor, Tensor, str, Optional[Tensor], Optional[str]]]:
    if means2d.device.type != "supa" or not _native_enabled("composite", native_options):
        return None
    forward_only = (not torch.is_grad_enabled()) or (
        not any(tensor.requires_grad for tensor in (means2d, conics, features, opacities))
        and (backgrounds is None or not backgrounds.requires_grad)
    )
    if not forward_only and features.shape[-1] > 64:
        return None

    image_dims = means2d.shape[:-2]
    I = math.prod(image_dims)
    N = means2d.shape[-2]
    channels = features.shape[-1]
    flat_backgrounds = None
    if backgrounds is not None:
        flat_backgrounds = backgrounds.reshape(I, channels).contiguous()

    try:
        from .supa._wrapper import alpha_composite, alpha_composite_fwd

        if forward_only and trust_forward_inputs:
            native_means2d = means2d
            native_conics = conics
            # No-grad trainer/eval paths often pass features/opacities produced
            # by BRPyTorch elementwise ops such as sigmoid(). Those tensors can
            # expose stale data to native pointer kernels unless materialized.
            native_features = _materialize_native_arg(features)
            native_opacities = _materialize_native_arg(opacities)
            native_depths = depths
            native_radii = radii
            native_backgrounds = flat_backgrounds
        elif forward_only and materialize_forward_features:
            native_means2d = means2d
            native_conics = conics
            native_features = _materialize_native_arg(features)
            native_opacities = _materialize_native_arg(opacities)
            native_depths = depths
            native_radii = radii
            native_backgrounds = flat_backgrounds
        else:
            native_means2d = _materialize_native_arg(means2d)
            native_conics = _materialize_native_arg(conics)
            native_features = _materialize_native_arg(features)
            native_opacities = _materialize_native_arg(opacities)
            native_depths = _materialize_native_arg(depths)
            native_radii = _materialize_native_arg(radii)
            native_backgrounds = _materialize_native_arg(flat_backgrounds)

        flat_means2d = native_means2d.reshape(I, N, 2).contiguous()
        flat_conics = native_conics.reshape(I, N, 3).contiguous()
        flat_features = native_features.reshape(I, N, channels).contiguous()
        flat_opacities = native_opacities.reshape(I, N).contiguous()
        flat_depths = native_depths.reshape(I, N).contiguous()
        flat_radii = native_radii.reshape(I, N, 2).contiguous()

        tiled_training = (not forward_only) and _native_tiled_train_enabled(native_options)
        alpha_threshold = _native_alpha_threshold(native_options)
        if tile_size > 0 and (
            tiled_training
            or _native_tiled_enabled(
                forward_only=forward_only,
                n_gaussians=N,
                width=width,
                height=height,
                options=native_options,
            )
        ):
            try:
                from .supa._wrapper import (
                    isect_tiles_unpacked_offsets,
                    rasterize_to_pixels_tiled,
                    rasterize_to_pixels_tiled_fwd,
                )

                tile_width = math.ceil(width / float(tile_size))
                tile_height = math.ceil(height / float(tile_size))
                isects = isect_tiles_unpacked_offsets(
                    flat_means2d,
                    flat_radii,
                    flat_depths,
                    tile_size,
                    tile_width,
                    tile_height,
                    synchronize_launch=False,
                    return_backend=True,
                    use_native_offsets=_native_tiled_metadata_enabled(native_options),
                    conics=flat_conics,
                    opacities=flat_opacities,
                    alpha_threshold=alpha_threshold,
                )
                if isects is not None:
                    tiles_per_gauss, tile_offsets, flatten_ids, metadata_backend = isects
                    if tiled_training:
                        tiled = rasterize_to_pixels_tiled(
                            flat_means2d,
                            flat_conics,
                            flat_features,
                            flat_opacities,
                            width,
                            height,
                            tile_size,
                            tile_offsets,
                            flatten_ids,
                            native_backgrounds,
                            backward_mode=_native_tiled_backward_mode(native_options),
                        )
                    else:
                        tiled = rasterize_to_pixels_tiled_fwd(
                            flat_means2d,
                            flat_conics,
                            flat_features,
                            flat_opacities,
                            width,
                            height,
                            tile_size,
                            tile_offsets,
                            flatten_ids,
                            native_backgrounds,
                            synchronize_launch=False,
                        )
                    if tiled is not None:
                        return tiled[0], tiled[1], "tiled", tiles_per_gauss.reshape(
                            means2d.shape[:-1]
                        ), metadata_backend
            except Exception:
                pass

        variant = "full_scan"
        if forward_only:
            composite = alpha_composite_fwd(
                flat_means2d,
                flat_conics,
                flat_features,
                flat_opacities,
                flat_depths,
                flat_radii,
                width,
                height,
                native_backgrounds,
                synchronize_launch=False,
            )
        else:
            composite = alpha_composite(
                flat_means2d,
                flat_conics,
                flat_features,
                flat_opacities,
                flat_depths,
                flat_radii,
                width,
                height,
                native_backgrounds,
            )
        if composite is None:
            return None
        return composite[0], composite[1], variant, None, None
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        import warnings

        warnings.warn(
            f"SUPA native alpha compositing kernel failed, using PyTorch fallback: {exc}",
            stacklevel=2,
        )
        return None


def rasterization(
    *,
    means: Tensor,
    quats: Optional[Tensor],
    scales: Optional[Tensor],
    opacities: Tensor,
    colors: Optional[Tensor],
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    eps2d: float,
    sh_degree: Optional[int],
    tile_size: int,
    backgrounds: Optional[Tensor],
    render_mode: str,
    rasterize_mode: str,
    camera_model: str,
    covars: Optional[Tensor],
    extra_signals: Optional[Tensor],
    extra_signals_sh_degree: Optional[int],
    chunk_size: int = 32,
    supa_native_options: Optional[Dict[str, object]] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize basic 3DGS inputs with PyTorch ops and optional native SUPA projection."""
    native_options = _native_options(supa_native_options)
    if camera_model not in ("pinhole", "ortho", "fisheye"):
        raise NotImplementedError(
            f"The PyTorch fallback does not support camera_model={camera_model!r}."
        )
    projection_covars = covars
    covariance_backend = "provided" if covars is not None else None
    if covars is None:
        if quats is None or scales is None:
            raise ValueError("quats and scales are required when covars is not provided.")
        if not (
            means.device.type == "supa"
            and camera_model == "pinhole"
            and _native_enabled("projection", native_options)
        ):
            projection_covars, covariance_backend = _compute_covar(
                quats,
                scales,
                native_options,
            )
    elif covars.shape[-1] == 6:
        projection_covars = _expand_triu_covars(covars)

    batch_dims = means.shape[:-2]
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    I = B * C

    calc_compensations = rasterize_mode == "antialiased"
    proj_results = _try_native_projection(
        means=means,
        covars=projection_covars,
        quats=quats,
        scales=scales,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        calc_compensations=calc_compensations,
        camera_model=camera_model,
        synchronize_launch=torch.is_grad_enabled(),
        native_options=native_options,
    )
    projection_backend = "supa_native" if proj_results is not None else "torch"
    if proj_results is None:
        if projection_covars is None:
            assert quats is not None and scales is not None
            projection_covars, covariance_backend = _compute_covar(
                quats,
                scales,
                native_options,
            )
        proj_results = _fully_fused_projection(
            means,
            projection_covars,
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
    radii, means2d, depths, conics, compensations = proj_results
    if radius_clip > 0:
        radii = torch.where(
            radii.max(dim=-1, keepdim=True).values <= radius_clip,
            torch.zeros_like(radii),
            radii,
        )
    view_opacities = torch.broadcast_to(opacities[..., None, :], batch_dims + (C, N))
    if compensations is not None:
        view_opacities = view_opacities * compensations

    has_color = render_mode in {"RGB", "RGB-d", "RGB-Ed", "RGB+D", "RGB+ED"}
    has_depth = render_mode in {"d", "Ed", "D", "ED", "RGB-d", "RGB-Ed", "RGB+D", "RGB+ED"}
    expected_depth = render_mode in {"Ed", "ED", "RGB-Ed", "RGB+ED"}
    D = colors.shape[-1] if has_color and colors is not None else 0

    feature_list = []
    sh_backend = None
    extra_signals_sh_backend = None
    if has_color:
        if colors is None:
            raise ValueError(f"colors must be provided for render_mode={render_mode!r}.")
        color_features, sh_backend = _evaluate_features(
            colors,
            sh_degree,
            means,
            radii,
            viewmats,
            batch_dims,
            C,
            N,
            True,
            native_options,
        )
        feature_list.append(color_features)
    if extra_signals is not None:
        extra_features, extra_signals_sh_backend = _evaluate_features(
            extra_signals,
            extra_signals_sh_degree,
            means,
            radii,
            viewmats,
            batch_dims,
            C,
            N,
            False,
            native_options,
        )
        feature_list.append(extra_features)
    if has_depth:
        feature_list.append(depths[..., None])
    if not feature_list:
        raise ValueError(f"render_mode={render_mode!r} does not request any output channels.")

    features = torch.cat(feature_list, dim=-1) if len(feature_list) > 1 else feature_list[0]
    if backgrounds is not None and backgrounds.shape[-1] != features.shape[-1]:
        # Backgrounds describe only color channels in the public API. Extra and
        # depth channels should use a zero background so compositing remains sane.
        aligned_backgrounds = torch.zeros(
            *backgrounds.shape[:-1],
            features.shape[-1],
            device=backgrounds.device,
            dtype=backgrounds.dtype,
        )
        if has_color:
            copy_channels = min(D, backgrounds.shape[-1], features.shape[-1])
            aligned_backgrounds[..., :copy_channels] = backgrounds[..., :copy_channels]
        backgrounds = aligned_backgrounds

    compact_ids = None
    visible_gaussians = N
    culled_gaussians = 0
    compacted = False
    compact_backend = None
    should_compact = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (means2d, conics, features, view_opacities, depths)
    )
    if should_compact:
        native_compact_ids = _try_native_visible_gaussian_ids(
            radii,
            depths,
            N,
            native_options,
        )
        if native_compact_ids is not None:
            compact_ids, visible_gaussians = native_compact_ids
            compact_backend = "supa_native"
        else:
            compact_ids, visible_gaussians = _visible_gaussian_ids(radii, depths, N)
            compact_backend = "torch"
        culled_gaussians = N - visible_gaussians
        compacted = culled_gaussians > 0
    composite_means2d = means2d
    composite_conics = conics
    composite_features = features
    composite_opacities = view_opacities
    composite_depths = depths
    composite_radii = radii
    if compacted:
        composite_means2d = _compact_select(means2d, compact_ids, -2, native_options)
        composite_conics = _compact_select(conics, compact_ids, -2, native_options)
        composite_features = _compact_select(features, compact_ids, -2, native_options)
        composite_opacities = _compact_select(
            view_opacities,
            compact_ids,
            -1,
            native_options,
        )
        composite_depths = _compact_select(depths, compact_ids, -1)
        composite_radii = radii.index_select(-2, compact_ids.to(dtype=torch.long))
    else:
        compact_ids = None

    composite_backend = "torch"
    composite_variant = "torch"
    composite_tiles_per_gauss = None
    tiled_metadata_backend = None
    native_composite = _try_native_alpha_composite(
        composite_means2d,
        composite_conics,
        composite_features,
        composite_opacities,
        composite_depths,
        composite_radii,
        width,
        height,
        tile_size,
        backgrounds,
        trust_forward_inputs=(
            render_mode == "RGB"
            and sh_degree is None
            and extra_signals is None
            and backgrounds is None
            and (not torch.is_grad_enabled())
            and not compacted
        ),
        materialize_forward_features=(
            render_mode in {"d", "Ed", "D", "ED", "RGB-d", "RGB-Ed", "RGB+D", "RGB+ED"}
            and sh_degree is None
            and extra_signals is None
            and backgrounds is None
            and (not torch.is_grad_enabled())
            and not compacted
        ),
        native_options=native_options,
    )
    if native_composite is None:
        render_features, render_alphas = _alpha_composite(
            composite_means2d,
            composite_conics,
            composite_features,
            composite_opacities,
            composite_depths,
            composite_radii,
            width,
            height,
            backgrounds,
            chunk_size,
        )
    else:
        (
            render_features,
            render_alphas,
            composite_variant,
            composite_tiles_per_gauss,
            tiled_metadata_backend,
        ) = native_composite
        image_dims = composite_means2d.shape[:-2]
        render_features = render_features.reshape(image_dims + (height, width, features.shape[-1]))
        render_alphas = render_alphas.reshape(image_dims + (height, width, 1))
        composite_backend = "supa_native"

    meta: Dict[str, object] = {
        "batch_ids": None,
        "camera_ids": None,
        "gaussian_ids": None,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": view_opacities,
        "tile_width": math.ceil(width / float(tile_size)),
        "tile_height": math.ceil(height / float(tile_size)),
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_batches": B,
        "n_cameras": C,
        "backend": "torch_fallback",
        "projection_backend": projection_backend,
        "covariance_backend": covariance_backend,
        "composite_backend": composite_backend,
        "composite_variant": composite_variant if composite_backend == "supa_native" else "torch",
        "sh_backend": sh_backend,
        "extra_signals_sh_backend": extra_signals_sh_backend,
        "visible_gaussians": visible_gaussians,
        "culled_gaussians": culled_gaussians,
        "compact_backend": compact_backend,
    }
    if composite_tiles_per_gauss is not None:
        meta["tiles_per_gauss"] = _expand_compacted_tiles(
            composite_tiles_per_gauss,
            compact_ids,
            N,
        )
        meta["tile_count_backend"] = (
            "cpu_numpy_alpha_threshold"
            if tiled_metadata_backend == "cpu_numpy_alpha_threshold"
            else "supa_native"
        )
        if tiled_metadata_backend is not None:
            meta["tiled_metadata_backend"] = tiled_metadata_backend
    else:
        native_tiles_per_gauss = _try_native_tiles_per_gauss(
            composite_means2d,
            composite_radii,
            tile_size,
            meta["tile_width"],
            meta["tile_height"],
            projection_backend,
            synchronize_launch=torch.is_grad_enabled(),
            native_options=native_options,
        )
        if native_tiles_per_gauss is not None:
            meta["tiles_per_gauss"] = _expand_compacted_tiles(
                native_tiles_per_gauss,
                compact_ids,
                N,
            )
            meta["tile_count_backend"] = "supa_native"
    if "tiles_per_gauss" not in meta:
        compact_tiles = _tiles_per_gauss(
            composite_means2d,
            composite_radii,
            tile_size,
            meta["tile_width"],
            meta["tile_height"],
        )
        meta["tiles_per_gauss"] = _expand_compacted_tiles(
            compact_tiles,
            compact_ids,
            N,
        )
        meta["tile_count_backend"] = "torch"
    meta["isect_ids"] = torch.empty(0, dtype=torch.int64, device=means.device)
    meta["flatten_ids"] = torch.empty(0, dtype=torch.int32, device=means.device)
    meta["isect_offsets"] = torch.zeros(
        batch_dims + (C, meta["tile_height"], meta["tile_width"]),
        dtype=torch.int32,
        device=means.device,
    )

    if extra_signals is not None:
        E = extra_signals.shape[-1]
        meta["render_extra_signals"] = render_features[..., D : D + E]
        if has_depth:
            render_depth = render_features[..., -1:]
            if expected_depth:
                render_depth = render_depth / render_alphas.clamp(min=1e-10)
            render_features = torch.cat((render_features[..., :D], render_depth), dim=-1)
        else:
            render_features = render_features[..., :D]
    elif expected_depth:
        render_depth = render_features[..., -1:] / render_alphas.clamp(min=1e-10)
        render_features = torch.cat((render_features[..., :D], render_depth), dim=-1)

    return render_features, render_alphas, meta
