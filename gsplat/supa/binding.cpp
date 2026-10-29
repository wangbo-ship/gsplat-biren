// SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/core/Tensor.h>
#include <c10/core/DeviceType.h>
#include <supa.h>
#include <torch_br/csrc/core/supa/TORCHStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <climits>
#include <limits>
#include <tuple>

constexpr int kAlphaCompositeMaxChannels = 64;

extern "C" suError_t gsplat_quat_scale_to_covar_preci_fwd_launch(
    const float *quats,
    const float *scales,
    int n,
    bool triu,
    float *covars,
    float *precis,
    suStream_t stream
);

extern "C" suError_t gsplat_quat_scale_to_covar_preci_bwd_launch(
    const float *quats,
    const float *scales,
    int n,
    bool triu,
    const float *v_covars,
    const float *v_precis,
    float *v_quats,
    float *v_scales,
    suStream_t stream
);

extern "C" suError_t gsplat_projection_ewa_3dgs_pinhole_fwd_launch(
    const float *means,
    const float *covars,
    const float *quats,
    const float *scales,
    const float *viewmats,
    const float *Ks,
    int B,
    int C,
    int N,
    int image_width,
    int image_height,
    float eps2d,
    float near_plane,
    float far_plane,
    bool calc_compensations,
    int *radii,
    float *means2d,
    float *depths,
    float *conics,
    float *compensations,
    suStream_t stream
);

extern "C" suError_t gsplat_projection_ewa_3dgs_pinhole_bwd_launch(
    const float *means,
    const float *covars,
    const float *quats,
    const float *scales,
    const float *viewmats,
    const float *Ks,
    int B,
    int C,
    int N,
    int image_width,
    int image_height,
    float eps2d,
    const int *radii,
    const float *conics,
    const float *compensations,
    const float *v_means2d,
    const float *v_depths,
    const float *v_conics,
    const float *v_compensations,
    float *v_means,
    float *v_covars,
    float *v_quats,
    float *v_scales,
    float *v_viewmats,
    suStream_t stream
);

extern "C" suError_t gsplat_spherical_harmonics_fwd_launch(
    const float *dirs,
    const float *coeffs,
    int degree,
    int N,
    int K,
    float *colors,
    suStream_t stream
);

extern "C" suError_t gsplat_spherical_harmonics_bwd_launch(
    const float *dirs,
    const float *coeffs,
    const float *v_colors,
    int degree,
    int N,
    int K,
    float *v_dirs,
    float *v_coeffs,
    suStream_t stream
);

extern "C" suError_t gsplat_alpha_composite_fwd_launch(
    const float *means2d,
    const float *conics,
    const float *features,
    const float *opacities,
    const float *depths,
    const int *radii,
    const int *sorted_ids,
    const float *backgrounds,
    int I,
    int N,
    int channels,
    int image_width,
    int image_height,
    float *render,
    float *alpha_out,
    suStream_t stream
);

extern "C" suError_t gsplat_tile_count_unpacked_launch(
    const float *means2d,
    const int *radii,
    int I,
    int N,
    int tile_size,
    int tile_width,
    int tile_height,
    int *tiles_per_gauss,
    suStream_t stream
);

extern "C" suError_t gsplat_tile_count_alpha_unpacked_launch(
    const float *means2d,
    const int *radii,
    const float *conics,
    const float *opacities,
    float alpha_threshold,
    int I,
    int N,
    int tile_size,
    int tile_width,
    int tile_height,
    int *tiles_per_gauss,
    suStream_t stream
);

extern "C" suError_t gsplat_visible_compact_mark_launch(
    const int *radii,
    const float *depths,
    int I,
    int N,
    int *visible_flags,
    suStream_t stream
);

extern "C" suError_t gsplat_visible_compact_ids_launch(
    const int *visible_flags,
    int N,
    int *compact_ids,
    int *visible_count,
    suStream_t stream
);

extern "C" suError_t gsplat_compact_select_fwd_launch(
    const float *input,
    const int *compact_ids,
    int outer,
    int N,
    int K,
    int row_size,
    float *output,
    suStream_t stream
);

extern "C" suError_t gsplat_compact_select_bwd_launch(
    const float *grad_output,
    const int *compact_ids,
    int outer,
    int N,
    int K,
    int row_size,
    float *grad_input,
    suStream_t stream
);

extern "C" suError_t gsplat_isect_tiles_unpacked_lanes_launch(
    const float *means2d,
    const int *radii,
    const float *depths,
    const int *tiles_per_gauss,
    const int *starts,
    int I,
    int N,
    int tile_size,
    int tile_width,
    int tile_height,
    int tile_n_bits,
    int *dense_tile_ids,
    int *flatten_ids,
    int *isect_ids_hi,
    int *isect_ids_lo,
    suStream_t stream
);

extern "C" suError_t gsplat_isect_tiles_alpha_unpacked_lanes_launch(
    const float *means2d,
    const int *radii,
    const float *depths,
    const float *conics,
    const float *opacities,
    const int *tiles_per_gauss,
    const int *starts,
    float alpha_threshold,
    int I,
    int N,
    int tile_size,
    int tile_width,
    int tile_height,
    int tile_n_bits,
    int *dense_tile_ids,
    int *flatten_ids,
    int *isect_ids_hi,
    int *isect_ids_lo,
    suStream_t stream
);

extern "C" suError_t gsplat_tiled_offsets_count_launch(
    const int *dense_tile_ids,
    int n_isects,
    int n_tiles,
    int *tile_counts,
    suStream_t stream
);

extern "C" suError_t gsplat_tiled_offsets_prefix_launch(
    const int *tile_counts,
    int n_tiles,
    int *tile_offsets,
    int *tile_cursors,
    suStream_t stream
);

extern "C" suError_t gsplat_tiled_offsets_scatter_launch(
    const int *dense_tile_ids,
    const int *flatten_ids,
    const int *isect_ids_lo,
    int n_isects,
    int n_tiles,
    int *tile_cursors,
    int *sorted_flatten_ids,
    int *sorted_isect_ids_lo,
    int *source_orders,
    suStream_t stream
);

extern "C" suError_t gsplat_tiled_offsets_sort_launch(
    int *sorted_flatten_ids,
    int *sorted_isect_ids_lo,
    int *source_orders,
    const int *tile_offsets,
    int n_tiles,
    int n_isects,
    suStream_t stream
);

extern "C" suError_t gsplat_rasterize_to_pixels_tiled_fwd_launch(
    const float *means2d,
    const float *conics,
    const float *features,
    const float *opacities,
    const float *backgrounds,
    const int *tile_offsets,
    const int *flatten_ids,
    int n_isects,
    int I,
    int channels,
    int image_width,
    int image_height,
    int tile_width,
    int tile_height,
    int tile_size,
    float *render,
    float *alpha_out,
    suStream_t stream
);

extern "C" suError_t gsplat_rasterize_to_pixels_tiled_bwd_launch(
    const float *means2d,
    const float *conics,
    const float *features,
    const float *opacities,
    const float *backgrounds,
    const int *tile_offsets,
    const int *flatten_ids,
    const float *v_render,
    const float *v_alpha,
    const float *forward_alpha,
    int n_isects,
    int I,
    int channels,
    int image_width,
    int image_height,
    int tile_width,
    int tile_height,
    int tile_size,
    int pixel_offset,
    int pixel_count,
    float *prefix_checkpoints,
    int prefix_checkpoint_count,
    float *v_means2d,
    float *v_conics,
    float *v_features,
    float *v_opacities,
    float *v_backgrounds,
    suStream_t stream
);

extern "C" suError_t gsplat_rasterize_to_pixels_tiled_bwd_tile_reduce_launch(
    const float *means2d,
    const float *conics,
    const float *features,
    const float *opacities,
    const float *backgrounds,
    const int *tile_offsets,
    const int *flatten_ids,
    const float *v_render,
    const float *v_alpha,
    const float *forward_alpha,
    int n_isects,
    int I,
    int image_width,
    int image_height,
    int tile_width,
    int tile_height,
    int tile_size,
    int tile_offset,
    int tile_count,
    float *prefix_checkpoints,
    int prefix_checkpoint_count,
    float *v_means2d,
    float *v_conics,
    float *v_features,
    float *v_opacities,
    float *v_backgrounds,
    suStream_t stream
);

extern "C" suError_t gsplat_alpha_composite_bwd_launch(
    const float *means2d,
    const float *conics,
    const float *features,
    const float *opacities,
    const float *depths,
    const int *radii,
    const int *sorted_ids,
    const float *backgrounds,
    const float *v_render,
    const float *v_alpha,
    int I,
    int N,
    int channels,
    int image_width,
    int image_height,
    int pixel_offset,
    int pixel_count,
    float *prefix_checkpoints,
    int prefix_checkpoint_count,
    float *v_means2d,
    float *v_conics,
    float *v_features,
    float *v_opacities,
    float *v_depths,
    float *v_backgrounds,
    suStream_t stream
);

namespace {

constexpr int kAlphaCompositePrefixCheckpointInterval = 32;
constexpr int64_t kAlphaCompositeBwdWorkspaceTargetBytes = 64LL * 1024LL * 1024LL;
constexpr int kTiledBwdTileReduceSize = 16;
constexpr int kTiledBwdTileReducePixels =
    kTiledBwdTileReduceSize * kTiledBwdTileReduceSize;

void check_supa_tensor(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        name,
        " must be a SUPA tensor"
    );
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_supa_int_tensor(const at::Tensor &tensor, const char *name) {
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        name,
        " must be a SUPA tensor"
    );
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

suStream_t current_supa_stream(const at::Tensor &tensor) {
    return torch_br::getCurrentTORCHStream(tensor.device().index()).stream();
}

void check_batch_shape(
    const at::Tensor &tensor,
    at::IntArrayRef batch_dims,
    const char *name
) {
    TORCH_CHECK(
        tensor.sizes().slice(0, batch_dims.size()) == batch_dims,
        name,
        " batch shape must match means"
    );
}

struct ProjectionShape {
    int64_t B;
    int64_t C;
    int64_t N;
    at::DimVector batch_dims;
};

ProjectionShape check_projection_inputs(
    const at::Tensor &means,
    const c10::optional<at::Tensor> &covars,
    const c10::optional<at::Tensor> &quats,
    const c10::optional<at::Tensor> &scales,
    const at::Tensor &viewmats,
    const at::Tensor &Ks
) {
    const bool has_covars = covars.has_value();
    const bool has_quats = quats.has_value();
    const bool has_scales = scales.has_value();
    TORCH_CHECK(
        has_covars || (has_quats && has_scales),
        "projection requires either covars or both quats and scales"
    );
    TORCH_CHECK(
        !has_covars || (!has_quats && !has_scales),
        "projection accepts covars or quats/scales, not both"
    );
    TORCH_CHECK(has_quats == has_scales, "quats and scales must be provided together");

    check_supa_tensor(means, "means");
    check_supa_tensor(viewmats, "viewmats");
    check_supa_tensor(Ks, "Ks");
    TORCH_CHECK(means.dim() >= 2 && means.size(-1) == 3, "means must have shape [..., N, 3]");
    if (has_covars) {
        const at::Tensor &covars_tensor = covars.value();
        check_supa_tensor(covars_tensor, "covars");
        TORCH_CHECK(covars_tensor.dim() == means.dim() + 1, "covars must have shape [..., N, 3, 3]");
        TORCH_CHECK(
            covars_tensor.sizes().slice(0, means.dim() - 1) == means.sizes().slice(0, means.dim() - 1) &&
                covars_tensor.size(-2) == 3 && covars_tensor.size(-1) == 3,
            "covars must have shape [..., N, 3, 3]"
        );
    } else {
        const at::Tensor &quats_tensor = quats.value();
        const at::Tensor &scales_tensor = scales.value();
        check_supa_tensor(quats_tensor, "quats");
        check_supa_tensor(scales_tensor, "scales");
        TORCH_CHECK(quats_tensor.dim() == means.dim(), "quats must have shape [..., N, 4]");
        TORCH_CHECK(scales_tensor.dim() == means.dim(), "scales must have shape [..., N, 3]");
        TORCH_CHECK(
            quats_tensor.sizes().slice(0, means.dim() - 1) == means.sizes().slice(0, means.dim() - 1) &&
                quats_tensor.size(-1) == 4,
            "quats must have shape [..., N, 4]"
        );
        TORCH_CHECK(
            scales_tensor.sizes().slice(0, means.dim() - 1) == means.sizes().slice(0, means.dim() - 1) &&
                scales_tensor.size(-1) == 3,
            "scales must have shape [..., N, 3]"
        );
    }
    TORCH_CHECK(viewmats.dim() == means.dim() + 1, "viewmats must have shape [..., C, 4, 4]");
    TORCH_CHECK(Ks.dim() == means.dim() + 1, "Ks must have shape [..., C, 3, 3]");

    at::IntArrayRef batch_dims_ref = means.sizes().slice(0, means.dim() - 2);
    check_batch_shape(viewmats, batch_dims_ref, "viewmats");
    check_batch_shape(Ks, batch_dims_ref, "Ks");
    TORCH_CHECK(viewmats.size(-2) == 4 && viewmats.size(-1) == 4, "viewmats must have shape [..., C, 4, 4]");
    TORCH_CHECK(Ks.size(-2) == 3 && Ks.size(-1) == 3, "Ks must have shape [..., C, 3, 3]");
    TORCH_CHECK(viewmats.size(-3) == Ks.size(-3), "viewmats and Ks camera dimensions must match");

    int64_t B64 = 1;
    for (int64_t dim : batch_dims_ref) {
        B64 *= dim;
    }
    const int64_t C64 = viewmats.size(-3);
    const int64_t N64 = means.size(-2);
    TORCH_CHECK(B64 <= static_cast<int64_t>(INT_MAX), "batch size is too large");
    TORCH_CHECK(C64 <= static_cast<int64_t>(INT_MAX), "camera count is too large");
    TORCH_CHECK(N64 <= static_cast<int64_t>(INT_MAX), "gaussian count is too large");

    ProjectionShape shape;
    shape.B = B64;
    shape.C = C64;
    shape.N = N64;
    shape.batch_dims = at::DimVector(batch_dims_ref);
    return shape;
}

} // namespace

std::tuple<at::Tensor, at::Tensor> quat_scale_to_covar_preci_fwd(
    const at::Tensor &quats,
    const at::Tensor &scales,
    bool compute_covar,
    bool compute_preci,
    bool triu
) {
    check_supa_tensor(quats, "quats");
    check_supa_tensor(scales, "scales");
    TORCH_CHECK(quats.dim() >= 1 && quats.size(-1) == 4, "quats must have shape [..., 4]");
    TORCH_CHECK(scales.sizes().slice(0, scales.dim() - 1) == quats.sizes().slice(0, quats.dim() - 1), "scales batch shape must match quats");
    TORCH_CHECK(scales.size(-1) == 3, "scales must have shape [..., 3]");

    at::DimVector out_shape(quats.sizes().slice(0, quats.dim() - 1));
    if (triu) {
        out_shape.append({6});
    } else {
        out_shape.append({3, 3});
    }

    at::Tensor covars;
    at::Tensor precis;
    if (compute_covar) {
        covars = at::empty(out_shape, quats.options());
    } else {
        covars = at::empty({0}, quats.options());
    }
    if (compute_preci) {
        precis = at::empty(out_shape, quats.options());
    } else {
        precis = at::empty({0}, quats.options());
    }

    const int64_t n64 = quats.numel() / 4;
    TORCH_CHECK(n64 <= static_cast<int64_t>(INT_MAX), "too many quaternions");
    if (n64 > 0) {
        suError_t err = gsplat_quat_scale_to_covar_preci_fwd_launch(
            quats.const_data_ptr<float>(),
            scales.const_data_ptr<float>(),
            static_cast<int>(n64),
            triu,
            compute_covar ? covars.data_ptr<float>() : nullptr,
            compute_preci ? precis.data_ptr<float>() : nullptr,
            current_supa_stream(quats)
        );
        TORCH_CHECK(err == suSuccess, "SUPA quat_scale_to_covar_preci_fwd failed: ", suGetErrorString(err));
    }
    return std::make_tuple(covars, precis);
}

std::tuple<at::Tensor, at::Tensor> quat_scale_to_covar_preci_bwd(
    const at::Tensor &quats,
    const at::Tensor &scales,
    bool triu,
    const c10::optional<at::Tensor> &v_covars,
    const c10::optional<at::Tensor> &v_precis
) {
    check_supa_tensor(quats, "quats");
    check_supa_tensor(scales, "scales");
    const bool has_v_covars = v_covars.has_value();
    const bool has_v_precis = v_precis.has_value();
    if (has_v_covars) {
        check_supa_tensor(v_covars.value(), "v_covars");
    }
    if (has_v_precis) {
        check_supa_tensor(v_precis.value(), "v_precis");
    }

    at::Tensor v_quats = at::empty_like(quats);
    at::Tensor v_scales = at::empty_like(scales);
    const int64_t n64 = quats.numel() / 4;
    TORCH_CHECK(n64 <= static_cast<int64_t>(INT_MAX), "too many quaternions");
    if (n64 == 0 || (!has_v_covars && !has_v_precis)) {
        v_quats.zero_();
        v_scales.zero_();
        return std::make_tuple(v_quats, v_scales);
    }

    suError_t err = gsplat_quat_scale_to_covar_preci_bwd_launch(
        quats.const_data_ptr<float>(),
        scales.const_data_ptr<float>(),
        static_cast<int>(n64),
        triu,
        has_v_covars ? v_covars.value().const_data_ptr<float>() : nullptr,
        has_v_precis ? v_precis.value().const_data_ptr<float>() : nullptr,
        v_quats.data_ptr<float>(),
        v_scales.data_ptr<float>(),
        current_supa_stream(quats)
    );
    TORCH_CHECK(err == suSuccess, "SUPA quat_scale_to_covar_preci_bwd failed: ", suGetErrorString(err));
    return std::make_tuple(v_quats, v_scales);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_pinhole_fwd(
    const at::Tensor &means,
    const c10::optional<at::Tensor> &covars,
    const c10::optional<at::Tensor> &quats,
    const c10::optional<at::Tensor> &scales,
    const at::Tensor &viewmats,
    const at::Tensor &Ks,
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    bool calc_compensations
) {
    ProjectionShape shape = check_projection_inputs(means, covars, quats, scales, viewmats, Ks);
    TORCH_CHECK(image_width <= static_cast<int64_t>(INT_MAX), "image_width is too large");
    TORCH_CHECK(image_height <= static_cast<int64_t>(INT_MAX), "image_height is too large");

    at::DimVector out_batch(shape.batch_dims);
    at::DimVector radii_shape(out_batch);
    radii_shape.append({shape.C, shape.N, 2});
    at::DimVector means2d_shape(out_batch);
    means2d_shape.append({shape.C, shape.N, 2});
    at::DimVector scalar_shape(out_batch);
    scalar_shape.append({shape.C, shape.N});
    at::DimVector conics_shape(out_batch);
    conics_shape.append({shape.C, shape.N, 3});

    at::Tensor radii = at::empty(radii_shape, means.options().dtype(at::kInt));
    at::Tensor means2d = at::empty(means2d_shape, means.options());
    at::Tensor depths = at::empty(scalar_shape, means.options());
    at::Tensor conics = at::empty(conics_shape, means.options());
    at::Tensor compensations = calc_compensations ? at::empty(scalar_shape, means.options()) : at::empty({0}, means.options());

    const int64_t total = shape.B * shape.C * shape.N;
    TORCH_CHECK(total <= static_cast<int64_t>(INT_MAX), "projection output is too large");
    if (total > 0) {
        const bool has_covars = covars.has_value();
        suError_t err = gsplat_projection_ewa_3dgs_pinhole_fwd_launch(
            means.const_data_ptr<float>(),
            has_covars ? covars.value().const_data_ptr<float>() : nullptr,
            has_covars ? nullptr : quats.value().const_data_ptr<float>(),
            has_covars ? nullptr : scales.value().const_data_ptr<float>(),
            viewmats.const_data_ptr<float>(),
            Ks.const_data_ptr<float>(),
            static_cast<int>(shape.B),
            static_cast<int>(shape.C),
            static_cast<int>(shape.N),
            static_cast<int>(image_width),
            static_cast<int>(image_height),
            static_cast<float>(eps2d),
            static_cast<float>(near_plane),
            static_cast<float>(far_plane),
            calc_compensations,
            radii.data_ptr<int>(),
            means2d.data_ptr<float>(),
            depths.data_ptr<float>(),
            conics.data_ptr<float>(),
            calc_compensations ? compensations.data_ptr<float>() : nullptr,
            current_supa_stream(means)
        );
        TORCH_CHECK(err == suSuccess, "SUPA projection_ewa_3dgs_pinhole_fwd failed: ", suGetErrorString(err));
    }
    return std::make_tuple(radii, means2d, depths, conics, compensations);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> projection_ewa_3dgs_pinhole_bwd(
    const at::Tensor &means,
    const c10::optional<at::Tensor> &covars,
    const c10::optional<at::Tensor> &quats,
    const c10::optional<at::Tensor> &scales,
    const at::Tensor &viewmats,
    const at::Tensor &Ks,
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    const at::Tensor &radii,
    const at::Tensor &conics,
    const c10::optional<at::Tensor> &compensations,
    const at::Tensor &v_means2d,
    const c10::optional<at::Tensor> &v_depths,
    const at::Tensor &v_conics,
    const c10::optional<at::Tensor> &v_compensations,
    bool compute_v_means,
    bool compute_v_covars,
    bool compute_v_quats,
    bool compute_v_scales,
    bool compute_v_viewmats
) {
    ProjectionShape shape = check_projection_inputs(means, covars, quats, scales, viewmats, Ks);
    check_supa_int_tensor(radii, "radii");
    check_supa_tensor(conics, "conics");
    check_supa_tensor(v_means2d, "v_means2d");
    if (v_depths.has_value()) {
        check_supa_tensor(v_depths.value(), "v_depths");
    }
    check_supa_tensor(v_conics, "v_conics");
    if (compensations.has_value()) {
        check_supa_tensor(compensations.value(), "compensations");
    }
    if (v_compensations.has_value()) {
        check_supa_tensor(v_compensations.value(), "v_compensations");
    }
    TORCH_CHECK(image_width <= static_cast<int64_t>(INT_MAX), "image_width is too large");
    TORCH_CHECK(image_height <= static_cast<int64_t>(INT_MAX), "image_height is too large");

    at::DimVector out_batch(shape.batch_dims);
    at::DimVector radii_shape(out_batch);
    radii_shape.append({shape.C, shape.N, 2});
    at::DimVector means2d_shape(out_batch);
    means2d_shape.append({shape.C, shape.N, 2});
    at::DimVector scalar_shape(out_batch);
    scalar_shape.append({shape.C, shape.N});
    at::DimVector conics_shape(out_batch);
    conics_shape.append({shape.C, shape.N, 3});
    TORCH_CHECK(radii.sizes() == radii_shape, "radii shape must match forward output");
    TORCH_CHECK(conics.sizes() == conics_shape, "conics shape must match forward output");
    TORCH_CHECK(v_means2d.sizes() == means2d_shape, "v_means2d shape must match forward output");
    if (v_depths.has_value()) {
        TORCH_CHECK(v_depths.value().sizes() == scalar_shape, "v_depths shape must match forward output");
    }
    TORCH_CHECK(v_conics.sizes() == conics_shape, "v_conics shape must match forward output");
    if (compensations.has_value()) {
        TORCH_CHECK(compensations.value().sizes() == scalar_shape, "compensations shape must match forward output");
    }
    if (v_compensations.has_value()) {
        TORCH_CHECK(v_compensations.value().sizes() == scalar_shape, "v_compensations shape must match forward output");
        TORCH_CHECK(compensations.has_value(), "compensations are required when v_compensations is provided");
    }

    const bool has_covars = covars.has_value();
    const bool need_v_covars = has_covars && compute_v_covars;
    const bool need_v_quats = !has_covars && compute_v_quats;
    const bool need_v_scales = !has_covars && compute_v_scales;
    at::Tensor v_means = compute_v_means ? at::zeros_like(means) : at::empty({0}, means.options());
    at::Tensor v_covars = need_v_covars ? at::zeros_like(covars.value()) : at::empty({0}, means.options());
    at::Tensor v_quats = need_v_quats ? at::zeros_like(quats.value()) : at::empty({0}, means.options());
    at::Tensor v_scales = need_v_scales ? at::zeros_like(scales.value()) : at::empty({0}, means.options());
    at::Tensor v_viewmats = compute_v_viewmats ? at::zeros_like(viewmats) : at::empty({0}, means.options());
    const int64_t total = shape.B * shape.C * shape.N;
    TORCH_CHECK(total <= static_cast<int64_t>(INT_MAX), "projection output is too large");
    if (total > 0) {
        suError_t err = gsplat_projection_ewa_3dgs_pinhole_bwd_launch(
            means.const_data_ptr<float>(),
            has_covars ? covars.value().const_data_ptr<float>() : nullptr,
            has_covars ? nullptr : quats.value().const_data_ptr<float>(),
            has_covars ? nullptr : scales.value().const_data_ptr<float>(),
            viewmats.const_data_ptr<float>(),
            Ks.const_data_ptr<float>(),
            static_cast<int>(shape.B),
            static_cast<int>(shape.C),
            static_cast<int>(shape.N),
            static_cast<int>(image_width),
            static_cast<int>(image_height),
            static_cast<float>(eps2d),
            radii.const_data_ptr<int>(),
            conics.const_data_ptr<float>(),
            compensations.has_value() ? compensations.value().const_data_ptr<float>() : nullptr,
            v_means2d.const_data_ptr<float>(),
            v_depths.has_value() ? v_depths.value().const_data_ptr<float>() : nullptr,
            v_conics.const_data_ptr<float>(),
            v_compensations.has_value() ? v_compensations.value().const_data_ptr<float>() : nullptr,
            compute_v_means ? v_means.data_ptr<float>() : nullptr,
            need_v_covars ? v_covars.data_ptr<float>() : nullptr,
            need_v_quats ? v_quats.data_ptr<float>() : nullptr,
            need_v_scales ? v_scales.data_ptr<float>() : nullptr,
            compute_v_viewmats ? v_viewmats.data_ptr<float>() : nullptr,
            current_supa_stream(means)
        );
        TORCH_CHECK(err == suSuccess, "SUPA projection_ewa_3dgs_pinhole_bwd failed: ", suGetErrorString(err));
    }
    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}

at::Tensor spherical_harmonics_fwd(
    int64_t degree,
    const at::Tensor &dirs,
    const at::Tensor &coeffs
) {
    check_supa_tensor(dirs, "dirs");
    check_supa_tensor(coeffs, "coeffs");
    TORCH_CHECK(degree >= 0 && degree <= 4, "spherical harmonics degree must be in [0, 4]");
    TORCH_CHECK(dirs.dim() >= 1 && dirs.size(-1) == 3, "dirs must have shape [..., 3]");
    TORCH_CHECK(coeffs.dim() == dirs.dim() + 1, "coeffs must have shape [..., K, 3]");
    TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have three RGB channels");
    TORCH_CHECK(
        coeffs.sizes().slice(0, dirs.dim() - 1) == dirs.sizes().slice(0, dirs.dim() - 1),
        "coeffs batch shape must match dirs"
    );
    const int64_t required_k = (degree + 1) * (degree + 1);
    TORCH_CHECK(coeffs.size(-2) >= required_k, "coeffs do not contain enough SH coefficients");

    at::Tensor colors = at::empty_like(dirs);
    const int64_t N64 = dirs.numel() / 3;
    TORCH_CHECK(N64 <= static_cast<int64_t>(INT_MAX), "spherical harmonics input is too large");
    TORCH_CHECK(coeffs.size(-2) <= static_cast<int64_t>(INT_MAX), "SH coefficient count is too large");
    if (N64 > 0) {
        suError_t err = gsplat_spherical_harmonics_fwd_launch(
            dirs.const_data_ptr<float>(),
            coeffs.const_data_ptr<float>(),
            static_cast<int>(degree),
            static_cast<int>(N64),
            static_cast<int>(coeffs.size(-2)),
            colors.data_ptr<float>(),
            current_supa_stream(dirs)
        );
        TORCH_CHECK(err == suSuccess, "SUPA spherical_harmonics_fwd failed: ", suGetErrorString(err));
    }
    return colors;
}

std::tuple<at::Tensor, at::Tensor> spherical_harmonics_bwd(
    int64_t degree,
    const at::Tensor &dirs,
    const at::Tensor &coeffs,
    const at::Tensor &v_colors,
    bool dirs_requires_grad
) {
    check_supa_tensor(dirs, "dirs");
    check_supa_tensor(coeffs, "coeffs");
    check_supa_tensor(v_colors, "v_colors");
    TORCH_CHECK(degree >= 0 && degree <= 4, "spherical harmonics degree must be in [0, 4]");
    TORCH_CHECK(dirs.dim() >= 1 && dirs.size(-1) == 3, "dirs must have shape [..., 3]");
    TORCH_CHECK(coeffs.dim() == dirs.dim() + 1, "coeffs must have shape [..., K, 3]");
    TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have three RGB channels");
    TORCH_CHECK(v_colors.sizes() == dirs.sizes(), "v_colors must match dirs shape");
    TORCH_CHECK(
        coeffs.sizes().slice(0, dirs.dim() - 1) == dirs.sizes().slice(0, dirs.dim() - 1),
        "coeffs batch shape must match dirs"
    );
    const int64_t required_k = (degree + 1) * (degree + 1);
    TORCH_CHECK(coeffs.size(-2) >= required_k, "coeffs do not contain enough SH coefficients");

    at::Tensor v_dirs = dirs_requires_grad ? at::zeros_like(dirs) : at::empty({0}, dirs.options());
    at::Tensor v_coeffs = at::zeros_like(coeffs);
    const int64_t N64 = dirs.numel() / 3;
    TORCH_CHECK(N64 <= static_cast<int64_t>(INT_MAX), "spherical harmonics input is too large");
    TORCH_CHECK(coeffs.size(-2) <= static_cast<int64_t>(INT_MAX), "SH coefficient count is too large");
    if (N64 > 0) {
        suError_t err = gsplat_spherical_harmonics_bwd_launch(
            dirs.const_data_ptr<float>(),
            coeffs.const_data_ptr<float>(),
            v_colors.const_data_ptr<float>(),
            static_cast<int>(degree),
            static_cast<int>(N64),
            static_cast<int>(coeffs.size(-2)),
            dirs_requires_grad ? v_dirs.data_ptr<float>() : nullptr,
            v_coeffs.data_ptr<float>(),
            current_supa_stream(dirs)
        );
        TORCH_CHECK(err == suSuccess, "SUPA spherical_harmonics_bwd failed: ", suGetErrorString(err));
    }
    return std::make_tuple(v_dirs, v_coeffs);
}

at::Tensor tile_count_unpacked(
    const at::Tensor &means2d,
    const at::Tensor &radii,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height
) {
    check_supa_tensor(means2d, "means2d");
    check_supa_int_tensor(radii, "radii");
    TORCH_CHECK(means2d.dim() == 3 && means2d.size(-1) == 2, "means2d must have shape [I, N, 2]");
    TORCH_CHECK(radii.sizes() == means2d.sizes(), "radii must have shape [I, N, 2]");
    TORCH_CHECK(tile_size > 0 && tile_size <= static_cast<int64_t>(INT_MAX), "tile_size must fit int32");
    TORCH_CHECK(tile_width >= 0 && tile_width <= static_cast<int64_t>(INT_MAX), "tile_width must fit int32");
    TORCH_CHECK(tile_height >= 0 && tile_height <= static_cast<int64_t>(INT_MAX), "tile_height must fit int32");
    TORCH_CHECK(means2d.size(0) <= static_cast<int64_t>(INT_MAX), "image count is too large");
    TORCH_CHECK(means2d.size(1) <= static_cast<int64_t>(INT_MAX), "gaussian count is too large");

    const int64_t total = means2d.size(0) * means2d.size(1);
    TORCH_CHECK(total <= static_cast<int64_t>(INT_MAX), "tile count output is too large");
    at::Tensor flat_tiles_per_gauss = at::empty({total}, radii.options());
    if (total > 0) {
        suError_t err = gsplat_tile_count_unpacked_launch(
            means2d.const_data_ptr<float>(),
            radii.const_data_ptr<int>(),
            static_cast<int>(means2d.size(0)),
            static_cast<int>(means2d.size(1)),
            static_cast<int>(tile_size),
            static_cast<int>(tile_width),
            static_cast<int>(tile_height),
            flat_tiles_per_gauss.data_ptr<int>(),
            current_supa_stream(means2d)
        );
        TORCH_CHECK(err == suSuccess, "SUPA tile_count_unpacked failed: ", suGetErrorString(err));
    }
    return flat_tiles_per_gauss.reshape({means2d.size(0), means2d.size(1)});
}

at::Tensor tile_count_alpha_unpacked(
    const at::Tensor &means2d,
    const at::Tensor &radii,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    double alpha_threshold,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height
) {
    check_supa_tensor(means2d, "means2d");
    check_supa_int_tensor(radii, "radii");
    check_supa_tensor(conics, "conics");
    check_supa_tensor(opacities, "opacities");
    TORCH_CHECK(means2d.dim() == 3 && means2d.size(-1) == 2, "means2d must have shape [I, N, 2]");
    TORCH_CHECK(radii.sizes() == means2d.sizes(), "radii must have shape [I, N, 2]");
    TORCH_CHECK(conics.dim() == 3 && conics.size(-1) == 3, "conics must have shape [I, N, 3]");
    TORCH_CHECK(conics.size(0) == means2d.size(0) && conics.size(1) == means2d.size(1), "conics must agree with means2d");
    TORCH_CHECK(opacities.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1)}), "opacities must have shape [I, N]");
    TORCH_CHECK(alpha_threshold > 0.0 && alpha_threshold < 1.0, "alpha_threshold must be in (0, 1)");
    TORCH_CHECK(tile_size > 0 && tile_size <= static_cast<int64_t>(INT_MAX), "tile_size must fit int32");
    TORCH_CHECK(tile_width >= 0 && tile_width <= static_cast<int64_t>(INT_MAX), "tile_width must fit int32");
    TORCH_CHECK(tile_height >= 0 && tile_height <= static_cast<int64_t>(INT_MAX), "tile_height must fit int32");
    TORCH_CHECK(means2d.size(0) <= static_cast<int64_t>(INT_MAX), "image count is too large");
    TORCH_CHECK(means2d.size(1) <= static_cast<int64_t>(INT_MAX), "gaussian count is too large");

    const int64_t total = means2d.size(0) * means2d.size(1);
    TORCH_CHECK(total <= static_cast<int64_t>(INT_MAX), "tile count output is too large");
    at::Tensor flat_tiles_per_gauss = at::empty({total}, radii.options());
    if (total > 0) {
        suError_t err = gsplat_tile_count_alpha_unpacked_launch(
            means2d.const_data_ptr<float>(),
            radii.const_data_ptr<int>(),
            conics.const_data_ptr<float>(),
            opacities.const_data_ptr<float>(),
            static_cast<float>(alpha_threshold),
            static_cast<int>(means2d.size(0)),
            static_cast<int>(means2d.size(1)),
            static_cast<int>(tile_size),
            static_cast<int>(tile_width),
            static_cast<int>(tile_height),
            flat_tiles_per_gauss.data_ptr<int>(),
            current_supa_stream(means2d)
        );
        TORCH_CHECK(err == suSuccess, "SUPA tile_count_alpha_unpacked failed: ", suGetErrorString(err));
    }
    return flat_tiles_per_gauss.reshape({means2d.size(0), means2d.size(1)});
}

std::tuple<at::Tensor, at::Tensor> visible_compact_ids(
    const at::Tensor &radii,
    const at::Tensor &depths
) {
    check_supa_int_tensor(radii, "radii");
    check_supa_tensor(depths, "depths");
    TORCH_CHECK(radii.dim() == 3 && radii.size(-1) == 2, "radii must have shape [I, N, 2]");
    TORCH_CHECK(depths.dim() == 2, "depths must have shape [I, N]");
    TORCH_CHECK(
        radii.size(0) == depths.size(0) && radii.size(1) == depths.size(1),
        "radii and depths must agree on I and N"
    );
    TORCH_CHECK(radii.size(0) <= static_cast<int64_t>(INT_MAX), "image count is too large");
    TORCH_CHECK(radii.size(1) <= static_cast<int64_t>(INT_MAX), "gaussian count is too large");

    const int64_t total64 = radii.size(0) * radii.size(1);
    TORCH_CHECK(total64 <= static_cast<int64_t>(INT_MAX), "visible compact input is too large");
    at::Tensor visible_flags = at::zeros({radii.size(1)}, radii.options());
    at::Tensor compact_ids = at::empty({radii.size(1)}, radii.options());
    at::Tensor visible_count = at::zeros({1}, radii.options());
    if (total64 > 0) {
        suError_t err = gsplat_visible_compact_mark_launch(
            radii.const_data_ptr<int>(),
            depths.const_data_ptr<float>(),
            static_cast<int>(radii.size(0)),
            static_cast<int>(radii.size(1)),
            visible_flags.data_ptr<int>(),
            current_supa_stream(radii)
        );
        TORCH_CHECK(err == suSuccess, "SUPA visible_compact mark failed: ", suGetErrorString(err));
    }
    if (radii.size(1) > 0) {
        suError_t err = gsplat_visible_compact_ids_launch(
            visible_flags.const_data_ptr<int>(),
            static_cast<int>(radii.size(1)),
            compact_ids.data_ptr<int>(),
            visible_count.data_ptr<int>(),
            current_supa_stream(radii)
        );
        TORCH_CHECK(err == suSuccess, "SUPA visible_compact ids failed: ", suGetErrorString(err));
    }
    return std::make_tuple(compact_ids, visible_count);
}

at::Tensor compact_select_fwd(
    const at::Tensor &input,
    const at::Tensor &compact_ids
) {
    check_supa_tensor(input, "input");
    check_supa_int_tensor(compact_ids, "compact_ids");
    TORCH_CHECK(input.dim() == 3, "input must have shape [outer, N, row_size]");
    TORCH_CHECK(compact_ids.dim() == 1, "compact_ids must be 1D");
    TORCH_CHECK(input.size(0) <= static_cast<int64_t>(INT_MAX), "outer dimension is too large");
    TORCH_CHECK(input.size(1) <= static_cast<int64_t>(INT_MAX), "gaussian count is too large");
    TORCH_CHECK(input.size(2) <= static_cast<int64_t>(INT_MAX), "row size is too large");
    TORCH_CHECK(compact_ids.numel() <= static_cast<int64_t>(INT_MAX), "compact id count is too large");

    at::Tensor output = at::empty({input.size(0), compact_ids.numel(), input.size(2)}, input.options());
    const int64_t total64 = input.size(0) * compact_ids.numel() * input.size(2);
    TORCH_CHECK(total64 <= static_cast<int64_t>(INT_MAX), "compact select output is too large");
    if (total64 > 0) {
        suError_t err = gsplat_compact_select_fwd_launch(
            input.const_data_ptr<float>(),
            compact_ids.const_data_ptr<int>(),
            static_cast<int>(input.size(0)),
            static_cast<int>(input.size(1)),
            static_cast<int>(compact_ids.numel()),
            static_cast<int>(input.size(2)),
            output.data_ptr<float>(),
            current_supa_stream(input)
        );
        TORCH_CHECK(err == suSuccess, "SUPA compact_select_fwd failed: ", suGetErrorString(err));
    }
    return output;
}

at::Tensor compact_select_bwd(
    const at::Tensor &grad_output,
    const at::Tensor &compact_ids,
    int64_t n_gaussians
) {
    check_supa_tensor(grad_output, "grad_output");
    check_supa_int_tensor(compact_ids, "compact_ids");
    TORCH_CHECK(grad_output.dim() == 3, "grad_output must have shape [outer, K, row_size]");
    TORCH_CHECK(compact_ids.dim() == 1, "compact_ids must be 1D");
    TORCH_CHECK(grad_output.size(1) == compact_ids.numel(), "grad_output K must match compact_ids");
    TORCH_CHECK(n_gaussians >= 0 && n_gaussians <= static_cast<int64_t>(INT_MAX), "gaussian count is too large");
    TORCH_CHECK(grad_output.size(0) <= static_cast<int64_t>(INT_MAX), "outer dimension is too large");
    TORCH_CHECK(grad_output.size(2) <= static_cast<int64_t>(INT_MAX), "row size is too large");

    at::Tensor grad_input = at::zeros({grad_output.size(0), n_gaussians, grad_output.size(2)}, grad_output.options());
    const int64_t total64 = grad_output.numel();
    TORCH_CHECK(total64 <= static_cast<int64_t>(INT_MAX), "compact select gradient is too large");
    if (total64 > 0) {
        suError_t err = gsplat_compact_select_bwd_launch(
            grad_output.const_data_ptr<float>(),
            compact_ids.const_data_ptr<int>(),
            static_cast<int>(grad_output.size(0)),
            static_cast<int>(n_gaussians),
            static_cast<int>(compact_ids.numel()),
            static_cast<int>(grad_output.size(2)),
            grad_input.data_ptr<float>(),
            current_supa_stream(grad_output)
        );
        TORCH_CHECK(err == suSuccess, "SUPA compact_select_bwd failed: ", suGetErrorString(err));
    }
    return grad_input;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
isect_tiles_unpacked_lanes(
    const at::Tensor &means2d,
    const at::Tensor &radii,
    const at::Tensor &depths,
    const at::Tensor &tiles_per_gauss,
    const at::Tensor &starts,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height
) {
    check_supa_tensor(means2d, "means2d");
    check_supa_tensor(depths, "depths");
    check_supa_int_tensor(radii, "radii");
    check_supa_int_tensor(tiles_per_gauss, "tiles_per_gauss");
    check_supa_int_tensor(starts, "starts");
    TORCH_CHECK(means2d.dim() == 3 && means2d.size(-1) == 2, "means2d must have shape [I, N, 2]");
    TORCH_CHECK(radii.sizes() == means2d.sizes(), "radii must have shape [I, N, 2]");
    TORCH_CHECK(depths.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1)}), "depths must have shape [I, N]");
    TORCH_CHECK(
        tiles_per_gauss.numel() == depths.numel(),
        "tiles_per_gauss must have I * N elements"
    );
    TORCH_CHECK(starts.numel() == depths.numel(), "starts must have I * N elements");
    TORCH_CHECK(tile_size > 0 && tile_size <= static_cast<int64_t>(INT_MAX), "tile_size must fit int32");
    TORCH_CHECK(tile_width > 0 && tile_width <= static_cast<int64_t>(INT_MAX), "tile_width must fit int32");
    TORCH_CHECK(tile_height > 0 && tile_height <= static_cast<int64_t>(INT_MAX), "tile_height must fit int32");
    TORCH_CHECK(means2d.size(0) <= static_cast<int64_t>(INT_MAX), "image count is too large");
    TORCH_CHECK(means2d.size(1) <= static_cast<int64_t>(INT_MAX), "gaussian count is too large");

    int64_t n_isects64 = 0;
    if (tiles_per_gauss.numel() > 0) {
        n_isects64 = tiles_per_gauss.sum().item<int64_t>();
    }
    TORCH_CHECK(n_isects64 >= 0 && n_isects64 <= static_cast<int64_t>(INT_MAX), "intersection count is too large");
    const int64_t tile_count64 = tile_width * tile_height;
    TORCH_CHECK(tile_count64 > 0 && tile_count64 <= static_cast<int64_t>(INT_MAX), "tile count is too large");
    int tile_n_bits = 0;
    int64_t bit_value = tile_count64;
    while (bit_value > 0) {
        ++tile_n_bits;
        bit_value >>= 1;
    }

    at::Tensor dense_tile_ids = at::empty({n_isects64}, radii.options());
    at::Tensor flatten_ids = at::empty({n_isects64}, radii.options());
    at::Tensor isect_ids_hi = at::empty({n_isects64}, radii.options());
    at::Tensor isect_ids_lo = at::empty({n_isects64}, radii.options());
    const int64_t total = means2d.size(0) * means2d.size(1);
    TORCH_CHECK(total <= static_cast<int64_t>(INT_MAX), "metadata input is too large");
    if (total > 0 && n_isects64 > 0) {
        suError_t err = gsplat_isect_tiles_unpacked_lanes_launch(
            means2d.const_data_ptr<float>(),
            radii.const_data_ptr<int>(),
            depths.const_data_ptr<float>(),
            tiles_per_gauss.const_data_ptr<int>(),
            starts.const_data_ptr<int>(),
            static_cast<int>(means2d.size(0)),
            static_cast<int>(means2d.size(1)),
            static_cast<int>(tile_size),
            static_cast<int>(tile_width),
            static_cast<int>(tile_height),
            tile_n_bits,
            dense_tile_ids.data_ptr<int>(),
            flatten_ids.data_ptr<int>(),
            isect_ids_hi.data_ptr<int>(),
            isect_ids_lo.data_ptr<int>(),
            current_supa_stream(means2d)
        );
        TORCH_CHECK(err == suSuccess, "SUPA isect_tiles_unpacked_lanes failed: ", suGetErrorString(err));
    }
    return std::make_tuple(dense_tile_ids, flatten_ids, isect_ids_hi, isect_ids_lo);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
isect_tiles_unpacked_lanes_from_counts(
    const at::Tensor &means2d,
    const at::Tensor &radii,
    const at::Tensor &depths,
    const at::Tensor &tiles_per_gauss,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height
) {
    check_supa_tensor(means2d, "means2d");
    check_supa_tensor(depths, "depths");
    check_supa_int_tensor(radii, "radii");
    check_supa_int_tensor(tiles_per_gauss, "tiles_per_gauss");
    TORCH_CHECK(means2d.dim() == 3 && means2d.size(-1) == 2, "means2d must have shape [I, N, 2]");
    TORCH_CHECK(radii.sizes() == means2d.sizes(), "radii must have shape [I, N, 2]");
    TORCH_CHECK(depths.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1)}), "depths must have shape [I, N]");
    TORCH_CHECK(tiles_per_gauss.sizes() == depths.sizes(), "tiles_per_gauss must have shape [I, N]");
    TORCH_CHECK(tile_size > 0 && tile_size <= static_cast<int64_t>(INT_MAX), "tile_size must fit int32");
    TORCH_CHECK(tile_width > 0 && tile_width <= static_cast<int64_t>(INT_MAX), "tile_width must fit int32");
    TORCH_CHECK(tile_height > 0 && tile_height <= static_cast<int64_t>(INT_MAX), "tile_height must fit int32");
    TORCH_CHECK(means2d.size(0) <= static_cast<int64_t>(INT_MAX), "image count is too large");
    TORCH_CHECK(means2d.size(1) <= static_cast<int64_t>(INT_MAX), "gaussian count is too large");

    int64_t n_isects64 = 0;
    if (tiles_per_gauss.numel() > 0) {
        n_isects64 = tiles_per_gauss.sum().item<int64_t>();
    }
    TORCH_CHECK(n_isects64 >= 0 && n_isects64 <= static_cast<int64_t>(INT_MAX), "intersection count is too large");
    const int64_t tile_count64 = tile_width * tile_height;
    TORCH_CHECK(tile_count64 > 0 && tile_count64 <= static_cast<int64_t>(INT_MAX), "tile count is too large");
    int tile_n_bits = 0;
    int64_t bit_value = tile_count64;
    while (bit_value > 0) {
        ++tile_n_bits;
        bit_value >>= 1;
    }

    at::Tensor dense_tile_ids = at::empty({n_isects64}, radii.options());
    at::Tensor flatten_ids = at::empty({n_isects64}, radii.options());
    at::Tensor isect_ids_hi = at::empty({n_isects64}, radii.options());
    at::Tensor isect_ids_lo = at::empty({n_isects64}, radii.options());
    const int64_t total = means2d.size(0) * means2d.size(1);
    TORCH_CHECK(total <= static_cast<int64_t>(INT_MAX), "metadata input is too large");
    if (total > 0 && n_isects64 > 0) {
        at::Tensor starts = at::empty({total}, radii.options());
        at::Tensor start_scratch = at::empty({total}, radii.options());
        suStream_t stream = current_supa_stream(means2d);
        suError_t err = gsplat_tiled_offsets_prefix_launch(
            tiles_per_gauss.const_data_ptr<int>(),
            static_cast<int>(total),
            starts.data_ptr<int>(),
            start_scratch.data_ptr<int>(),
            stream
        );
        TORCH_CHECK(err == suSuccess, "SUPA isect_tiles_unpacked_lanes starts failed: ", suGetErrorString(err));

        err = gsplat_isect_tiles_unpacked_lanes_launch(
            means2d.const_data_ptr<float>(),
            radii.const_data_ptr<int>(),
            depths.const_data_ptr<float>(),
            tiles_per_gauss.const_data_ptr<int>(),
            starts.const_data_ptr<int>(),
            static_cast<int>(means2d.size(0)),
            static_cast<int>(means2d.size(1)),
            static_cast<int>(tile_size),
            static_cast<int>(tile_width),
            static_cast<int>(tile_height),
            tile_n_bits,
            dense_tile_ids.data_ptr<int>(),
            flatten_ids.data_ptr<int>(),
            isect_ids_hi.data_ptr<int>(),
            isect_ids_lo.data_ptr<int>(),
            stream
        );
        TORCH_CHECK(err == suSuccess, "SUPA isect_tiles_unpacked_lanes failed: ", suGetErrorString(err));
    }
    return std::make_tuple(dense_tile_ids, flatten_ids, isect_ids_hi, isect_ids_lo);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
isect_tiles_alpha_unpacked_lanes_from_counts(
    const at::Tensor &means2d,
    const at::Tensor &radii,
    const at::Tensor &depths,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &tiles_per_gauss,
    double alpha_threshold,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height
) {
    check_supa_tensor(means2d, "means2d");
    check_supa_tensor(depths, "depths");
    check_supa_tensor(conics, "conics");
    check_supa_tensor(opacities, "opacities");
    check_supa_int_tensor(radii, "radii");
    check_supa_int_tensor(tiles_per_gauss, "tiles_per_gauss");
    TORCH_CHECK(means2d.dim() == 3 && means2d.size(-1) == 2, "means2d must have shape [I, N, 2]");
    TORCH_CHECK(radii.sizes() == means2d.sizes(), "radii must have shape [I, N, 2]");
    TORCH_CHECK(depths.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1)}), "depths must have shape [I, N]");
    TORCH_CHECK(conics.dim() == 3 && conics.size(-1) == 3, "conics must have shape [I, N, 3]");
    TORCH_CHECK(conics.size(0) == means2d.size(0) && conics.size(1) == means2d.size(1), "conics must agree with means2d");
    TORCH_CHECK(opacities.sizes() == depths.sizes(), "opacities must have shape [I, N]");
    TORCH_CHECK(tiles_per_gauss.sizes() == depths.sizes(), "tiles_per_gauss must have shape [I, N]");
    TORCH_CHECK(alpha_threshold > 0.0 && alpha_threshold < 1.0, "alpha_threshold must be in (0, 1)");
    TORCH_CHECK(tile_size > 0 && tile_size <= static_cast<int64_t>(INT_MAX), "tile_size must fit int32");
    TORCH_CHECK(tile_width > 0 && tile_width <= static_cast<int64_t>(INT_MAX), "tile_width must fit int32");
    TORCH_CHECK(tile_height > 0 && tile_height <= static_cast<int64_t>(INT_MAX), "tile_height must fit int32");
    TORCH_CHECK(means2d.size(0) <= static_cast<int64_t>(INT_MAX), "image count is too large");
    TORCH_CHECK(means2d.size(1) <= static_cast<int64_t>(INT_MAX), "gaussian count is too large");

    int64_t n_isects64 = 0;
    if (tiles_per_gauss.numel() > 0) {
        n_isects64 = tiles_per_gauss.sum().item<int64_t>();
    }
    TORCH_CHECK(n_isects64 >= 0 && n_isects64 <= static_cast<int64_t>(INT_MAX), "intersection count is too large");
    const int64_t tile_count64 = tile_width * tile_height;
    TORCH_CHECK(tile_count64 > 0 && tile_count64 <= static_cast<int64_t>(INT_MAX), "tile count is too large");
    int tile_n_bits = 0;
    int64_t bit_value = tile_count64;
    while (bit_value > 0) {
        ++tile_n_bits;
        bit_value >>= 1;
    }

    at::Tensor dense_tile_ids = at::empty({n_isects64}, radii.options());
    at::Tensor flatten_ids = at::empty({n_isects64}, radii.options());
    at::Tensor isect_ids_hi = at::empty({n_isects64}, radii.options());
    at::Tensor isect_ids_lo = at::empty({n_isects64}, radii.options());
    const int64_t total = means2d.size(0) * means2d.size(1);
    TORCH_CHECK(total <= static_cast<int64_t>(INT_MAX), "metadata input is too large");
    if (total > 0 && n_isects64 > 0) {
        at::Tensor starts = at::empty({total}, radii.options());
        at::Tensor start_scratch = at::empty({total}, radii.options());
        suStream_t stream = current_supa_stream(means2d);
        suError_t err = gsplat_tiled_offsets_prefix_launch(
            tiles_per_gauss.const_data_ptr<int>(),
            static_cast<int>(total),
            starts.data_ptr<int>(),
            start_scratch.data_ptr<int>(),
            stream
        );
        TORCH_CHECK(err == suSuccess, "SUPA isect_tiles_alpha_unpacked_lanes starts failed: ", suGetErrorString(err));

        err = gsplat_isect_tiles_alpha_unpacked_lanes_launch(
            means2d.const_data_ptr<float>(),
            radii.const_data_ptr<int>(),
            depths.const_data_ptr<float>(),
            conics.const_data_ptr<float>(),
            opacities.const_data_ptr<float>(),
            tiles_per_gauss.const_data_ptr<int>(),
            starts.const_data_ptr<int>(),
            static_cast<float>(alpha_threshold),
            static_cast<int>(means2d.size(0)),
            static_cast<int>(means2d.size(1)),
            static_cast<int>(tile_size),
            static_cast<int>(tile_width),
            static_cast<int>(tile_height),
            tile_n_bits,
            dense_tile_ids.data_ptr<int>(),
            flatten_ids.data_ptr<int>(),
            isect_ids_hi.data_ptr<int>(),
            isect_ids_lo.data_ptr<int>(),
            stream
        );
        TORCH_CHECK(err == suSuccess, "SUPA isect_tiles_alpha_unpacked_lanes failed: ", suGetErrorString(err));
    }
    return std::make_tuple(dense_tile_ids, flatten_ids, isect_ids_hi, isect_ids_lo);
}

std::tuple<at::Tensor, at::Tensor>
tiled_offsets_from_lanes(
    const at::Tensor &dense_tile_ids,
    const at::Tensor &flatten_ids,
    const at::Tensor &isect_ids_lo,
    int64_t n_images,
    int64_t tile_width,
    int64_t tile_height
) {
    check_supa_int_tensor(dense_tile_ids, "dense_tile_ids");
    check_supa_int_tensor(flatten_ids, "flatten_ids");
    check_supa_int_tensor(isect_ids_lo, "isect_ids_lo");
    TORCH_CHECK(dense_tile_ids.dim() == 1, "dense_tile_ids must be 1D");
    TORCH_CHECK(flatten_ids.dim() == 1, "flatten_ids must be 1D");
    TORCH_CHECK(isect_ids_lo.dim() == 1, "isect_ids_lo must be 1D");
    TORCH_CHECK(flatten_ids.numel() == dense_tile_ids.numel(), "flatten_ids must match dense_tile_ids");
    TORCH_CHECK(isect_ids_lo.numel() == dense_tile_ids.numel(), "isect_ids_lo must match dense_tile_ids");
    TORCH_CHECK(n_images >= 0 && n_images <= static_cast<int64_t>(INT_MAX), "image count must fit int32");
    TORCH_CHECK(tile_width >= 0 && tile_width <= static_cast<int64_t>(INT_MAX), "tile_width must fit int32");
    TORCH_CHECK(tile_height >= 0 && tile_height <= static_cast<int64_t>(INT_MAX), "tile_height must fit int32");
    TORCH_CHECK(dense_tile_ids.numel() <= static_cast<int64_t>(INT_MAX), "intersection count is too large");
    const int64_t n_tiles64 = n_images * tile_width * tile_height;
    TORCH_CHECK(n_tiles64 >= 0 && n_tiles64 <= static_cast<int64_t>(INT_MAX), "tile count is too large");

    at::Tensor tile_offsets = at::zeros({n_images, tile_height, tile_width}, dense_tile_ids.options());
    at::Tensor sorted_flatten_ids = at::empty_like(flatten_ids);
    if (dense_tile_ids.numel() == 0 || n_tiles64 == 0) {
        return std::make_tuple(tile_offsets, sorted_flatten_ids);
    }

    at::Tensor tile_counts = at::zeros({n_tiles64}, dense_tile_ids.options());
    at::Tensor tile_cursors = at::zeros({n_tiles64}, dense_tile_ids.options());
    at::Tensor sorted_isect_ids_lo = at::empty_like(isect_ids_lo);
    at::Tensor source_orders = at::empty_like(flatten_ids);

    const int n_isects = static_cast<int>(dense_tile_ids.numel());
    const int n_tiles = static_cast<int>(n_tiles64);
    suStream_t stream = current_supa_stream(dense_tile_ids);
    suError_t err = gsplat_tiled_offsets_count_launch(
        dense_tile_ids.const_data_ptr<int>(),
        n_isects,
        n_tiles,
        tile_counts.data_ptr<int>(),
        stream
    );
    TORCH_CHECK(err == suSuccess, "SUPA tiled_offsets count failed: ", suGetErrorString(err));

    err = gsplat_tiled_offsets_prefix_launch(
        tile_counts.const_data_ptr<int>(),
        n_tiles,
        tile_offsets.data_ptr<int>(),
        tile_cursors.data_ptr<int>(),
        stream
    );
    TORCH_CHECK(err == suSuccess, "SUPA tiled_offsets prefix failed: ", suGetErrorString(err));

    err = gsplat_tiled_offsets_scatter_launch(
        dense_tile_ids.const_data_ptr<int>(),
        flatten_ids.const_data_ptr<int>(),
        isect_ids_lo.const_data_ptr<int>(),
        n_isects,
        n_tiles,
        tile_cursors.data_ptr<int>(),
        sorted_flatten_ids.data_ptr<int>(),
        sorted_isect_ids_lo.data_ptr<int>(),
        source_orders.data_ptr<int>(),
        stream
    );
    TORCH_CHECK(err == suSuccess, "SUPA tiled_offsets scatter failed: ", suGetErrorString(err));

    err = gsplat_tiled_offsets_sort_launch(
        sorted_flatten_ids.data_ptr<int>(),
        sorted_isect_ids_lo.data_ptr<int>(),
        source_orders.data_ptr<int>(),
        tile_offsets.const_data_ptr<int>(),
        n_tiles,
        n_isects,
        stream
    );
    TORCH_CHECK(err == suSuccess, "SUPA tiled_offsets sort failed: ", suGetErrorString(err));

    return std::make_tuple(tile_offsets, sorted_flatten_ids);
}

std::tuple<at::Tensor, at::Tensor> rasterize_to_pixels_tiled_fwd(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &features,
    const at::Tensor &opacities,
    const c10::optional<at::Tensor> &backgrounds,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids
) {
    check_supa_tensor(means2d, "means2d");
    check_supa_tensor(conics, "conics");
    check_supa_tensor(features, "features");
    check_supa_tensor(opacities, "opacities");
    check_supa_int_tensor(tile_offsets, "tile_offsets");
    check_supa_int_tensor(flatten_ids, "flatten_ids");
    if (backgrounds.has_value()) {
        check_supa_tensor(backgrounds.value(), "backgrounds");
    }
    TORCH_CHECK(means2d.dim() == 3 && means2d.size(-1) == 2, "means2d must have shape [I, N, 2]");
    TORCH_CHECK(conics.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1), 3}), "conics must have shape [I, N, 3]");
    TORCH_CHECK(features.dim() == 3, "features must have shape [I, N, channels]");
    TORCH_CHECK(features.size(0) == means2d.size(0) && features.size(1) == means2d.size(1), "features shape must match means2d");
    TORCH_CHECK(opacities.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1)}), "opacities must have shape [I, N]");
    TORCH_CHECK(tile_offsets.dim() == 3 && tile_offsets.size(0) == means2d.size(0), "tile_offsets must have shape [I, tile_height, tile_width]");
    if (backgrounds.has_value()) {
        TORCH_CHECK(
            backgrounds.value().sizes() == at::IntArrayRef({means2d.size(0), features.size(2)}),
            "backgrounds must have shape [I, channels]"
        );
    }
    TORCH_CHECK(tile_size > 0 && tile_size <= static_cast<int64_t>(INT_MAX), "tile_size must fit int32");
    TORCH_CHECK(image_width > 0 && image_width <= static_cast<int64_t>(INT_MAX), "image_width must fit int32");
    TORCH_CHECK(image_height > 0 && image_height <= static_cast<int64_t>(INT_MAX), "image_height must fit int32");
    TORCH_CHECK(tile_offsets.size(1) > 0 && tile_offsets.size(1) <= static_cast<int64_t>(INT_MAX), "tile_height must fit int32");
    TORCH_CHECK(tile_offsets.size(2) > 0 && tile_offsets.size(2) <= static_cast<int64_t>(INT_MAX), "tile_width must fit int32");
    TORCH_CHECK(means2d.size(0) <= static_cast<int64_t>(INT_MAX), "image count is too large");
    TORCH_CHECK(features.size(2) <= static_cast<int64_t>(INT_MAX), "channel count is too large");
    TORCH_CHECK(flatten_ids.size(0) <= static_cast<int64_t>(INT_MAX), "intersection count is too large");

    at::Tensor render = features.size(2) <= kAlphaCompositeMaxChannels
        ? at::empty({means2d.size(0), image_height, image_width, features.size(2)}, features.options())
        : at::zeros({means2d.size(0), image_height, image_width, features.size(2)}, features.options());
    at::Tensor alpha = at::empty({means2d.size(0), image_height, image_width, 1}, features.options());
    const int64_t total = means2d.size(0) * image_width * image_height;
    TORCH_CHECK(total <= static_cast<int64_t>(INT_MAX), "render output is too large");
    if (total > 0) {
        suError_t err = gsplat_rasterize_to_pixels_tiled_fwd_launch(
            means2d.const_data_ptr<float>(),
            conics.const_data_ptr<float>(),
            features.const_data_ptr<float>(),
            opacities.const_data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
            tile_offsets.const_data_ptr<int>(),
            flatten_ids.const_data_ptr<int>(),
            static_cast<int>(flatten_ids.size(0)),
            static_cast<int>(means2d.size(0)),
            static_cast<int>(features.size(2)),
            static_cast<int>(image_width),
            static_cast<int>(image_height),
            static_cast<int>(tile_offsets.size(2)),
            static_cast<int>(tile_offsets.size(1)),
            static_cast<int>(tile_size),
            render.data_ptr<float>(),
            alpha.data_ptr<float>(),
            current_supa_stream(means2d)
        );
        TORCH_CHECK(err == suSuccess, "SUPA rasterize_to_pixels_tiled_fwd failed: ", suGetErrorString(err));
    }
    return std::make_tuple(render, alpha);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_tiled_bwd(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &features,
    const at::Tensor &opacities,
    const c10::optional<at::Tensor> &backgrounds,
    const at::Tensor &v_render,
    const c10::optional<at::Tensor> &v_alpha,
    const c10::optional<at::Tensor> &forward_alpha,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    int64_t max_tile_range
) {
    check_supa_tensor(means2d, "means2d");
    check_supa_tensor(conics, "conics");
    check_supa_tensor(features, "features");
    check_supa_tensor(opacities, "opacities");
    check_supa_tensor(v_render, "v_render");
    if (v_alpha.has_value()) {
        check_supa_tensor(v_alpha.value(), "v_alpha");
    }
    if (forward_alpha.has_value()) {
        check_supa_tensor(forward_alpha.value(), "forward_alpha");
    }
    check_supa_int_tensor(tile_offsets, "tile_offsets");
    check_supa_int_tensor(flatten_ids, "flatten_ids");
    if (backgrounds.has_value()) {
        check_supa_tensor(backgrounds.value(), "backgrounds");
    }
    TORCH_CHECK(means2d.dim() == 3 && means2d.size(-1) == 2, "means2d must have shape [I, N, 2]");
    TORCH_CHECK(conics.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1), 3}), "conics must have shape [I, N, 3]");
    TORCH_CHECK(features.dim() == 3, "features must have shape [I, N, channels]");
    TORCH_CHECK(features.size(0) == means2d.size(0) && features.size(1) == means2d.size(1), "features shape must match means2d");
    TORCH_CHECK(opacities.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1)}), "opacities must have shape [I, N]");
    TORCH_CHECK(tile_offsets.dim() == 3 && tile_offsets.size(0) == means2d.size(0), "tile_offsets must have shape [I, tile_height, tile_width]");
    TORCH_CHECK(
        v_render.sizes() == at::IntArrayRef({means2d.size(0), image_height, image_width, features.size(2)}),
        "v_render must have shape [I, image_height, image_width, channels]"
    );
    if (v_alpha.has_value()) {
        TORCH_CHECK(
            v_alpha.value().sizes() == at::IntArrayRef({means2d.size(0), image_height, image_width, 1}),
            "v_alpha must have shape [I, image_height, image_width, 1]"
        );
    }
    if (forward_alpha.has_value()) {
        TORCH_CHECK(
            forward_alpha.value().sizes() == at::IntArrayRef({means2d.size(0), image_height, image_width, 1}),
            "forward_alpha must have shape [I, image_height, image_width, 1]"
        );
    }
    if (backgrounds.has_value()) {
        TORCH_CHECK(
            backgrounds.value().sizes() == at::IntArrayRef({means2d.size(0), features.size(2)}),
            "backgrounds must have shape [I, channels]"
        );
    }
    TORCH_CHECK(tile_size > 0 && tile_size <= static_cast<int64_t>(INT_MAX), "tile_size must fit int32");
    TORCH_CHECK(image_width > 0 && image_width <= static_cast<int64_t>(INT_MAX), "image_width must fit int32");
    TORCH_CHECK(image_height > 0 && image_height <= static_cast<int64_t>(INT_MAX), "image_height must fit int32");
    TORCH_CHECK(tile_offsets.size(1) > 0 && tile_offsets.size(1) <= static_cast<int64_t>(INT_MAX), "tile_height must fit int32");
    TORCH_CHECK(tile_offsets.size(2) > 0 && tile_offsets.size(2) <= static_cast<int64_t>(INT_MAX), "tile_width must fit int32");
    TORCH_CHECK(means2d.size(0) <= static_cast<int64_t>(INT_MAX), "image count is too large");
    TORCH_CHECK(features.size(2) <= kAlphaCompositeMaxChannels, "channel count must be at most 64 for rasterize_to_pixels_tiled_bwd");
    TORCH_CHECK(flatten_ids.size(0) <= static_cast<int64_t>(INT_MAX), "intersection count is too large");
    TORCH_CHECK(max_tile_range >= 0 && max_tile_range <= static_cast<int64_t>(INT_MAX), "max_tile_range must fit int32");

    at::Tensor v_means2d = at::zeros_like(means2d);
    at::Tensor v_conics = at::zeros_like(conics);
    at::Tensor v_features = at::zeros_like(features);
    at::Tensor v_opacities = at::zeros_like(opacities);
    at::Tensor v_backgrounds = backgrounds.has_value() ? at::zeros_like(backgrounds.value()) : at::empty({0}, features.options());

    const int64_t total = means2d.size(0) * image_width * image_height;
    TORCH_CHECK(total <= static_cast<int64_t>(INT_MAX), "render output is too large");
    const int64_t prefix_checkpoint_count =
        (max_tile_range + kAlphaCompositePrefixCheckpointInterval - 1) /
        kAlphaCompositePrefixCheckpointInterval;
    TORCH_CHECK(
        prefix_checkpoint_count == 0 ||
            total <= std::numeric_limits<int64_t>::max() / prefix_checkpoint_count,
        "rasterize_to_pixels_tiled_bwd prefix checkpoint workspace is too large"
    );
    TORCH_CHECK(prefix_checkpoint_count <= static_cast<int64_t>(INT_MAX), "prefix checkpoint count is too large");
    if (total > 0) {
        // Bound the temporary prefix workspace by processing pixels in chunks.
        // Kernels run on the current stream, so reusing this buffer is ordered.
        const int64_t target_workspace_elements =
            kAlphaCompositeBwdWorkspaceTargetBytes / static_cast<int64_t>(sizeof(float));
        const int64_t max_chunk_pixels = prefix_checkpoint_count > 0
            ? std::max<int64_t>(1, target_workspace_elements / prefix_checkpoint_count)
            : total;
        const int64_t chunk_pixels = std::min(total, max_chunk_pixels);
        TORCH_CHECK(chunk_pixels <= static_cast<int64_t>(INT_MAX), "pixel chunk is too large");
        at::Tensor prefix_checkpoints = prefix_checkpoint_count > 0
            ? at::empty({chunk_pixels, prefix_checkpoint_count}, features.options())
            : at::empty({0}, features.options());

        for (int64_t pixel_offset = 0; pixel_offset < total; pixel_offset += chunk_pixels) {
            const int64_t current_pixels = std::min(chunk_pixels, total - pixel_offset);
            suError_t err = gsplat_rasterize_to_pixels_tiled_bwd_launch(
                means2d.const_data_ptr<float>(),
                conics.const_data_ptr<float>(),
                features.const_data_ptr<float>(),
                opacities.const_data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
                tile_offsets.const_data_ptr<int>(),
                flatten_ids.const_data_ptr<int>(),
                v_render.const_data_ptr<float>(),
                v_alpha.has_value() ? v_alpha.value().const_data_ptr<float>() : nullptr,
                forward_alpha.has_value() ? forward_alpha.value().const_data_ptr<float>() : nullptr,
                static_cast<int>(flatten_ids.size(0)),
                static_cast<int>(means2d.size(0)),
                static_cast<int>(features.size(2)),
                static_cast<int>(image_width),
                static_cast<int>(image_height),
                static_cast<int>(tile_offsets.size(2)),
                static_cast<int>(tile_offsets.size(1)),
                static_cast<int>(tile_size),
                static_cast<int>(pixel_offset),
                static_cast<int>(current_pixels),
                prefix_checkpoint_count > 0 ? prefix_checkpoints.data_ptr<float>() : nullptr,
                static_cast<int>(prefix_checkpoint_count),
                v_means2d.data_ptr<float>(),
                v_conics.data_ptr<float>(),
                v_features.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                backgrounds.has_value() ? v_backgrounds.data_ptr<float>() : nullptr,
                current_supa_stream(means2d)
            );
            TORCH_CHECK(err == suSuccess, "SUPA rasterize_to_pixels_tiled_bwd failed: ", suGetErrorString(err));
        }
    }
    return std::make_tuple(v_means2d, v_conics, v_features, v_opacities, v_backgrounds);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_tiled_bwd_tile_reduce(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &features,
    const at::Tensor &opacities,
    const c10::optional<at::Tensor> &backgrounds,
    const at::Tensor &v_render,
    const c10::optional<at::Tensor> &v_alpha,
    const c10::optional<at::Tensor> &forward_alpha,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    int64_t max_tile_range
) {
    check_supa_tensor(means2d, "means2d");
    check_supa_tensor(conics, "conics");
    check_supa_tensor(features, "features");
    check_supa_tensor(opacities, "opacities");
    check_supa_tensor(v_render, "v_render");
    if (v_alpha.has_value()) {
        check_supa_tensor(v_alpha.value(), "v_alpha");
    }
    if (forward_alpha.has_value()) {
        check_supa_tensor(forward_alpha.value(), "forward_alpha");
    }
    check_supa_int_tensor(tile_offsets, "tile_offsets");
    check_supa_int_tensor(flatten_ids, "flatten_ids");
    if (backgrounds.has_value()) {
        check_supa_tensor(backgrounds.value(), "backgrounds");
    }
    TORCH_CHECK(means2d.dim() == 3 && means2d.size(-1) == 2, "means2d must have shape [I, N, 2]");
    TORCH_CHECK(conics.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1), 3}), "conics must have shape [I, N, 3]");
    TORCH_CHECK(features.dim() == 3, "features must have shape [I, N, channels]");
    TORCH_CHECK(features.size(0) == means2d.size(0) && features.size(1) == means2d.size(1), "features shape must match means2d");
    TORCH_CHECK(features.size(2) == 3, "tile-reduce tiled backward currently supports RGB channels only");
    TORCH_CHECK(opacities.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1)}), "opacities must have shape [I, N]");
    TORCH_CHECK(tile_offsets.dim() == 3 && tile_offsets.size(0) == means2d.size(0), "tile_offsets must have shape [I, tile_height, tile_width]");
    TORCH_CHECK(
        v_render.sizes() == at::IntArrayRef({means2d.size(0), image_height, image_width, 3}),
        "v_render must have shape [I, image_height, image_width, 3]"
    );
    if (v_alpha.has_value()) {
        TORCH_CHECK(
            v_alpha.value().sizes() == at::IntArrayRef({means2d.size(0), image_height, image_width, 1}),
            "v_alpha must have shape [I, image_height, image_width, 1]"
        );
    }
    if (forward_alpha.has_value()) {
        TORCH_CHECK(
            forward_alpha.value().sizes() == at::IntArrayRef({means2d.size(0), image_height, image_width, 1}),
            "forward_alpha must have shape [I, image_height, image_width, 1]"
        );
    }
    if (backgrounds.has_value()) {
        TORCH_CHECK(
            backgrounds.value().sizes() == at::IntArrayRef({means2d.size(0), 3}),
            "backgrounds must have shape [I, 3]"
        );
    }
    TORCH_CHECK(tile_size == kTiledBwdTileReduceSize, "tile-reduce tiled backward requires tile_size=16");
    TORCH_CHECK(image_width > 0 && image_width <= static_cast<int64_t>(INT_MAX), "image_width must fit int32");
    TORCH_CHECK(image_height > 0 && image_height <= static_cast<int64_t>(INT_MAX), "image_height must fit int32");
    TORCH_CHECK(tile_offsets.size(1) > 0 && tile_offsets.size(1) <= static_cast<int64_t>(INT_MAX), "tile_height must fit int32");
    TORCH_CHECK(tile_offsets.size(2) > 0 && tile_offsets.size(2) <= static_cast<int64_t>(INT_MAX), "tile_width must fit int32");
    TORCH_CHECK(means2d.size(0) <= static_cast<int64_t>(INT_MAX), "image count is too large");
    TORCH_CHECK(flatten_ids.size(0) <= static_cast<int64_t>(INT_MAX), "intersection count is too large");
    TORCH_CHECK(max_tile_range >= 0 && max_tile_range <= static_cast<int64_t>(INT_MAX), "max_tile_range must fit int32");

    at::Tensor v_means2d = at::zeros_like(means2d);
    at::Tensor v_conics = at::zeros_like(conics);
    at::Tensor v_features = at::zeros_like(features);
    at::Tensor v_opacities = at::zeros_like(opacities);
    at::Tensor v_backgrounds = backgrounds.has_value() ? at::zeros_like(backgrounds.value()) : at::empty({0}, features.options());

    const int64_t tile_count_total = tile_offsets.numel();
    TORCH_CHECK(tile_count_total <= static_cast<int64_t>(INT_MAX), "tile count is too large");
    const int64_t prefix_checkpoint_count =
        (max_tile_range + kAlphaCompositePrefixCheckpointInterval - 1) /
        kAlphaCompositePrefixCheckpointInterval;
    TORCH_CHECK(prefix_checkpoint_count <= static_cast<int64_t>(INT_MAX), "prefix checkpoint count is too large");
    if (tile_count_total > 0) {
        const int64_t target_workspace_elements =
            kAlphaCompositeBwdWorkspaceTargetBytes / static_cast<int64_t>(sizeof(float));
        const int64_t tile_workspace_elements =
            prefix_checkpoint_count * static_cast<int64_t>(kTiledBwdTileReducePixels);
        const int64_t max_chunk_tiles = tile_workspace_elements > 0
            ? std::max<int64_t>(1, target_workspace_elements / tile_workspace_elements)
            : tile_count_total;
        const int64_t chunk_tiles = std::min(tile_count_total, max_chunk_tiles);
        TORCH_CHECK(chunk_tiles <= static_cast<int64_t>(INT_MAX), "tile chunk is too large");
        at::Tensor prefix_checkpoints = prefix_checkpoint_count > 0
            ? at::empty({chunk_tiles, kTiledBwdTileReducePixels, prefix_checkpoint_count}, features.options())
            : at::empty({0}, features.options());

        for (int64_t tile_offset = 0; tile_offset < tile_count_total; tile_offset += chunk_tiles) {
            const int64_t current_tiles = std::min(chunk_tiles, tile_count_total - tile_offset);
            suError_t err = gsplat_rasterize_to_pixels_tiled_bwd_tile_reduce_launch(
                means2d.const_data_ptr<float>(),
                conics.const_data_ptr<float>(),
                features.const_data_ptr<float>(),
                opacities.const_data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
                tile_offsets.const_data_ptr<int>(),
                flatten_ids.const_data_ptr<int>(),
                v_render.const_data_ptr<float>(),
                v_alpha.has_value() ? v_alpha.value().const_data_ptr<float>() : nullptr,
                forward_alpha.has_value() ? forward_alpha.value().const_data_ptr<float>() : nullptr,
                static_cast<int>(flatten_ids.size(0)),
                static_cast<int>(means2d.size(0)),
                static_cast<int>(image_width),
                static_cast<int>(image_height),
                static_cast<int>(tile_offsets.size(2)),
                static_cast<int>(tile_offsets.size(1)),
                static_cast<int>(tile_size),
                static_cast<int>(tile_offset),
                static_cast<int>(current_tiles),
                prefix_checkpoint_count > 0 ? prefix_checkpoints.data_ptr<float>() : nullptr,
                static_cast<int>(prefix_checkpoint_count),
                v_means2d.data_ptr<float>(),
                v_conics.data_ptr<float>(),
                v_features.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                backgrounds.has_value() ? v_backgrounds.data_ptr<float>() : nullptr,
                current_supa_stream(means2d)
            );
            TORCH_CHECK(err == suSuccess, "SUPA rasterize_to_pixels_tiled_bwd_tile_reduce failed: ", suGetErrorString(err));
        }
    }
    return std::make_tuple(v_means2d, v_conics, v_features, v_opacities, v_backgrounds);
}

std::tuple<at::Tensor, at::Tensor> alpha_composite_fwd(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &features,
    const at::Tensor &opacities,
    const at::Tensor &depths,
    const at::Tensor &radii,
    const c10::optional<at::Tensor> &sorted_ids,
    const c10::optional<at::Tensor> &backgrounds,
    int64_t image_width,
    int64_t image_height
) {
    check_supa_tensor(means2d, "means2d");
    check_supa_tensor(conics, "conics");
    check_supa_tensor(features, "features");
    check_supa_tensor(opacities, "opacities");
    check_supa_tensor(depths, "depths");
    check_supa_int_tensor(radii, "radii");
    if (sorted_ids.has_value()) {
        check_supa_int_tensor(sorted_ids.value(), "sorted_ids");
    }
    if (backgrounds.has_value()) {
        check_supa_tensor(backgrounds.value(), "backgrounds");
    }
    TORCH_CHECK(means2d.dim() == 3 && means2d.size(-1) == 2, "means2d must have shape [I, N, 2]");
    TORCH_CHECK(conics.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1), 3}), "conics must have shape [I, N, 3]");
    TORCH_CHECK(features.dim() == 3, "features must have shape [I, N, channels]");
    TORCH_CHECK(features.size(0) == means2d.size(0) && features.size(1) == means2d.size(1), "features shape must match means2d");
    TORCH_CHECK(opacities.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1)}), "opacities must have shape [I, N]");
    TORCH_CHECK(depths.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1)}), "depths must have shape [I, N]");
    TORCH_CHECK(radii.sizes() == means2d.sizes(), "radii must have shape [I, N, 2]");
    if (sorted_ids.has_value()) {
        TORCH_CHECK(
            sorted_ids.value().sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1)}),
            "sorted_ids must have shape [I, N]"
        );
    }
    if (backgrounds.has_value()) {
        TORCH_CHECK(
            backgrounds.value().sizes() == at::IntArrayRef({means2d.size(0), features.size(2)}),
            "backgrounds must have shape [I, channels]"
        );
    }
    TORCH_CHECK(means2d.size(0) <= static_cast<int64_t>(INT_MAX), "image count is too large");
    TORCH_CHECK(means2d.size(1) <= static_cast<int64_t>(INT_MAX), "gaussian count is too large");
    TORCH_CHECK(features.size(2) <= static_cast<int64_t>(INT_MAX), "channel count is too large");
    TORCH_CHECK(image_width <= static_cast<int64_t>(INT_MAX), "image_width is too large");
    TORCH_CHECK(image_height <= static_cast<int64_t>(INT_MAX), "image_height is too large");

    at::Tensor render = features.size(2) <= kAlphaCompositeMaxChannels
        ? at::empty({means2d.size(0), image_height, image_width, features.size(2)}, features.options())
        : at::zeros({means2d.size(0), image_height, image_width, features.size(2)}, features.options());
    at::Tensor alpha = at::empty({means2d.size(0), image_height, image_width, 1}, features.options());
    const int64_t total = means2d.size(0) * image_width * image_height;
    TORCH_CHECK(total <= static_cast<int64_t>(INT_MAX), "render output is too large");
    if (total > 0) {
        suError_t err = gsplat_alpha_composite_fwd_launch(
            means2d.const_data_ptr<float>(),
            conics.const_data_ptr<float>(),
            features.const_data_ptr<float>(),
            opacities.const_data_ptr<float>(),
            depths.const_data_ptr<float>(),
            radii.const_data_ptr<int>(),
            sorted_ids.has_value() ? sorted_ids.value().const_data_ptr<int>() : nullptr,
            backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
            static_cast<int>(means2d.size(0)),
            static_cast<int>(means2d.size(1)),
            static_cast<int>(features.size(2)),
            static_cast<int>(image_width),
            static_cast<int>(image_height),
            render.data_ptr<float>(),
            alpha.data_ptr<float>(),
            current_supa_stream(means2d)
        );
        TORCH_CHECK(err == suSuccess, "SUPA alpha_composite_fwd failed: ", suGetErrorString(err));
    }
    return std::make_tuple(render, alpha);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
alpha_composite_bwd(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &features,
    const at::Tensor &opacities,
    const at::Tensor &depths,
    const at::Tensor &radii,
    const c10::optional<at::Tensor> &sorted_ids,
    const c10::optional<at::Tensor> &backgrounds,
    const at::Tensor &v_render,
    const at::Tensor &v_alpha,
    int64_t image_width,
    int64_t image_height
) {
    check_supa_tensor(means2d, "means2d");
    check_supa_tensor(conics, "conics");
    check_supa_tensor(features, "features");
    check_supa_tensor(opacities, "opacities");
    check_supa_tensor(depths, "depths");
    check_supa_int_tensor(radii, "radii");
    if (sorted_ids.has_value()) {
        check_supa_int_tensor(sorted_ids.value(), "sorted_ids");
    }
    check_supa_tensor(v_render, "v_render");
    check_supa_tensor(v_alpha, "v_alpha");
    if (backgrounds.has_value()) {
        check_supa_tensor(backgrounds.value(), "backgrounds");
    }
    TORCH_CHECK(means2d.dim() == 3 && means2d.size(-1) == 2, "means2d must have shape [I, N, 2]");
    TORCH_CHECK(conics.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1), 3}), "conics must have shape [I, N, 3]");
    TORCH_CHECK(features.dim() == 3, "features must have shape [I, N, channels]");
    TORCH_CHECK(features.size(0) == means2d.size(0) && features.size(1) == means2d.size(1), "features shape must match means2d");
    TORCH_CHECK(opacities.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1)}), "opacities must have shape [I, N]");
    TORCH_CHECK(depths.sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1)}), "depths must have shape [I, N]");
    TORCH_CHECK(radii.sizes() == means2d.sizes(), "radii must have shape [I, N, 2]");
    if (sorted_ids.has_value()) {
        TORCH_CHECK(
            sorted_ids.value().sizes() == at::IntArrayRef({means2d.size(0), means2d.size(1)}),
            "sorted_ids must have shape [I, N]"
        );
    }
    TORCH_CHECK(
        v_render.sizes() == at::IntArrayRef({means2d.size(0), image_height, image_width, features.size(2)}),
        "v_render must have shape [I, image_height, image_width, channels]"
    );
    TORCH_CHECK(
        v_alpha.sizes() == at::IntArrayRef({means2d.size(0), image_height, image_width, 1}),
        "v_alpha must have shape [I, image_height, image_width, 1]"
    );
    if (backgrounds.has_value()) {
        TORCH_CHECK(
            backgrounds.value().sizes() == at::IntArrayRef({means2d.size(0), features.size(2)}),
            "backgrounds must have shape [I, channels]"
        );
    }
    TORCH_CHECK(means2d.size(0) <= static_cast<int64_t>(INT_MAX), "image count is too large");
    TORCH_CHECK(means2d.size(1) <= static_cast<int64_t>(INT_MAX), "gaussian count is too large");
    TORCH_CHECK(features.size(2) <= 64, "channel count must be at most 64 for alpha_composite_bwd");
    TORCH_CHECK(image_width <= static_cast<int64_t>(INT_MAX), "image_width is too large");
    TORCH_CHECK(image_height <= static_cast<int64_t>(INT_MAX), "image_height is too large");

    at::Tensor v_means2d = at::zeros_like(means2d);
    at::Tensor v_conics = at::zeros_like(conics);
    at::Tensor v_features = at::zeros_like(features);
    at::Tensor v_opacities = at::zeros_like(opacities);
    at::Tensor v_depths = at::zeros_like(depths);
    at::Tensor v_backgrounds = backgrounds.has_value() ? at::zeros_like(backgrounds.value()) : at::empty({0}, features.options());

    const int64_t total = means2d.size(0) * image_width * image_height;
    TORCH_CHECK(total <= static_cast<int64_t>(INT_MAX), "render output is too large");
    const int64_t prefix_checkpoint_count =
        (means2d.size(1) + kAlphaCompositePrefixCheckpointInterval - 1) /
        kAlphaCompositePrefixCheckpointInterval;
    TORCH_CHECK(
        prefix_checkpoint_count == 0 ||
            total <= std::numeric_limits<int64_t>::max() / prefix_checkpoint_count,
        "alpha_composite_bwd prefix checkpoint workspace is too large"
    );
    TORCH_CHECK(prefix_checkpoint_count <= static_cast<int64_t>(INT_MAX), "prefix checkpoint count is too large");
    if (total > 0 && prefix_checkpoint_count > 0) {
        const int64_t target_workspace_elements =
            kAlphaCompositeBwdWorkspaceTargetBytes / static_cast<int64_t>(sizeof(float));
        const int64_t max_chunk_pixels =
            std::max<int64_t>(1, target_workspace_elements / prefix_checkpoint_count);
        const int64_t chunk_pixels = std::min(total, max_chunk_pixels);
        TORCH_CHECK(chunk_pixels <= static_cast<int64_t>(INT_MAX), "pixel chunk is too large");
        at::Tensor prefix_checkpoints = at::empty({chunk_pixels, prefix_checkpoint_count}, features.options());

        for (int64_t pixel_offset = 0; pixel_offset < total; pixel_offset += chunk_pixels) {
            const int64_t current_pixels = std::min(chunk_pixels, total - pixel_offset);
            suError_t err = gsplat_alpha_composite_bwd_launch(
                means2d.const_data_ptr<float>(),
                conics.const_data_ptr<float>(),
                features.const_data_ptr<float>(),
                opacities.const_data_ptr<float>(),
                depths.const_data_ptr<float>(),
                radii.const_data_ptr<int>(),
                sorted_ids.has_value() ? sorted_ids.value().const_data_ptr<int>() : nullptr,
                backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
                v_render.const_data_ptr<float>(),
                v_alpha.const_data_ptr<float>(),
                static_cast<int>(means2d.size(0)),
                static_cast<int>(means2d.size(1)),
                static_cast<int>(features.size(2)),
                static_cast<int>(image_width),
                static_cast<int>(image_height),
                static_cast<int>(pixel_offset),
                static_cast<int>(current_pixels),
                prefix_checkpoints.data_ptr<float>(),
                static_cast<int>(prefix_checkpoint_count),
                v_means2d.data_ptr<float>(),
                v_conics.data_ptr<float>(),
                v_features.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                v_depths.data_ptr<float>(),
                backgrounds.has_value() ? v_backgrounds.data_ptr<float>() : nullptr,
                current_supa_stream(means2d)
            );
            TORCH_CHECK(err == suSuccess, "SUPA alpha_composite_bwd failed: ", suGetErrorString(err));
        }
    }
    return std::make_tuple(v_means2d, v_conics, v_features, v_opacities, v_depths, v_backgrounds);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "quat_scale_to_covar_preci_fwd",
        torch::wrap_pybind_function(quat_scale_to_covar_preci_fwd),
        "quat_scale_to_covar_preci_fwd"
    );
    m.def(
        "quat_scale_to_covar_preci_bwd",
        torch::wrap_pybind_function(quat_scale_to_covar_preci_bwd),
        "quat_scale_to_covar_preci_bwd"
    );
    m.def(
        "projection_ewa_3dgs_pinhole_fwd",
        torch::wrap_pybind_function(projection_ewa_3dgs_pinhole_fwd),
        "projection_ewa_3dgs_pinhole_fwd"
    );
    m.def(
        "projection_ewa_3dgs_pinhole_bwd",
        torch::wrap_pybind_function(projection_ewa_3dgs_pinhole_bwd),
        "projection_ewa_3dgs_pinhole_bwd"
    );
    m.def(
        "spherical_harmonics_fwd",
        torch::wrap_pybind_function(spherical_harmonics_fwd),
        "spherical_harmonics_fwd"
    );
    m.def(
        "spherical_harmonics_bwd",
        torch::wrap_pybind_function(spherical_harmonics_bwd),
        "spherical_harmonics_bwd"
    );
    m.def(
        "tile_count_unpacked",
        torch::wrap_pybind_function(tile_count_unpacked),
        "tile_count_unpacked"
    );
    m.def(
        "tile_count_alpha_unpacked",
        torch::wrap_pybind_function(tile_count_alpha_unpacked),
        "tile_count_alpha_unpacked"
    );
    m.def(
        "visible_compact_ids",
        torch::wrap_pybind_function(visible_compact_ids),
        "visible_compact_ids"
    );
    m.def(
        "compact_select_fwd",
        torch::wrap_pybind_function(compact_select_fwd),
        "compact_select_fwd"
    );
    m.def(
        "compact_select_bwd",
        torch::wrap_pybind_function(compact_select_bwd),
        "compact_select_bwd"
    );
    m.def(
        "isect_tiles_unpacked_lanes",
        torch::wrap_pybind_function(isect_tiles_unpacked_lanes),
        "isect_tiles_unpacked_lanes"
    );
    m.def(
        "isect_tiles_unpacked_lanes_from_counts",
        torch::wrap_pybind_function(isect_tiles_unpacked_lanes_from_counts),
        "isect_tiles_unpacked_lanes_from_counts"
    );
    m.def(
        "isect_tiles_alpha_unpacked_lanes_from_counts",
        torch::wrap_pybind_function(isect_tiles_alpha_unpacked_lanes_from_counts),
        "isect_tiles_alpha_unpacked_lanes_from_counts"
    );
    m.def(
        "tiled_offsets_from_lanes",
        torch::wrap_pybind_function(tiled_offsets_from_lanes),
        "tiled_offsets_from_lanes"
    );
    m.def(
        "rasterize_to_pixels_tiled_fwd",
        torch::wrap_pybind_function(rasterize_to_pixels_tiled_fwd),
        "rasterize_to_pixels_tiled_fwd"
    );
    m.def(
        "rasterize_to_pixels_tiled_bwd",
        torch::wrap_pybind_function(rasterize_to_pixels_tiled_bwd),
        "rasterize_to_pixels_tiled_bwd"
    );
    m.def(
        "rasterize_to_pixels_tiled_bwd_tile_reduce",
        torch::wrap_pybind_function(rasterize_to_pixels_tiled_bwd_tile_reduce),
        "rasterize_to_pixels_tiled_bwd_tile_reduce"
    );
    m.def(
        "alpha_composite_fwd",
        torch::wrap_pybind_function(alpha_composite_fwd),
        "alpha_composite_fwd"
    );
    m.def(
        "alpha_composite_bwd",
        torch::wrap_pybind_function(alpha_composite_bwd),
        "alpha_composite_bwd"
    );
}
