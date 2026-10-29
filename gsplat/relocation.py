# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
from torch import Tensor


def compute_relocation(
    opacities: Tensor,  # [N]
    scales: Tensor,  # [N, 3]
    ratios: Tensor,  # [N]
    binoms: Tensor,  # [n_max, n_max]
) -> Tuple[Tensor, Tensor]:
    """Compute new Gaussians from a set of old Gaussians.

    This function interprets the Gaussians as samples from a likelihood distribution.
    It uses the old opacities and scales to compute the new opacities and scales.
    This is an implementation of the paper
    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/pdf/2404.09591>`_,

    Args:
        opacities: The opacities of the Gaussians. [N]
        scales: The scales of the Gaussians. [N, 3]
        ratios: The relative frequencies for each of the Gaussians. [N]
        binoms: Precomputed lookup table for binomial coefficients used in
          Equation 9 in the paper. [n_max, n_max]

    Returns:
        A tuple:

        **new_opacities**: The opacities of the new Gaussians. [N]
        **new_scales**: The scales of the Gaussians. [N, 3]
    """

    n = opacities.shape[0]
    n_max, _ = binoms.shape
    assert scales.shape == (n, 3), scales.shape
    assert ratios.shape == (n,), ratios.shape

    ratios = ratios.clamp(min=1, max=n_max).to(dtype=torch.long)
    new_opacities = 1.0 - torch.pow(1.0 - opacities, 1.0 / ratios.to(opacities.dtype))

    denom_sum = torch.zeros_like(opacities)
    sqrt_denoms = torch.sqrt(
        torch.arange(1, n_max + 1, device=opacities.device, dtype=opacities.dtype)
    )
    binoms = binoms.to(device=opacities.device, dtype=opacities.dtype)
    for i in range(1, n_max + 1):
        active = ratios >= i
        if not bool(active.any()):
            break
        for k in range(i):
            sign = -1.0 if k % 2 else 1.0
            coeff = binoms[i - 1, k]
            term = sign / sqrt_denoms[k] * torch.pow(new_opacities, k + 1)
            denom_sum = torch.where(active, denom_sum + coeff * term, denom_sum)
    coeff = opacities / denom_sum.clamp_min(torch.finfo(opacities.dtype).eps)
    new_scales = coeff[:, None] * scales
    return new_opacities, new_scales
