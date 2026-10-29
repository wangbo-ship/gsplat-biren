# SPDX-License-Identifier: Apache-2.0

"""Optimizer namespace kept for compatibility.

The upstream fused selective optimizer depended on an extension outside the
frozen SUPA-only surface.
Use standard PyTorch optimizers in the SUPA-only fork.
"""

__all__: list[str] = []
