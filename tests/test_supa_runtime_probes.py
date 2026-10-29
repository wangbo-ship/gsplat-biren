# SPDX-License-Identifier: Apache-2.0

"""Focused SUPA runtime behavior probes promoted to tests."""

from __future__ import annotations

import pytest
import torch


def _supa_device() -> torch.device:
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pytest.skip("torch_br is not installed")
    if not hasattr(torch, "supa") or not torch.supa.is_available():
        pytest.skip("SUPA device is not available")
    return torch.device("supa:0")


def test_supa_int32_metadata_lanes_roundtrip_and_packed_int64_does_not():
    device = _supa_device()
    high = torch.tensor([0, 1, 7, 1023, 131071], dtype=torch.int32)
    low_u32_bits = torch.tensor(
        [0x00000000, 0x3F800000, 0x40000000, 0x7F7FFFFF, 0xBF800000],
        dtype=torch.int64,
    )
    low = low_u32_bits.to(torch.int32)
    packed = (high.to(torch.int64) << 32) | (low.to(torch.int64) & 0xFFFFFFFF)

    high_roundtrip = high.to(device).cpu()
    low_roundtrip = low.to(device).cpu()
    packed_roundtrip = packed.to(device).cpu()
    repacked = (high_roundtrip.to(torch.int64) << 32) | (
        low_roundtrip.to(torch.int64) & 0xFFFFFFFF
    )

    assert torch.equal(high, high_roundtrip)
    assert torch.equal(low, low_roundtrip)
    assert torch.equal(packed, repacked)
    assert not torch.equal(packed, packed_roundtrip)
