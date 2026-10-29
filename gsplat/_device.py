# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device helpers for Biren SUPA execution paths."""

from __future__ import annotations

import os
from typing import Any, Optional, Union

import torch
from torch import Tensor

DeviceLike = Optional[Union[str, int, torch.device]]

SUPA_DEVICE_TYPE = "supa"


def ensure_supa_runtime() -> Any:
    """Import torch_br and return torch.supa after the PrivateUse1 bridge is ready."""
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "Biren SUPA support requires the torch_br package from BRPyTorch."
        ) from exc

    if not hasattr(torch, SUPA_DEVICE_TYPE):
        raise RuntimeError("torch_br was imported but torch.supa is not registered.")
    return torch.supa


def supa_available() -> bool:
    """Return whether the Biren SUPA backend is importable and has visible devices."""
    try:
        supa = ensure_supa_runtime()
    except RuntimeError:
        return False
    return bool(supa.is_available())


def _looks_like_supa(device: DeviceLike) -> bool:
    if isinstance(device, str):
        return device == SUPA_DEVICE_TYPE or device.startswith(f"{SUPA_DEVICE_TYPE}:")
    if isinstance(device, torch.device):
        return device.type == SUPA_DEVICE_TYPE
    return False


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Resolve a device spec, importing torch_br first when a SUPA device is requested."""
    if device is None:
        return torch.device(default_accelerator_type())
    if isinstance(device, int):
        return torch.device(default_accelerator_type(), device)
    if _looks_like_supa(device):
        ensure_supa_runtime()
    return torch.device(device)


def device_type(device: DeviceLike = None) -> str:
    """Return the canonical device type for a device spec."""
    if device is None:
        return default_accelerator_type()
    return resolve_device(device).type


def is_supa_device(device_or_tensor: Union[DeviceLike, Tensor]) -> bool:
    """Return whether a device spec or tensor lives on Biren SUPA."""
    if isinstance(device_or_tensor, Tensor):
        return device_or_tensor.device.type == SUPA_DEVICE_TYPE
    return _looks_like_supa(device_or_tensor)


def is_accelerator_tensor(tensor: Tensor) -> bool:
    """Return whether a tensor is on a supported SUPA accelerator."""
    return tensor.device.type == SUPA_DEVICE_TYPE


def default_accelerator_type() -> str:
    """Return the default Biren SUPA accelerator type."""
    if supa_available():
        return SUPA_DEVICE_TYPE
    raise RuntimeError("No Biren SUPA accelerator is available.")


def accelerator_device_count(device: DeviceLike = None) -> int:
    """Return the visible SUPA device count."""
    dtype = device_type(device)
    if dtype == SUPA_DEVICE_TYPE:
        return ensure_supa_runtime().device_count()
    raise ValueError(f"Unsupported accelerator device type: {dtype}")


def set_accelerator_device(device: DeviceLike) -> None:
    """Set the active SUPA device."""
    resolved = resolve_device(device)
    index = 0 if resolved.index is None else resolved.index
    if resolved.type == SUPA_DEVICE_TYPE:
        ensure_supa_runtime().set_device(index)
    else:
        raise ValueError(f"Unsupported accelerator device type: {resolved.type}")


def synchronize(device: DeviceLike = None) -> None:
    """Synchronize the selected SUPA accelerator, or all SUPA devices when omitted."""
    if device is None:
        if supa_available():
            ensure_supa_runtime().synchronize()
        return

    dtype = device_type(device)
    if dtype == SUPA_DEVICE_TYPE:
        ensure_supa_runtime().synchronize()


def empty_cache(device: DeviceLike = None) -> None:
    """Release cached SUPA allocator blocks when the backend exposes it."""
    dtype = device_type(device)
    if dtype == SUPA_DEVICE_TYPE:
        supa = ensure_supa_runtime()
        if hasattr(supa, "empty_cache"):
            supa.empty_cache()


def distributed_backend(device: DeviceLike = None) -> str:
    """Return the process-group backend name matching SUPA."""
    dtype = device_type(device)
    if dtype == SUPA_DEVICE_TYPE:
        return "sccl"
    raise ValueError(f"Unsupported distributed device type: {dtype}")


def torch_fallback_enabled(device: torch.device) -> bool:
    """Return whether gsplat should use its PyTorch fallback for this device."""
    override = os.getenv("GSPLAT_USE_TORCH_FALLBACK")
    if override is not None:
        return override == "1"
    return device.type in {SUPA_DEVICE_TYPE, "cpu"}
