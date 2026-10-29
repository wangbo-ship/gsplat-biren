# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused distributed tests for Biren SUPA collectives."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distF
import torch.multiprocessing as mp

from gsplat.distributed import all_gather_tensor_list


_SPLAT_KEYS = ("means", "quats", "scales", "opacities", "colors")


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return int(port)


def _require_supa_devices(count: int) -> int:
    try:
        import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        pytest.skip("torch_br is not installed")

    if not hasattr(torch, "supa") or not torch.supa.is_available():
        pytest.skip("SUPA device is not available")
    device_count = int(torch.supa.device_count())
    if device_count < count:
        pytest.skip(f"need at least {count} SUPA devices, got {device_count}")
    return count


def _supa_all_gather_tensor_list_worker(rank: int, world_size: int, port: int) -> None:
    import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.supa.set_device(rank)
    device = torch.device("supa", rank)
    dist.init_process_group("sccl", rank=rank, world_size=world_size)

    try:
        values = [
            torch.full((2, 2), float(rank + 1), device=device, requires_grad=True),
            torch.full((2, 1), float(rank + 3), device=device, requires_grad=True),
        ]
        gathered = all_gather_tensor_list(world_size, values)

        expected = [
            torch.cat(
                [
                    torch.full((2, 2), float(peer + 1), device=device)
                    for peer in range(world_size)
                ]
            ),
            torch.cat(
                [
                    torch.full((2, 1), float(peer + 3), device=device)
                    for peer in range(world_size)
                ]
            ),
        ]
        for got, want in zip(gathered, expected):
            torch.testing.assert_close(got, want)

        loss = gathered[0].square().sum() + 0.5 * gathered[1].square().sum()
        loss.backward()

        torch.testing.assert_close(
            values[0].grad,
            torch.full_like(values[0], 2.0 * world_size * float(rank + 1)),
        )
        torch.testing.assert_close(
            values[1].grad,
            torch.full_like(values[1], world_size * float(rank + 3)),
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_supa_all_gather_tensor_list_forward_backward():
    world_size = _require_supa_devices(2)
    mp.spawn(
        _supa_all_gather_tensor_list_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
    )


def _supa_dist_functional_probe_worker(rank: int, world_size: int, port: int) -> None:
    import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.supa.set_device(rank)
    device = torch.device("supa", rank)
    dist.init_process_group("sccl", rank=rank, world_size=world_size)

    try:
        x = torch.full((2, 3), float(rank + 1), device=device, requires_grad=True)
        gathered = distF.all_gather(x)
        gathered_tensor = torch.cat(gathered, dim=0)
        assert tuple(gathered_tensor.shape) == (world_size * 2, 3)
        expected_gathered = torch.cat(
            [
                torch.full((2, 3), float(peer + 1), device=device)
                for peer in range(world_size)
            ],
            dim=0,
        )
        torch.testing.assert_close(gathered_tensor, expected_gathered)

        send0 = torch.full((rank + 1, 2), float(rank), device=device)
        send1 = torch.full((2 - rank, 2), float(rank), device=device)
        recv = [
            torch.empty((rank + 1, 2), device=device),
            torch.empty((2 - rank, 2), device=device),
        ]
        dist.all_to_all(recv, [send0, send1])
        expected_sums = [0.0, 4.0] if rank == 0 else [0.0, 2.0]
        assert [tuple(tensor.shape) for tensor in recv] == [
            (rank + 1, 2),
            (2 - rank, 2),
        ]
        torch.testing.assert_close(
            torch.tensor([float(tensor.cpu().sum().item()) for tensor in recv]),
            torch.tensor(expected_sums),
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_supa_dist_functional_all_gather_and_all_to_all_probe():
    world_size = _require_supa_devices(2)
    mp.spawn(
        _supa_dist_functional_probe_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
    )


def _supa_dist_functional_backward_probe_worker(
    rank: int, world_size: int, port: int
) -> None:
    import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.supa.set_device(rank)
    device = torch.device("supa", rank)
    dist.init_process_group("sccl", rank=rank, world_size=world_size)

    try:
        x = torch.full((2, 3), float(rank + 1), device=device, requires_grad=True)
        gathered = distF.all_gather(x)
        torch.cat(gathered, dim=0).square().sum().backward()
        torch.testing.assert_close(
            x.grad,
            torch.full_like(x, 2.0 * world_size * float(rank + 1)),
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.xfail(
    reason=(
        "BRPyTorch distF.all_gather backward currently routes through SCCL "
        "all-to-all, which is not reliable on this stack."
    ),
    strict=False,
)
def test_supa_dist_functional_all_gather_backward_probe():
    world_size = _require_supa_devices(2)
    mp.spawn(
        _supa_dist_functional_backward_probe_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
    )


def _supa_raw_collectives_probe_worker(rank: int, world_size: int, port: int) -> None:
    import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.supa.set_device(rank)
    device = torch.device("supa", rank)
    dist.init_process_group("sccl", rank=rank, world_size=world_size)

    try:
        x = torch.full((2, 3), float(rank + 1), device=device)
        gathered = torch.empty((world_size, 2, 3), device=device)
        dist.all_gather_into_tensor(gathered, x)
        torch.testing.assert_close(
            gathered,
            torch.stack(
                [
                    torch.full((2, 3), float(peer + 1), device=device)
                    for peer in range(world_size)
                ],
                dim=0,
            ),
        )

        reduced = x.clone()
        dist.all_reduce(reduced)
        torch.testing.assert_close(reduced, torch.full_like(reduced, 3.0))

        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_supa_raw_sccl_collectives_probe():
    world_size = _require_supa_devices(2)
    mp.spawn(
        _supa_raw_collectives_probe_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
    )


def _supa_raw_all_to_all_probe_worker(rank: int, world_size: int, port: int) -> None:
    import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.supa.set_device(rank)
    device = torch.device("supa", rank)
    dist.init_process_group("sccl", rank=rank, world_size=world_size)

    try:
        out_equal_base = [torch.empty((6,), device=device) for _ in range(world_size)]
        in_equal_base = [
            torch.full((6,), float(rank), device=device),
            torch.full((6,), float(rank + 10), device=device),
        ]
        dist.all_to_all(
            [tensor.view_as_buffer() for tensor in out_equal_base],
            [tensor.view_as_buffer() for tensor in in_equal_base],
        )
        expected_equal = [0.0, 6.0] if rank == 0 else [60.0, 66.0]
        torch.testing.assert_close(
            torch.tensor([float(tensor.cpu().sum().item()) for tensor in out_equal_base]),
            torch.tensor(expected_equal),
        )

        in_var = [
            torch.full((rank + 1, 2), float(rank), device=device),
            torch.full((2 - rank, 2), float(rank), device=device),
        ]
        out_var = [
            torch.empty((rank + 1, 2), device=device),
            torch.empty((2 - rank, 2), device=device),
        ]
        dist.all_to_all(out_var, in_var)
        expected_var = [0.0, 4.0] if rank == 0 else [0.0, 2.0]
        torch.testing.assert_close(
            torch.tensor([float(tensor.cpu().sum().item()) for tensor in out_var]),
            torch.tensor(expected_var),
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.xfail(
    reason="SCCL all-to-all remains less trusted than all-gather/all-reduce.",
    strict=False,
)
def test_supa_raw_sccl_all_to_all_probe():
    world_size = _require_supa_devices(2)
    mp.spawn(
        _supa_raw_all_to_all_probe_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
    )


def _all_to_all_buffer(data, splits, output_splits):
    input_chunks = [
        chunk.clone().contiguous().view_as_buffer()
        for chunk in data.split(splits, dim=0)
    ]
    output_base = [
        torch.empty((length, *data.shape[1:]), dtype=data.dtype, device=data.device)
        for length in output_splits
    ]
    output_chunks = [chunk.view_as_buffer() for chunk in output_base]
    dist.all_to_all(output_chunks, input_chunks)
    return torch.cat(output_base, dim=0)


class _GatherWithAllReduceBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        ctx.world_size = world_size
        ctx.rank = rank
        out = torch.empty((world_size, *x.shape), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out.flatten(0, 1)

    @staticmethod
    def backward(ctx, grad_output):
        chunks = grad_output.reshape(ctx.world_size, -1, *grad_output.shape[1:])
        grad = chunks[ctx.rank].contiguous()
        dist.all_reduce(grad)
        return grad


class _AllToAllBuffer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, splits_tensor, output_splits_tensor):
        splits = [int(value) for value in splits_tensor.cpu().tolist()]
        output_splits = [int(value) for value in output_splits_tensor.cpu().tolist()]
        ctx.splits = splits
        ctx.output_splits = output_splits
        return _all_to_all_buffer(x, splits, output_splits)

    @staticmethod
    def backward(ctx, grad_output):
        grad = _all_to_all_buffer(
            grad_output.contiguous(),
            ctx.output_splits,
            ctx.splits,
        )
        return grad, None, None


def _supa_dist_custom_autograd_probe_worker(rank: int, world_size: int, port: int) -> None:
    import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.supa.set_device(rank)
    device = torch.device("supa", rank)
    dist.init_process_group("sccl", rank=rank, world_size=world_size)

    try:
        x = torch.full((2, 3), float(rank + 1), device=device, requires_grad=True)
        y = _GatherWithAllReduceBackward.apply(x)
        torch.testing.assert_close(
            y,
            torch.cat(
                [
                    torch.full((2, 3), float(peer + 1), device=device)
                    for peer in range(world_size)
                ],
                dim=0,
            ),
        )
        y.square().sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 6.0))

        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_supa_dist_custom_autograd_collectives_probe():
    world_size = _require_supa_devices(2)
    mp.spawn(
        _supa_dist_custom_autograd_probe_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
    )


def _supa_dist_custom_all_to_all_autograd_probe_worker(
    rank: int, world_size: int, port: int
) -> None:
    import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.supa.set_device(rank)
    device = torch.device("supa", rank)
    dist.init_process_group("sccl", rank=rank, world_size=world_size)

    try:
        splits = [rank + 1, 2 - rank]
        split_tensor = torch.tensor(splits, dtype=torch.int32, device=device)
        output_split_tensor = torch.tensor(splits, dtype=torch.int32, device=device)
        z = torch.arange(sum(splits) * 2, dtype=torch.float32, device=device).reshape(
            sum(splits),
            2,
        )
        z = (z + rank * 100).requires_grad_(True)
        w = _AllToAllBuffer.apply(z, split_tensor, output_split_tensor)
        assert tuple(w.shape) == (sum(splits), 2)
        w.square().sum().backward()
        torch.testing.assert_close(z.grad, 2.0 * z.detach())
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.xfail(
    reason="SCCL all-to-all autograd probes remain unreliable on this stack.",
    strict=False,
)
def test_supa_dist_custom_all_to_all_autograd_probe():
    world_size = _require_supa_devices(2)
    mp.spawn(
        _supa_dist_custom_all_to_all_autograd_probe_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
    )


def _make_distributed_scene(device: torch.device) -> dict[str, torch.Tensor]:
    tensors = {
        "means": [
            [-0.22, -0.12, 2.0],
            [0.12, 0.04, 2.3],
            [0.02, 0.22, 2.8],
            [0.25, -0.08, 3.1],
        ],
        "quats": [
            [1.0, 0.0, 0.0, 0.0],
            [0.98, 0.05, 0.1, 0.02],
            [0.9, 0.2, 0.1, 0.05],
            [0.95, -0.02, 0.08, 0.12],
        ],
        "scales": [
            [0.08, 0.07, 0.09],
            [0.1, 0.08, 0.07],
            [0.06, 0.09, 0.08],
            [0.07, 0.06, 0.11],
        ],
        "opacities": [0.7, 0.55, 0.65, 0.5],
        "colors": [
            [0.9, 0.1, 0.2],
            [0.2, 0.8, 0.1],
            [0.1, 0.3, 0.9],
            [0.8, 0.6, 0.2],
        ],
    }
    scene = {
        name: torch.tensor(value, dtype=torch.float32, device=device)
        for name, value in tensors.items()
    }

    viewmat0 = torch.eye(4, dtype=torch.float32).to(device)
    viewmat1 = torch.eye(4, dtype=torch.float32).to(device)
    viewmat1[0, 3] = 0.08
    scene["viewmats"] = torch.stack((viewmat0, viewmat1), dim=0)
    scene["Ks"] = torch.tensor(
        [
            [[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]],
            [[8.5, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
        device=device,
    )
    return scene


def _clone_scene_params(scene: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: scene[name].detach().clone().requires_grad_(True)
        for name in _SPLAT_KEYS
    }


def _call_basic_rasterization(
    params: dict[str, torch.Tensor],
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    *,
    distributed: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    from gsplat import rasterization  # pylint: disable=import-outside-toplevel

    return rasterization(
        params["means"],
        params["quats"],
        params["scales"],
        params["opacities"],
        params["colors"],
        viewmats,
        Ks,
        width=8,
        height=8,
        packed=False,
        render_mode="RGB",
        distributed=distributed,
    )


def _supa_distributed_rasterization_worker(
    rank: int, world_size: int, port: int
) -> None:
    import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["GSPLAT_USE_SUPA_NATIVE"] = "0"
    torch.supa.set_device(rank)
    device = torch.device("supa", rank)
    dist.init_process_group("sccl", rank=rank, world_size=world_size)

    try:
        scene = _make_distributed_scene(device)
        ref_params = _clone_scene_params(scene)
        ref_render, ref_alpha, _ = _call_basic_rasterization(
            ref_params,
            scene["viewmats"],
            scene["Ks"],
            distributed=False,
        )
        ref_loss = ref_render.square().mean() + 0.25 * ref_alpha.square().mean()
        ref_loss.backward()

        start = rank * 2
        stop = start + 2
        dist_params = {
            name: scene[name][start:stop].detach().clone().requires_grad_(True)
            for name in _SPLAT_KEYS
        }
        render, alpha, meta = _call_basic_rasterization(
            dist_params,
            scene["viewmats"][rank : rank + 1],
            scene["Ks"][rank : rank + 1],
            distributed=True,
        )
        torch.testing.assert_close(render, ref_render[rank : rank + 1])
        torch.testing.assert_close(alpha, ref_alpha[rank : rank + 1])
        assert meta["distributed"] is True
        assert meta["distributed_backend"] == "supa_gather"
        assert meta["N_world"] == [2, 2]

        # Scale the local-camera loss to match the full two-camera reference mean.
        loss = (render.square().mean() + 0.25 * alpha.square().mean()) / world_size
        loss.backward()

        for name in _SPLAT_KEYS:
            torch.testing.assert_close(
                dist_params[name].grad.cpu(),
                ref_params[name].grad[start:stop].cpu(),
                rtol=3e-3,
                atol=1e-4,
            )

        no_grad_params = {
            name: scene[name][start:stop].detach().clone() for name in _SPLAT_KEYS
        }
        with torch.no_grad():
            no_grad_render, no_grad_alpha, no_grad_meta = _call_basic_rasterization(
                no_grad_params,
                scene["viewmats"][rank : rank + 1],
                scene["Ks"][rank : rank + 1],
                distributed=True,
            )
        torch.testing.assert_close(no_grad_render, ref_render[rank : rank + 1])
        torch.testing.assert_close(no_grad_alpha, ref_alpha[rank : rank + 1])
        assert no_grad_meta["distributed_backend"] == "supa_gather"

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _supa_distributed_rasterization_empty_shard_worker(
    rank: int, world_size: int, port: int
) -> None:
    import torch_br  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["GSPLAT_USE_SUPA_NATIVE"] = "0"
    torch.supa.set_device(rank)
    device = torch.device("supa", rank)
    dist.init_process_group("sccl", rank=rank, world_size=world_size)

    try:
        scene = _make_distributed_scene(device)
        ref_params = {
            name: scene[name][:1].detach().clone().requires_grad_(True)
            for name in _SPLAT_KEYS
        }
        ref_render, ref_alpha, _ = _call_basic_rasterization(
            ref_params,
            scene["viewmats"],
            scene["Ks"],
            distributed=False,
        )
        ref_loss = ref_render.square().mean() + 0.25 * ref_alpha.square().mean()
        ref_loss.backward()

        start = 0 if rank == 0 else 1
        stop = 1 if rank == 0 else 1
        dist_params = {
            name: scene[name][start:stop].detach().clone().requires_grad_(True)
            for name in _SPLAT_KEYS
        }
        render, alpha, meta = _call_basic_rasterization(
            dist_params,
            scene["viewmats"][rank : rank + 1],
            scene["Ks"][rank : rank + 1],
            distributed=True,
        )
        torch.testing.assert_close(render, ref_render[rank : rank + 1])
        torch.testing.assert_close(alpha, ref_alpha[rank : rank + 1])
        assert meta["distributed"] is True
        assert meta["distributed_backend"] == "supa_gather"
        assert meta["N_world"] == [1, 0]

        loss = (render.square().mean() + 0.25 * alpha.square().mean()) / world_size
        loss.backward()

        for name in _SPLAT_KEYS:
            if rank == 0:
                torch.testing.assert_close(
                    dist_params[name].grad.cpu(),
                    ref_params[name].grad.cpu(),
                    rtol=3e-3,
                    atol=1e-4,
                )
            elif dist_params[name].grad is not None:
                assert dist_params[name].grad.numel() == 0

        no_grad_params = {
            name: scene[name][start:stop].detach().clone() for name in _SPLAT_KEYS
        }
        with torch.no_grad():
            no_grad_render, no_grad_alpha, no_grad_meta = _call_basic_rasterization(
                no_grad_params,
                scene["viewmats"][rank : rank + 1],
                scene["Ks"][rank : rank + 1],
                distributed=True,
            )
        torch.testing.assert_close(no_grad_render, ref_render[rank : rank + 1])
        torch.testing.assert_close(no_grad_alpha, ref_alpha[rank : rank + 1])
        assert no_grad_meta["N_world"] == [1, 0]

        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_supa_distributed_rasterization_matches_full_reference():
    world_size = _require_supa_devices(2)
    mp.spawn(
        _supa_distributed_rasterization_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
    )


def test_supa_distributed_rasterization_allows_empty_shard():
    world_size = _require_supa_devices(2)
    mp.spawn(
        _supa_distributed_rasterization_empty_shard_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
    )
