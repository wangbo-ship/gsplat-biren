# gsplat-biren

`gsplat-biren` is a Biren SUPA / BRPyTorch fork of upstream `gsplat` focused on
reliable basic 3D Gaussian Splatting training and rendering on SUPA devices.
The feature surface is intentionally frozen around the maintained SUPA path so
future work can focus on correctness, stability, and performance improvements.

## Current Surface

Supported and actively maintained:

- Basic unpacked 3DGS rasterization through `gsplat.rasterization()` for
  `pinhole`, `ortho`, and `fisheye` camera models.
- PyTorch fallback execution on CPU and SUPA tensors.
- Optional native SUPA acceleration for selected hot paths.
- Single-card SUPA training with `applications/supa_multiview_trainer.py`.
- Single-node SCCL distributed SUPA training on the hardened trainer path.
- NPZ and COLMAP trainer inputs.
- Train crops, SH color training, densification/pruning, opacity reset,
  checkpointing, PLY export, and simple per-frame exposure refinement.

Rejected by the frozen public surface:

- `packed=True`, sparse gradients, AbsGrad, segmented rendering, UT/Eval3D,
  lidar, camera distortion, rolling shutter, rays, hit-distance render modes,
  normals, viewer parity, and fused selective optimizer APIs.

The old accelerator extension tree is not part of this fork. Reference
math/projection/tile helpers now live in top-level pure-PyTorch modules such as
`gsplat/_math.py` and `gsplat/_torch_impl.py`.

## Environment

Use the fixed Biren-provided Python stack. Do not replace system PyTorch,
BRPyTorch, `torch_br`, SUPA SDK, drivers, or globally installed packages from
inside this repository.

For local commands:

```bash
source env.sh
```

`env.sh` adds BRCC to `PATH`, keeps the checkout on `PYTHONPATH`, and redirects
Python caches, native builds, and temporary files under `.agents/temporary/`.
Generated datasets, logs, benchmarks, and training outputs should stay under
the allowed `.agents/` subdirectories: `dataset/`, `reference/`, `temporary/`,
and `output/`.

Import `torch_br` before constructing or using SUPA tensors:

```python
import torch
import torch_br  # registers torch.supa

device = torch.device("supa:0")
tensor = torch.ones(4).to(device)
```

The root `setup.py` is only package metadata for this fork. It is not the main
development workflow.

## Main Entry Points

- `gsplat/rendering.py`: public SUPA-focused `rasterization()` shim.
- `gsplat/_device.py`: SUPA runtime, device, synchronization, cache, and SCCL
  helper functions.
- `gsplat/_torch_fallback.py`: differentiable PyTorch fallback plus optional
  native SUPA dispatch.
- `gsplat/supa/`: optional native SUPA kernels, bindings, JIT loader, and
  autograd wrappers.
- `applications/supa_multiview_trainer.py`: SUPA-safe multiview trainer.
- `tools/diagnose_clean_colmap.py`: COLMAP sparse diagnostics and cleanup.
- `tools/clean_3dgs_ply.py`: exported 3DGS PLY cleanup.
- `benchmark/supa_rasterization.py` and `benchmark/supa_operators.py`: SUPA
  benchmarks.

## Native SUPA Controls

Training native behavior is controlled by CLI arguments and `TrainConfig`, not
environment variables:

- `--native true|false`
- `--native-tiled false|true|auto`
- `--native-tiled-metadata true|false`
- `--native-tiled-train true|false`
- `--native-tiled-backward-mode auto|pixel|tile-reduce|parallel`
- `--native-alpha-threshold <value>`

`--native true` selects the supported native trainer profile: tiled training,
native tiled metadata, `native_tiled=auto`, the standard `1/255` alpha cutoff,
and safe `parallel` tiled backward preference. Pass
`--native-alpha-threshold 0` for exact current SUPA tiled metadata comparisons.
Native projection acceleration is currently for `pinhole`; `ortho` and
`fisheye` public rasterization use the PyTorch fallback projection.

## Training

Minimal single-card COLMAP run using the canonical local dataset shape:

```bash
source env.sh
/usr/bin/python3 applications/supa_multiview_trainer.py \
  --data .agents/dataset/minecraft_dungle_pyramid_undistorted \
  --data-type colmap \
  --data-factor 4 \
  --device supa:0 \
  --iterations 100 \
  --init-num-points -1 \
  --auto-output-root .agents/output/train
```

Native profile example:

```bash
source env.sh
/usr/bin/python3 applications/supa_multiview_trainer.py \
  --data .agents/dataset/minecraft_dungle_pyramid_undistorted \
  --data-type colmap \
  --data-factor 4 \
  --device supa:0 \
  --iterations 100 \
  --init-num-points 65535 \
  --native true \
  --auto-output-root .agents/output/train
```

Use `--result-dir` for an exact output directory, or `--auto-output-root` for
an automatically named child directory. Project CLI booleans use explicit
`true|false` values.

## Validation

Common structural and correctness checks:

```bash
source env.sh
/usr/bin/python3 -m compileall -q gsplat applications tools benchmark tests
/usr/bin/python3 -m pytest -q tests/test_supa_runtime_probes.py
/usr/bin/python3 -m pytest -q tests/test_supa_fallback.py
/usr/bin/python3 -m pytest -q tests/test_supa_multiview_trainer.py
/usr/bin/python3 -m pytest -q tests/test_supa_distributed.py
```

Project-wide smoke:

```bash
source env.sh
/usr/bin/python3 -m pytest -q \
  -o cache_dir=.agents/temporary/pytest-cache \
  --basetemp=.agents/temporary/pytest-tmp/full
```

Rendering and training quality gates:

```bash
source env.sh
/usr/bin/python3 tools/render_correctness_gate.py
/usr/bin/python3 tools/training_quality_gate.py --run true
```

The fixed training gate is `quality-gate-v2`: one 100-step COLMAP run on
`.agents/dataset/minecraft_dungle_pyramid_undistorted` with crop 256,
`--native true`, and outputs under `.agents/output/train/quality-gate-v2/`.

## Benchmarks

Write benchmark JSON under `.agents/output/perf/`.

```bash
source env.sh
/usr/bin/python3 benchmark/supa_rasterization.py \
  --scene synthetic \
  --gaussians 4096 \
  --width 128 \
  --height 96 \
  --mode train \
  --json-out .agents/output/perf/rasterization_bench.json
```

```bash
source env.sh
/usr/bin/python3 benchmark/supa_operators.py \
  --bench sh,compact,composite,metadata,public-train \
  --json-out .agents/output/perf/operator_bench.json
```

## Development Rules

- Use `/usr/bin/python3` directly.
- Source `env.sh` for local training, tests, validation, and benchmarks.
- Use `gsplat._logging` for user-visible command output.
- Use helpers in `gsplat/_device.py` for SUPA runtime and device handling.
- Keep generated artifacts under `.agents/temporary/` or `.agents/output/`.
- Commit coherent milestones with messages like `type(scope): detail`.

## Upstream

This project is derived from `nerfstudio-project/gsplat`. The maintained
surface in this repository targets Biren SUPA.
