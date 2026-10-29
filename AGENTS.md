# Agent Memory for gsplat-biren

This file is the durable working memory for agents in this repository. Keep it
English, concrete, and easy to extend. Add new facts under the closest existing
section instead of rewriting the whole file.

## 0. High-Frequency Rules

- Read this file first before making repository changes.
- Work inside the fixed existing environment. Never add, remove, modify,
  upgrade, downgrade, reinstall, or otherwise mutate system Python packages,
  system Python environments, system software packages, drivers, SDKs, or
  globally installed tools.
- Keep newly discovered durable project knowledge in this file. When a task
  reveals an important convention, runtime quirk, validation result, or
  gotcha that future agents should know, append it promptly under the closest
  section.
- After every major quality or performance change, regenerate the fixed
  quality/performance baseline and record the result in this file under
  Validation Records before treating the work as complete.
- Treat README.md as non-authoritative for working decisions. When a change
  affects user-facing setup, workflows, or documented behavior, update
  README.md when practical so it does not drift further from the code.
- When AGENTS.md information is stale or contradicted by verified local
  reality, update AGENTS.md with the true current information under the
  closest existing section.
- Code comments must be in English and should be complete enough for future
  maintainers to understand non-obvious control flow, SUPA/BRPyTorch quirks,
  tensor shape contracts, native dispatch choices, and validation assumptions.
  Follow the style of nearby code: concise comments for ordinary code, more
  explicit comments where SUPA behavior is surprising.
- Add necessary log output for user-visible workflows, probes, benchmarks,
  training, native fallback decisions, and validation gates. Match existing
  logging/printing style in the touched module and avoid noisy logs in hot
  tensor paths.
- User-visible command output must go through `gsplat._logging`, not raw
  `print()`. Use `get_logger()` plus `log_info()`/`log_warning()`/`log_error()`
  and the ASCII table helpers when output has structured fields. CLI entry
  points that need immediate output should call `configure_logging()` at the
  start of `main()`. Keep tests and docstring examples exempt unless the task
  specifically targets them.
- Root `setup.py` is only a placeholder for this fork. Do not treat it as the
  main workflow or run it unless a task explicitly targets packaging metadata.
  The primary working entry point is the training script under `applications/`.
- `.agents/` has exactly four allowed top-level subdirectories:
  `.agents/dataset/` for datasets, `.agents/reference/` for reference
  material, `.agents/temporary/` for caches/native builds/temp files/probes,
  and `.agents/output/` for durable run outputs such as training results,
  benchmark JSON, comparison artifacts, correctness gates, logs, and reports.
  Do not create other `.agents/*` top-level directories.
- This repository is managed with git. After code changes pass the appropriate
  tests and validation, commit at a suitable granularity. Do not mix unrelated
  layers in one commit, for example Python fallback changes, native SUPA
  kernels, trainer behavior, docs, and tests should be split when they are
  independently meaningful.
- Do not create standalone commits for handoff-only AGENTS.md updates. The
  existing handoff-only commit `5619b8e docs(agents): record handoff probe`
  should be amended into the next substantive repository commit instead of
  remaining as a separate history entry.
- Commit messages must follow `type(sphere): detail`, for example
  `fix(supa): materialize projection gradients`.

## 1. Project Mission

- `gsplat-biren` is a Biren SUPA / BRPyTorch fork of upstream `gsplat`.
- The active goal is reliable basic 3D Gaussian Splatting training and
  rendering on Biren SUPA devices.
- Correctness, reproducibility, and clear validation records currently matter
  more than upstream feature parity or maximum performance.
- Quality/performance baseline stability is an engineering gate for major
  changes, not a standalone roadmap direction. Future roadmap summaries should
  assume baseline generation and AGENTS.md recording are part of the normal
  delivery process.
- SUPA support is layered on top of the existing Python API where practical,
  but the maintained public surface is now explicitly SUPA-focused.
- Native SUPA kernels are optional acceleration paths. The safe baseline is the
  differentiable PyTorch fallback running on `torch_br` SUPA tensors.

## 2. Non-Negotiable Environment Rules

- Do not replace, reinstall, or mutate the system-provided Biren Python stack,
  PyTorch, BRPyTorch, `torch_br`, SUPA SDK, or driver packages.
- Do not modify any system Python package or system software package in any
  way. This includes installs, uninstalls, upgrades, downgrades, edits, cache
  rewrites, and global configuration changes. All work must stay within the
  fixed pre-existing environment and repository-local files.
- Users may manually install missing lightweight Python dependencies with
  `/usr/bin/python3 -m pip install pycolmap pytest -i https://pypi.tuna.tsinghua.edu.cn/simple`
  when they choose to mutate their own environment. AI agents are still not
  allowed to run pip or otherwise mutate the system Python environment.
- Prefer `/usr/bin/python3` for local commands unless the user explicitly says
  otherwise.
- For local training, benchmarking, and validation commands, source the root
  environment file first: `source env.sh`.
- Import `torch_br` before constructing or using SUPA tensors. `torch_br`
  registers the PyTorch `PrivateUse1` backend as `supa` and exposes
  `torch.supa`.
- Use explicit SUPA devices such as `torch.device("supa:0")` or the helpers in
  `gsplat/_device.py`; do not assume CUDA semantics.
- Native SUPA JIT paths need BRCC on `PATH`, normally:
  `PATH=/usr/local/birensupa/sdk/latest/brcc/bin:$PATH`.
- `env.sh` sets `PATH` for BRCC, keeps the checkout and
  `.agents/temporary/python-local` on `PYTHONPATH`, and redirects caches/builds
  into `.agents/temporary/`:
  `PYTHONPYCACHEPREFIX=$PWD/.agents/temporary/pycache`,
  `TORCH_EXTENSIONS_DIR=$PWD/.agents/temporary/native-build`,
  `XDG_CACHE_HOME=$PWD/.agents/temporary/cache`, and
  `TMPDIR=$PWD/.agents/temporary/tmp`.
- `pytest --basetemp=.agents/temporary/pytest-tmp/<name>` requires the parent
  `.agents/temporary/pytest-tmp` to exist first; `env.sh` creates it.
- `gsplat/supa/_backend.py` defaults native JIT builds to
  `.agents/temporary/native-build/supa_native_build`; do not redirect native
  extension builds into repository root, system cache directories, or
  user-global caches.

## 3. Repository Map

- `gsplat/rendering.py`: public SUPA-focused rasterization API. It keeps a
  broad compatibility signature, defaults `packed=False`, dispatches to the
  PyTorch fallback, and rejects unsupported frozen-surface options clearly.
- `gsplat/_device.py`: SUPA runtime/device helpers. Use this module for
  runtime import, device resolution, synchronization, cache clearing, device
  count, and distributed backend selection.
- `gsplat/_constants.py`, `gsplat/_math.py`, `gsplat/_torch_impl.py`, and
  `gsplat/_types.py`: top-level pure-PyTorch/reference helpers and public
  compatibility types used after removing the old accelerator extension tree.
- `gsplat/_torch_fallback.py`: SUPA-safe differentiable PyTorch implementation
  for basic rasterization, plus opt-in native SUPA dispatch for hot paths.
- `gsplat/supa/_backend.py`: cached JIT loader for the optional native SUPA
  extension.
- `gsplat/supa/_wrapper.py`: Python/autograd wrappers for native SUPA kernels.
  It performs dtype, shape, contiguity, and device gating before calling native
  code, and returns `None` for unsupported cases so callers can fall back.
- `gsplat/supa/binding.cpp`: pybind11 C++ bridge. It validates tensors with
  `TORCH_CHECK`, gets the current SUPA stream, passes raw pointers to launch
  wrappers, and checks `suError_t`.
- `gsplat/supa/kernels.su`: SUPA C++ kernels and `extern "C"` launch wrappers.
- `applications/supa_multiview_trainer.py`: SUPA-safe multiview 3DGS trainer
  for NPZ/COLMAP data, single-card training, and single-node SCCL training.
- `applications/datasets/`: dataset adapters used by SUPA applications.
- `gsplat/distributed.py`: SUPA distributed helpers. SUPA uses SCCL and custom
  gather backward paths where PyTorch distributed primitives are not reliable
  enough.
- `benchmark/`: SUPA rasterization and operator benchmarks. Store output JSON
  under `.agents/output/perf/`.
- `benchmark/supa_rasterization.py --mode tiled-bwd` supports
  `--tiled-bwd-modes pixel,auto,parallel` to benchmark several backward modes
  in one run, and rasterization metadata benchmarks record `max_tile_range`.
- `tools/`: dataset conversion, render correctness, and training quality gates.
- `tools/convert_nerf_synthetic_to_npz.py`: converts NeRF synthetic
  `transforms_*.json` datasets, such as `.agents/dataset/lego`, to the trainer
  NPZ format.
- `tools/diagnose_clean_colmap.py`: diagnoses COLMAP sparse reconstruction
  quality, reports low-observation/camera-outlier images and point-cloud
  outliers, writes JSON/manifests, and can write a cleaned sparse model by
  deleting selected 3D points. It can also deregister flagged images when
  explicitly run with `--clean-images true`.
- `tools/clean_3dgs_ply.py`: cleans exported standard vertex-only 3DGS PLY
  files by oversized scale, spatial KNN outlier distance, or a collaborative
  score that combines large and isolated splats. It treats `scale_*` as
  log-scale values by default, matching this trainer/exporter; pass
  `--scale-domain raw` for PLY files with linear scale magnitudes.
- `tools/compare_train_supa_fallback.py`,
  `tools/compare_train_supa_native.py`, and
  `tools/compare_backend_training_common.py`: SUPA-only COLMAP training
  diagnostic scripts. Defaults now target
  `.agents/dataset/minecraft_dungle_pyramid_undistorted`.
- `tests/test_supa_fallback.py`: main SUPA fallback/native correctness suite.
- `tests/test_supa_operator_guard.py`: combined public rasterization
  correctness/speed guard for native full-scan and tiled-train profiles.
- `tests/test_supa_multiview_trainer.py`: trainer, checkpoint, refinement,
  CLI, and SUPA smoke coverage.
- `tests/test_supa_distributed.py`: SCCL/distributed probes and rasterization.
- `tests/test_supa_runtime_probes.py`: runtime behavior probes, including
  metadata lane behavior.

## 4. SUPA Programming Notes from the Local Guides

Reference docs live in `.agents/reference/doc/supa_guide/`. The points below are
the parts most relevant to this repo.

### 4.1 Programming Model

- SUPA is a host/device programming model with C/C++ language extensions,
  runtime APIs, driver APIs, BRCC, and device libraries.
- Device kernels are `void` functions marked `__global__` for G-Mode or
  `__global_mega__` for T-Mode.
- Host code launches kernels through runtime APIs such as `suLaunchKernel()`.
  SUPA does not use CUDA `<<<...>>>` launch syntax.
- `suLaunchKernel(kernel, grid, block, dynamic_shared_mem, stream, args...)`
  takes the function pointer, grid shape, block shape, dynamic shared memory
  size, stream, then kernel arguments.
- Kernel launches and async copies are asynchronous by default. Use
  `suDeviceSynchronize()` or `suStreamSynchronize()` when correctness or timing
  requires completion.
- `SUPA_LAUNCH_BLOCKING=1` forces launch blocking and is useful for debugging,
  but it should not be used as a performance path.

### 4.2 G-Mode Threads and Memory

- Use SUPA built-ins `grid_dim`, `block_idx`, `block_dim`, and `thread_idx`
  instead of CUDA names `gridDim`, `blockIdx`, `blockDim`, and `threadIdx`.
- A warp has 32 threads. Prefer block sizes that are multiples of 32.
- G-Mode block size max is 1024 threads.
- Thread blocks can execute in any order. Never depend on inter-block ordering.
- Common linear index pattern:
  `idx = block_idx.x * block_dim.x + thread_idx.x`, followed by an OOB guard.
- Memory hierarchy: thread-local memory (TLM), group/shared memory (GSM),
  global memory (GLM), and constant memory (CM).
- `__shared__` memory is not zero-initialized on SUPA. Initialize it explicitly
  and synchronize before consumers read it.
- Use `__syncthreads()` for G-Mode thread-block barriers. Use T-Mode specific
  synchronization only in T-Mode code.

### 4.3 T-Mode and Tensor Notes

- T-Mode uses `__global_mega__` and maps a block to an SPC.
- T-Mode thread count must be a multiple of 512, from 512 to 4096. Larger
  blocks require `__launch_bounds__(THREAD_NUM)`.
- Useful T-Mode built-ins include `warp_idx`, `warp_count`,
  `warp_thread_idx`, and `warp_size`.
- Tensor APIs live under the SUPA tensor namespace and use layouts such as
  Vectors, Matrix, Matrix3D, and Activation. Tensor warp load/store APIs use
  `__short_vector`.
- Warp tensor load/store granularity is commonly 128 bytes per warp
  (32 threads * 4 bytes), with alignment requirements.
- This repo's current native kernels are G-Mode scalar kernels, not T-Mode
  tensor kernels. Do not introduce T-Mode unless the task specifically needs it
  and there is test coverage.

### 4.4 Runtime, Streams, and Memory

- Allocate device memory with `suMallocDevice()` and release it with `suFree()`
  in raw SUPA programs. In this PyTorch extension, normally allocate outputs
  as ATen tensors on the input SUPA device.
- `suMemcpy()` and `suMemcpyAsync()` can infer copy direction with
  `suMemcpyDefault`.
- Streams are ordered queues. Operations in one stream run in order; different
  streams may overlap.
- Each device has default streams. Treat stream ownership carefully; a stream
  must match the current device for kernel launch.
- Events can measure elapsed time and coordinate streams.
- Some operations cause implicit synchronization, including device allocation,
  pinned host allocation/registration, device memset, same-device memory copy,
  and device configuration changes.

### 4.5 CUDA Compatibility Traps

- SUPA is CUDA-like, not CUDA-identical.
- Runtime APIs use `su*` names; driver APIs use `sudrv*` names.
- Macros include `__SUPA__`, `__BRCC__`, and `__SUPA_ARCH__`.
- Device-side `double` is not supported in the documented target. Avoid double
  in SUPA kernels.
- int64/uint64 math support is limited on hardware generation 1.x. This repo
  uses int32 metadata lanes for native SUPA tile data.
- 3D thread-block warp construction differs from CUDA: SUPA warps do not cross
  the Z dimension. Be careful when porting CUDA kernels that rely on linear
  warp layout for 3D blocks.
- `__shfl_sync()` with divergent `srcLane` must pass an explicit `width`.
- BF16 bit operations can be surprising; convert to FP32 for bit-style logic.
- Atomics have memory-type and device-scope limits. Do not use atomics to host
  memory, peer memory, unsupported 4K memory styles, or cross-chip locations.
- If multiple lanes in one warp write the same address without atomics, the
  final value is undefined and may not equal any lane's source value.
- SUPA does not support CUDA texture/surface memory, CUDA managed memory,
  device-side runtime API calls, kernel callbacks, CUDA dynamic parallelism, or
  CUDA WMMA. Use SUPA-native tensor/T-Core APIs if tensor-core work is needed.

### 4.6 Debugging and Performance

- Check every runtime return value in native host code. Clear stale errors with
  `suGetLastError()` when necessary.
- `BR_UMD_TRACE_LEVEL=1` enables UMD trace diagnostics for runtime misuse,
  allocation errors, and OOB reports.
- `supa-sanitizer --tool synccheck|memcheck|racecheck` can check synchronization,
  memory, and shared-memory race issues. Debug builds need `brcc -g -G`; add
  `--supa-device-opt` if optimized debug compilation is needed.
- `brSimulator` can run via `BR_UMD_BACKEND=simulator`.
- `brmsg` helps analyze kernel logs after hangs or health-check resets.
- Use events and warmups for timing. First launch/JIT includes overhead and is
  not representative.
- Tune by first deciding whether the kernel is memory-bound or compute-bound,
  then increase parallelism, balance work across SPC/CU/EU, optimize memory
  access, and finally reduce instruction overhead.

## 5. torch_br / BRPyTorch Memory

- `torch_br` imports PyTorch, disables backend autoload during import to avoid
  circular dependency, then patches PyTorch.
- It calls `torch.utils.rename_privateuse1_backend("supa")` and
  `torch._register_device_module("supa", torch_br.supa)`.
- After `import torch_br`, `torch.device("supa:0")`, `torch.supa`, SUPA tensor
  methods, SUPA storages, SUPA streams/events, SUPA AMP, and SCCL distributed
  integration are available.
- Runtime probe on 2026-06-29 in the current environment: `/usr/bin/python3`
  is Python 3.10.12, `torch` is `2.8.0+cpu`, `torch_br` is
  `1.10.0.20800+br1xx`, `torch.supa.is_available()` is true, 8 SUPA devices
  are visible, and device 0 reports `Li-SEE MR106`.
- Useful `torch.supa` APIs include `is_available()`, `device_count()`,
  `set_device()`, `current_device()`, `get_device_properties()`,
  `synchronize()`, `empty_cache()`, `Stream`, and `Event`.
- `torch_br` registers SCCL as a distributed backend named `sccl`.
- `torch_br.utils.cpp_extension.load()` is the JIT path this repo uses. It
  detects `.su`/`.suh` sources, invokes BRCC, emits Ninja build files, and
  links the resulting extension into Python.
- `torch_br.utils.cpp_extension.load_inline()` can concatenate inline C++ and
  SUPA sources, but this repo uses file-based sources.
- The repo's native loader passes `extra_include_paths` from
  `torch_br.supa._adapter.get_br_include_paths()` and links `libtorch_br` from
  the installed `torch_br/lib` directory with rpath.
- Prefer the repo's JIT loader pattern in `gsplat/supa/_backend.py` over adding
  setuptools extension builds.

## 6. SUPA Development Pattern in This Repo

### 6.1 Dispatch and Fallback

- SUPA and CPU tensors use the PyTorch fallback by default.
- Public code should call device helpers from `gsplat/_device.py` instead of
  importing `torch_br` everywhere.
- `ensure_supa_runtime()` is the canonical way to import `torch_br` and assert
  that `torch.supa` exists.
- `torch_fallback_enabled(device)` defaults to true for SUPA and CPU and can be
  overridden by `GSPLAT_USE_TORCH_FALLBACK=1`.
- Native paths should be opt-in, return `None` when unsupported, and let the
  caller use a PyTorch implementation.
- Native failures in optional paths should warn and fall back unless the caller
  explicitly invoked a native-only wrapper.

### 6.2 Native Training Controls

- Training behavior must be controlled by CLI arguments and `TrainConfig`
  fields, not environment variables.
- The trainer's native controls are `--native true|false`,
  `--native-tiled false|true|auto`, `--native-tiled-metadata true|false`,
  `--native-tiled-train true|false`, and `--native-alpha-threshold <value>`.
- `--native-alpha-threshold` defaults to `None`/auto. When native training is
  requested, auto resolves to the standard `1/255` alpha cutoff
  (`0.00392156862745098`) to skip visually negligible tiled intersections.
  Pass `--native-alpha-threshold 0` to keep exact current SUPA tiled metadata
  for reference comparisons.
- Alpha-threshold tiled metadata now has a native SUPA path when native tiled
  metadata is enabled. Its metadata backend is
  `supa_native_alpha_threshold`; the older `cpu_numpy_alpha_threshold` path
  remains a fallback/reference when native offsets are not requested or native
  gating fails.
- Training code passes native settings explicitly through
  `supa_native_options` into rasterization. Do not reintroduce
  `GSPLAT_USE_SUPA_NATIVE*` environment-variable writes or reads in trainer,
  backend-comparison training scripts, benchmarks, correctness gates, or
  quality gates.
- Trainer native option normalization treats `--native true` as the full
  supported native profile: `native_tiled_train=true`,
  `native_tiled_metadata=true`, `native_tiled=auto`, and effective
  `native_alpha_threshold=1/255` plus
  `native_tiled_backward_mode=parallel` when no explicit tiled, alpha, or
  backward mode is set. `--native-tiled-train true`,
  `--native-tiled true|auto`, and a positive `--native-alpha-threshold` also
  enable native tiled metadata so the trainer does not silently fall back to
  CPU numpy tiled metadata.
- Low-level tests may keep legacy environment-toggle compatibility helpers only
  for old assertions; production dispatch, trainer commands, benchmarks, and
  gates must pass explicit `supa_native_options`.
- Keep experimental tiled metadata and tiled training opt-in outside the
  trainer native profile.

### 6.3 Python Wrapper Rules

- Gate native calls on exact supported device, dtype, shape, and contiguity.
  Most current kernels require SUPA device, `torch.float32` inputs, `torch.int32`
  metadata, and contiguous tensors.
- If a wrapper cannot prove support, return `None` rather than partially
  running native code.
- For native autograd, use `torch.autograd.Function` wrappers and save only the
  tensors needed for backward.
- Mark non-differentiable integer outputs such as radii.
- Current BRPyTorch behavior: some SUPA tensors produced by PyTorch SUPA ops
  can report as contiguous while exposing stale or partial data to native
  pointer kernels. Existing code mitigates this with CPU round-trips:
  `tensor.cpu().to(tensor.device).contiguous()`.
- Preserve these materialization helpers when adding native pointer paths:
  `_materialize_native_arg`, `_materialize_projection_grad`, and
  `_materialize_projection_saved`.
- For native autograd outputs that are not consumed by the loss, prefer
  preserving Python `None` through the wrapper and passing `nullptr` into the
  C++/SUPA path instead of allocating explicit zero gradient tensors. Kernels
  must treat `nullptr` as exact zero contribution and tests should compare the
  `None` and explicit-zero paths.
- For native autograd input gradients that PyTorch does not need, use
  `ctx.needs_input_grad` in the Python wrapper and pass `nullptr` output
  pointers through the C++ binding when the kernel can skip work safely. The
  projection backward path uses this to skip `v_viewmats` allocation,
  viewmat-only VJP loops, and atomics during normal trainer runs because
  camera matrices do not require gradients.
- Direct native wrappers should keep `synchronize_launch=True` defaults for
  simple call-site safety, but autograd hot paths may pass
  `synchronize_launch=False` when same-stream ordering is sufficient. The
  projection backward autograd path uses this to avoid a host synchronization
  after each native projection backward launch.
- Build non-differentiable sort/order metadata on CPU when the SUPA-produced
  index tensor may be unsafe for native pointer access.

### 6.4 C++ Binding Rules

- Validate all tensor inputs with `TORCH_CHECK` before reading raw pointers.
- Check device type, scalar type, contiguity, dimensionality, and int32-size
  limits.
- Use `const_data_ptr<T>()` for inputs and `data_ptr<T>()` for outputs.
- Allocate outputs with ATen on the input tensor's device, not raw SUPA
  allocation APIs.
- Use the current SUPA stream from the tensor/device context. Do not launch on
  an unrelated stream.
- Launch wrappers return `suError_t`; check `err == suSuccess` and include
  `suGetErrorString(err)` in the error message.
- Keep pybind names small and stable; wrappers in Python depend on them.

### 6.5 SUPA Kernel Rules

- Use `extern "C"` for launch functions and kernel symbols that C++ bindings
  declare.
- Use G-Mode `__global__` kernels unless there is a deliberate T-Mode design.
- Prefer simple 1D launches for current kernels:
  `idx = block_idx.x * block_dim.x + thread_idx.x`.
- Guard every kernel against `idx >= total`.
- Use `int`/int32 for sizes passed to kernels after host-side range checks.
  Avoid packed int64 tile metadata on SUPA.
- Current tile metadata uses int32 lanes (`dense_tile_ids`, `flatten_ids`,
  `isect_ids_hi`, `isect_ids_lo`) and reconstructs compatibility metadata
  outside the hot path when needed.
- Native tiled rasterization backward uses per-pixel prefix checkpoint
  workspace for the low-transmittance path. Keep the C++ binding-side pixel
  chunking so the workspace stays bounded, and do not restore repeated
  `range_start` to `chunk_start` prefix rescans inside reverse chunks.
- Native tiled rasterization backward also has a tile-parallel RGB
  `tile_reduce` path for `tile_size=16`. It uses one block per tile, reduces
  per-pixel gradient contributions in shared memory, and then issues one set
  of global atomics per Gaussian/tile. The tile-reduce kernel now always uses
  prefix-checkpoint transmittance for backward instead of recovering prefixes
  by dividing from final transmittance. Keep this stable prefix strategy for
  every tile range; do not reintroduce fast transmittance recovery without
  proving it against long-range and low-transmittance references. Auto dispatch
  only uses tile-reduce for moderate tile ranges as a performance policy, while
  explicit `parallel` and strict `tile_reduce` request the stable tile-parallel
  kernel for all supported RGB `tile_size=16` ranges.
- Historical trap: a 2026-07-14 optimization attempt that let
  non-low-transmittance lanes use fast transmittance recovery inside a
  low-transmittance tile-reduce block produced NaNs in
  `tile_reduce_low_transmittance_edge_tile`. A later 2026-07-21 fix removed the
  fast recovery branch from tile-reduce entirely. Because this kernel uses
  block-wide `__syncthreads()` in its gradient reductions, all lanes in a tile
  block must keep following the same reduction loop sequence.
- Return `suPeekAtLastError()` from launch wrappers after `suLaunchKernel()`.
- Be conservative with atomics. Existing gradient kernels use `atomicAdd` only
  on SUPA device memory. Do not introduce peer/host atomics.

### 6.6 Public Metadata Contract

- The fallback returns metadata such as:
  `backend`, `projection_backend`, `covariance_backend`, `composite_backend`,
  `composite_variant`, `sh_backend`, `extra_signals_sh_backend`,
  `visible_gaussians`, `culled_gaussians`, `compact_backend`,
  `tile_count_backend`, `tiled_metadata_backend`, `tiles_per_gauss`,
  `isect_ids`, `flatten_ids`, and `isect_offsets`.
- Preserve metadata shapes expected by upstream callers and tests, even when
  internal native paths use compacted or int32-lane representations.
- `meta["backend"]` for the SUPA fallback path is `"torch_fallback"`.
- Explicit full covariance inputs use full-matrix autograd semantics. When
  comparing against packed upper-triangle references, sum the `(i, j)` and
  `(j, i)` off-diagonal full-matrix gradients before comparing with the packed
  gradient lane.

## 7. Supported and Unstable Feature Surface

Supported and actively maintained on SUPA:

- Basic unpacked 3DGS rasterization through the PyTorch fallback for
  `pinhole`, `ortho`, and `fisheye` camera models. Native projection
  acceleration is currently pinhole-only.
- Optional native SUPA kernels for selected hot paths.
- Single-card SUPA training with `applications/supa_multiview_trainer.py`.
- Single-node distributed SUPA training through SCCL for the hardened trainer
  path.
- NPZ and COLMAP inputs for the trainer.
- Train crops, SH color training, densification/pruning, opacity reset,
  checkpointing, PLY export, and simple per-frame exposure refinement.

Not currently stable on SUPA:

- `packed=True`
- sparse gradients
- AbsGrad
- segmented rendering
- UT / Eval3D
- lidar
- camera distortion
- rolling shutter
- rays
- normals
- viewer parity
- native tiled metadata as a default path
- upstream-only accelerator packaging from this fork

## 8. Trainer Notes

- `TrainConfig.device` defaults to `"supa:0"`.
- Use `resolve_device()` and `set_accelerator_device()` before moving tensors.
- Images, masks, camera matrices, and input points are loaded on CPU and moved
  to the target device at training/evaluation time.
- Keep SH parameters in the current upstream-style shapes: `sh0` as
  `(N, 1, 3)` and `shN` as `(N, K, 3)`. Older flattened `shN` checkpoints are
  accepted and converted on load, but new trainer parameters/checkpoints should
  stay 3D.
- Distributed trainer mode currently requires SUPA devices and uses SCCL via
  `distributed_backend(device)`.
- Distributed launch sets one local SUPA device per rank
  (`supa:{local_rank}`), initializes the process group, shards data, and
  destroys the group at the end.
- Checkpoints store splats on CPU using upstream-style keys: `means`,
  `scales`, `quats`, `opacities`, `sh0`, and optional `shN`. Current
  checkpoints store `sh0` as `(N, 1, 3)` and `shN` as `(N, K, 3)`. Older SUPA
  checkpoints with `log_scales` and RGB-logit `colors` are converted on load.
- `applications/supa_multiview_trainer.py --data-type auto` treats `.npz`
  inputs as NPZ and other paths as COLMAP. NeRF synthetic directories must be
  converted to NPZ first.
- Trainer defaults started moving toward upstream `gsplat` semantics on
  2026-07-03: `--init-num-points` defaults to `-1`, active SH degree increases
  every `--sh-degree-interval 1000` steps, and trainer scene scale is
  `loaded_scene_scale * 1.1 * --global-scale`.
- Means LR decay follows upstream `ExponentialLR` timing: the LR used at
  one-based trainer step `s` is `base_lr * final_factor ** ((s - 1) / iterations)`.
- Active SH degree follows upstream zero-based timing after mapping local
  one-based step `s` to `s - 1`; degree 1 starts at local step
  `sh_degree_interval + 1`.
- Training frame shuffling uses a single CPU `torch.Generator` stream seeded
  once per logical stream, then draws consecutive epoch `randperm` orders. This
  is closer to upstream `DataLoader(..., shuffle=True)` than reseeding with
  `seed + epoch`, while keeping the trainer's stateless resume-friendly helper.
- Trainer rasterization exposes upstream-style `near_plane` and `far_plane`
  configuration through `TrainConfig`, CLI, training, evaluation, and
  distributed evaluation paths. Defaults are `0.01` and `1e10`.
- Trainer training steps support upstream-style random RGB backgrounds via
  `--random-bkgd true`. The random background is applied after rasterization and
  before exposure/loss using `render + background * (1 - alpha)`; evaluation
  remains deterministic and does not use random backgrounds.
- Trainer CLI `--init-num-points` accepts negative values. After data loading,
  a negative value is resolved to the loaded input point-cloud count; for
  COLMAP this is the `points3D` sparse point count. Datasets without an input
  point cloud must use a positive value for random initialization.
- Trainer output paths are selected with mutually exclusive `--result-dir`
  and `--auto-output-root`. Use `--result-dir` for an exact output directory.
  Use `--auto-output-root` for a parent directory; the trainer generates a
  timestamped child name from the dataset name and key run parameters.
- Trainer CLI logging has two intentionally different levels:
  `--print-all-step-info true` prints one concise line per training step, while
  `--print-debug-info true` prints verbose flat debug events for initialization
  and individual operations inside each step, including rasterization, loss,
  backward, optimizer, refinement, evaluation, and save operations. Debug lines
  include wall-clock time plus monotonic `elapsed_s` and `delta_s` fields for
  duration analysis; prefer those monotonic fields over wall time when
  comparing operation costs.
- Trainer metrics include `train.step_times_s`, `train.last_step_time_s`, and
  `train.step_time_summary_s` with finite sample `count`, `min`, `mean`,
  `median`, and `max`. The training quality gate surfaces median and mean step
  time from that summary for performance comparisons.
- Trainer metrics also include `train.phase_names`, `train.phase_times_s`, and
  `train.phase_time_summaries_s` for low-overhead host-side phase timing.
  Phases include data/crop, target transfer, rasterize, postprocess, loss,
  backward, grad sync, refinement state, optimizer step, refinement,
  synchronization, CPU metric extraction, and save-step work. These timings do
  not synchronize between phases, so asynchronous SUPA work can be attributed
  to the later operation that waits for it; use them to find synchronization
  boundaries before adding heavier event/synchronized profiling.
- Trainer PLY export filters rows with non-finite means, SH DC, opacities,
  scales, or rotations before writing `element vertex`. If a saved PLY appears
  to lose many points while the checkpoint and `metrics.json` still report a
  larger Gaussian count, inspect checkpoint tensors for NaN/Inf rather than
  assuming densification/pruning deleted those Gaussians.
- Single-card SUPA training calls public `gsplat.rasterization()` so trainer
  dispatch goes through the public API boundary. CPU tests still call the
  PyTorch fallback directly where they exercise CPU-only behavior.
- Masked trainer SSIM loss follows upstream's zeroed-image convention: L1 is
  computed only on valid mask pixels, while SSIM is computed after multiplying
  both render and target by the mask.
- Trainer SSIM on SUPA uses a separable grouped-convolution box filter instead
  of `avg_pool2d` for the local means/variances. The filter is mathematically
  equivalent to the old zero-padded `avg_pool2d(..., count_include_pad=True)`
  path for the trainer's box window, but avoids a slow BRPyTorch
  `avg_pool2d` backward path. CPU continues to use `avg_pool2d`.
- Trainer initialization and optimizer configuration now expose upstream-style
  knobs: `--init-opa`, `--init-scale`, `--means-lr`, `--scales-lr`,
  `--opacities-lr`, `--quats-lr`, `--sh0-lr`, and `--shN-lr`.
- Upstream `gsplat/examples/simple_trainer.py` initializes SFM Gaussian log
  scales from exact full-point-cloud 3-nearest-neighbor RMS distances:
  `sqrt((knn(points, 4)[:, 1:] ** 2).mean(-1)) * init_scale`, repeated across
  xyz axes. Its example helper uses sklearn `NearestNeighbors`, and
  `gsplat/init_utils.py::knn_scale_init` provides an equivalent chunked
  `torch.cdist` implementation. Do not replace this with a small random
  reference subset when matching upstream training quality; that can
  systematically overestimate scales and saturate alpha.
- Exact full-cloud KNN scale initialization is important for training quality.
  A local south-building probe showed small random reference subsets can
  overestimate initial scales, saturate alpha, and flatten early renders.
- New trainer runs use an upstream-style per-parameter optimizer dictionary.
  Checkpoints save optimizer state as a tagged `optimizer_dict`; old
  single-optimizer checkpoint state is skipped safely on resume.
- BRPyTorch Adam on SUPA can call unsupported `LerpScalar` through both
  PyTorch's foreach path (`torch._foreach_lerp_`) and single-tensor path
  (`exp_avg.lerp_`). Trainer Adam optimizers use the local `_SupaSafeAdam`
  implementation instead of `torch.optim.Adam`; the maintained trainer path is
  now only the per-parameter optimizer dictionary, with legacy single-optimizer
  checkpoint state skipped on resume.
- Trainer-local refinement now resizes its running `grad2d`, `count`, and
  `radii` state during duplicate, split, and prune operations to match
  upstream `gsplat.strategy.ops` semantics while preserving local growth-cap
  controls.
- The current fixed environment imports `pycolmap 4.1.0` from the system
  Python path. This checkout no longer relies on a repo-local pycolmap wheel,
  but `env.sh` still leaves `.agents/temporary/python-local` on `PYTHONPATH`
  so restored local wheels work without changing the system environment.
- Local import probe on 2026-07-06 found `cv2` and `imageio` are not installed
  in the fixed environment. `applications/datasets/colmap.py` now lazy-loads
  optional COLMAP helpers by actual feature path: `pycolmap` for sparse model
  loading, `cv2` only for image undistortion/debug point drawing, and
  `imageio` only for debug video writing. Users may install `pycolmap` and
  `cv2` when their workflow needs them, but agents must still not mutate the
  system Python environment.
- The COLMAP loader prefers `pycolmap.Reconstruction`, which reads COLMAP
  binary or text sparse models, then falls back to legacy
  `pycolmap.SceneManager`, then to the pure-Python text reader only when
  `pycolmap` is unavailable. Normal COLMAP work should use `pycolmap`.
- The south-building dataset under `.agents/dataset/south-building` is COLMAP
  text format with `images/` and `sparse/{cameras.txt,images.txt,points3D.txt}`.
  The current fixed environment does not provide `piexif`; the COLMAP loader
  imports EXIF dependencies only when exposure loading is requested.
- Current dataset state: `.agents/dataset/minecraft_dungle_pyramid_undistorted`
  is the runnable canonical COLMAP quality-gate dataset and includes `images/`,
  `images_4_png/`, and `sparse/`.
  `.agents/dataset/minecraft_dungle_pyramid_undistorted_improved` currently has
  only `sparse/` and is not runnable for trainer commands unless images are
  restored. `.agents/dataset/lego` is absent in this checkout.
- For COLMAP training with `--data-factor N`, the loader can derive missing
  downscaled image folders from `images/` into `images_N_png/`. Keep those
  generated image folders under `.agents/dataset/` with the dataset.
- Use `tools/diagnose_clean_colmap.py` before training on suspicious COLMAP
  reconstructions. The tool uses `pycolmap.Reconstruction`, reports robust
  camera/point quantiles through the shared logger, and treats point-drop
  percentages as percentages in `[0, 100]` (`0.1` means 0.1%). By default it
  only flags bad images; `--clean-images true` deregisters flagged frames in
  the written sparse model and should be used deliberately.
- The lego dataset under `.agents/dataset/lego` is NeRF synthetic format. A
  working conversion command is:
  `source env.sh && /usr/bin/python3 tools/convert_nerf_synthetic_to_npz.py --dataset-dir .agents/dataset/lego --transforms .agents/dataset/lego/transforms_train.json --output .agents/dataset/lego/lego-train.npz --background white --eval-every 8`.
- The NeRF synthetic converter writes images, masks, alpha, camera matrices,
  split indices, and scene metadata, but not `points` or `point_colors`.
  Trainer NPZ mode can initialize random points when the point cloud is absent.
- `benchmark/common.py::load_npz_scene()` requires `points` or `means` for
  `--scene npz`. For lego rasterization benchmarks, create a derived NPZ with
  visible `points` and `point_colors` or use `--scene synthetic`.
- Do not assume a tiny trainer checkpoint provides a good benchmark point cloud
  for the first lego camera. Verify `meta["visible_gaussians"]` before timing
  train benchmarks; if all Gaussians are culled, the rendered output can be a
  constant and `loss.backward()` is not meaningful.

## 9. Validation Habits

- Fast structural syntax check:
  `source env.sh && /usr/bin/python3 -m compileall -q gsplat/_device.py gsplat/rendering.py gsplat/_torch_fallback.py gsplat/supa/_backend.py gsplat/supa/_wrapper.py applications/supa_multiview_trainer.py`
- Focused SUPA runtime probe:
  `source env.sh && /usr/bin/python3 -m pytest -q tests/test_supa_runtime_probes.py -o cache_dir=.agents/temporary/pytest-cache --basetemp=.agents/temporary/pytest-tmp/runtime`
- Focused SUPA fallback/native tests:
  `source env.sh && /usr/bin/python3 -m pytest -q tests/test_supa_fallback.py`
- Native rasterization operator guard:
  `source env.sh && /usr/bin/python3 -m pytest -q tests/test_supa_operator_guard.py -o cache_dir=.agents/temporary/pytest-cache --basetemp=.agents/temporary/pytest-tmp/operator-guard`
- Trainer tests:
  `source env.sh && /usr/bin/python3 -m pytest -q tests/test_supa_multiview_trainer.py`
- Rendering correctness gate:
  `source env.sh && /usr/bin/python3 tools/render_correctness_gate.py`
- Training quality gate:
  `source env.sh && /usr/bin/python3 tools/training_quality_gate.py --run true`
- When SUPA hardware, BRCC, or `torch_br` are unavailable, tests should skip
  cleanly rather than fail due to import or device assumptions.
- Local validation re-probe on 2026-07-06 found
  `/usr/bin/python3 -m pytest --version` reports `pytest 9.1.1` from the fixed
  environment. Use the repo-local cache flags above for test runs and still do
  not install or mutate global Python packages.
- For benchmarks, warm up native/JIT paths and write JSON under
  `.agents/output/perf/`.
- Major changes that can affect training quality, runtime, native dispatch, or
  measurement tooling must include a refreshed fixed baseline before handoff.
  This includes native SUPA kernels, rasterization metadata, trainer behavior,
  optimizer/refinement logic, public rasterization dispatch, benchmark scripts,
  and quality gates. Use the current canonical
  `.agents/dataset/minecraft_dungle_pyramid_undistorted` baseline shape unless
  the user explicitly selects a replacement: COLMAP data factor 4,
  `iterations=100`, `init_num_points=65535`, `train_crop_size=256`,
  `native=true`, `native_tiled=auto`, omit `--native-alpha-threshold` so the
  native profile uses the effective standard `1/255` alpha cutoff and omit
  `--native-tiled-backward-mode` so the native profile resolves auto to
  `parallel`, `save_ply=false`, `seed=42`, `refine_profile=crop-balanced`,
  `refine_start_iter=25`,
  `refine_stop_iter=100`, `refine_every=25`, `reset_every=0`, and
  `max_gaussians=65535`. Record `eval_max_frames`; current trainer defaults
  to `8`, while older fixed baseline records may have used `4`, so compare
  eval metrics only when the frame count matches or when a first-4-frame
  comparison is explicitly reported.
- Each refreshed baseline record should include the exact command or enough
  configuration to reproduce it, the dataset path, result/perf JSON paths,
  final loss, L1/SSIM loss, eval PSNR/L1/MSE when available, backend metadata,
  step-time summary, key perf benchmark medians, and whether the result is
  intended to replace or compare against the previous fixed baseline.
- Native operator benchmarks use the repo-local native JIT build directory
  `.agents/temporary/native-build/supa_native_build` by default.
- Local probes found `scipy` is not installed in the fixed environment. Do not
  install scipy as an agent workaround. `filter_3dgs_ply` and `clean_3dgs_ply`
  both have NumPy KNN fallbacks for isolated-point filtering when
  `scipy.spatial.cKDTree` is unavailable.
- Retained historical implementation facts from older validation records:
  native tiled metadata uses SUPA device-side stable `torch.argsort` plus
  `bincount`/`cumsum` after native lane expansion, with legacy custom offsets
  kept only as fallback/benchmark reference; native tiled backward uses bounded
  prefix-checkpoint workspace; the RGB `tile_size=16` tile-reduce path must use
  the stable prefix-checkpoint strategy; optional garden asset tests skip when
  `.agents/dataset/test_garden/test_garden.npz` is absent; COLMAP diagnostics
  can clean suspicious reconstructions but image deregistration requires
  explicit `--clean-images true`.
- Old detailed per-change pytest command logs and superseded benchmark numbers
  were intentionally condensed in July 2026. Prefer the latest records in
  Section 11 plus focused tests near the changed code when validating new work.

## 10. Coding Style and Review Checklist

- Keep edits scoped. Do not refactor unrelated code when working on SUPA
  bring-up.
- Preserve upstream API compatibility unless the task explicitly narrows scope.
- Write English comments for non-obvious code. Comments should explain why a
  choice is needed, especially for SUPA runtime behavior, BRPyTorch tensor
  materialization, shape invariants, metadata layouts, and fallback decisions.
- Add necessary logs or status output in long-running workflows and scripts,
  using the existing module's style. Keep low-level kernels and per-pixel hot
  paths quiet unless debugging is explicitly requested.
- Do not add new raw `print()` calls for project runtime output. The unified
  logging format lives in `gsplat/_logging.py`; COLMAP loaders, trainers,
  benchmarks, gates, and tools should all use it.
- Prefer explicit shape comments in function signatures for tensor-heavy code.
- Use type hints and small helper functions for device, shape, and metadata
  normalization.
- Project CLI boolean parameters must be explicit `--flag true|false`. Do not
  use `argparse.BooleanOptionalAction`, bare `store_true`/`store_false` feature
  toggles, or generated `--no-*` aliases. Use named values for multi-state
  modes, for example `--native-tiled false|true|auto`.
- For user-facing output path options, prefer a mutually exclusive exact path
  and auto-root pair where practical: `--result-dir` for an exact directory
  and `--auto-output-root` for a parent directory that receives an
  automatically named run subdirectory.
- Prefer CPU references or the top-level pure-PyTorch reference helpers for
  correctness comparisons.
- Use warnings for optional native fallback failures.
- Avoid broad `except Exception` except at environment-dependent native opt-in
  boundaries where fallback is intentional.
- Keep generated build products and experiments out of source-controlled paths.
- Add or update tests when changing dispatch, metadata, gradients, trainer
  checkpoint format, distributed behavior, or native kernels.
- For native additions, verify forward and backward against PyTorch references,
  then verify public `rasterization()` metadata and trainer integration.
- After code changes pass relevant validation, commit with coherent granularity
  and message format `type(sphere): detail`. Avoid mixing unrelated layers in
  a single commit.

## 11. Validation Records

Keep this section concise. Record only the latest current baseline, the latest
project-wide validation state, and still-actionable regression findings. Do not
keep long superseded `pytest -k` command logs or every historical training
baseline once a newer fixed gate covers the same behavior.

- 2026-07-24 slimmed `AGENTS.md` by condensing obsolete training/test history
  and fixing stale trainer notes. Removed redundant old dataset-presence
  probes, superseded baseline chains, takeover-only probes, and repeated
  focused-test command logs. Preserved durable rules, current environment
  constraints, current dataset state, native/tiled regression gotchas, and the
  latest fixed quality-gate result. Validation for the cleanup:
  `git diff --check`; no training baseline regenerated because this was documentation
  cleanup only.
- 2026-07-24 generated the static Chinese SUPA operator optimization handoff
  guide under `.agents/output/supa_improve_guide/`: `index.html`,
  `01-project-structure.html`, `02-runbook.html`, `03-call-chain.html`,
  `04-operators.html`, `05-optimization.html`, `06-code-appendix.html`,
  `07-optimization-case.html`, and `assets/style.css`. The revised handoff
  scope treats `gsplat/supa/kernels.su` as the only routine modification file
  for a SUPA-only operator/framework optimizer; wrapper, binding, fallback,
  trainer, tests, and benchmarks are documented as read/run context. The guide
  now includes Chinese terminology explanations, detailed operator correctness
  and speed-test commands, and annotated key `kernels.su` code excerpts. A later
  2026-07-24 revision removed homepage sidebar links to source/test/benchmark
  files and added a local hover-tooltip glossary (`assets/terms.js`) for
  high-cost SUPA, native, fallback, metadata, test, benchmark, dtype, shape,
  stream, and kernel terms without modifying embedded code blocks. A 2026-07-27
  revision added chapter 7 as a worked optimization case for precomputing
  `scale[k] * scale[k]` in `covar_from_rot_scale()`, with problem discovery,
  localization, patch, correctness validation, and before/after projection
  benchmark results. Validation for the guide: `git diff --check`,
  `node --check` for the glossary script, an HTML parser smoke pass over every
  generated page, homepage-link scan, and meta-narration/trailing-whitespace
  scans.
- 2026-07-27 optimized `gsplat/supa/kernels.su::covar_from_rot_scale()` by
  precomputing the three scale squares once per Gaussian and replacing the
  innermost 3-way loop with explicit accumulation. This reduces scale-square
  multiplications from 27 to 3 per covariance construction without changing
  the helper signature, output layout, wrapper, binding, fallback, or trainer
  code. Validation after `source env.sh`: `pytest -q
  tests/test_supa_fallback.py -k "quat_scale or projection_pinhole" -o
  cache_dir=.agents/temporary/pytest-cache
  --basetemp=.agents/temporary/pytest-tmp/scale-sq-projection` passed (`13
  passed, 102 deselected`); `pytest -q tests/test_supa_operator_guard.py -o
  cache_dir=.agents/temporary/pytest-cache
  --basetemp=.agents/temporary/pytest-tmp/scale-sq-operator-guard` passed (`2
  passed`). Projection benchmark command before/after:
  `benchmark/supa_operators.py --bench projection --warmup 2 --iters 7`; JSON
  outputs are `.agents/output/perf/projection_scale_sq_before.json` and
  `.agents/output/perf/projection_scale_sq_after.json`. Median
  `native_autograd_no_viewmat_grad` changed from `0.013064343s` to
  `0.013027761s` (`+0.280%`), `native_backward_no_viewmats` from
  `0.009507820s` to `0.009497270s` (`+0.111%`),
  `native_autograd_with_viewmat_grad` from `0.635195352s` to `0.629289813s`
  (`+0.930%`), and `native_backward_all_grads` was effectively neutral within
  sampling noise (`0.602442600s` to `0.602716036s`). No fixed training quality
  baseline was regenerated because this was a narrow helper-level native
  micro-optimization with focused correctness and public guard coverage.
- 2026-07-24 added `tests/test_supa_operator_guard.py`, a public
  rasterization guard that compares native full-scan and tiled-train profiles
  against PyTorch fallback for outputs, gradients, dispatch metadata, and
  coarse speed. The static handoff guide was later regenerated under
  `.agents/output/supa_improve_guide/`; keep using this guard, the source code,
  and the regenerated guide as current handoff material.
  Validation after `source env.sh`: `compileall -q tests/test_supa_operator_guard.py`
  passed and `pytest -q tests/test_supa_operator_guard.py -o
  cache_dir=.agents/temporary/pytest-cache
  --basetemp=.agents/temporary/pytest-tmp/operator-guard` passed (`2 passed`).
  No training quality baseline was regenerated because this added tests and
  docs without changing runtime behavior.
- 2026-07-24 takeover re-probe after `source env.sh`: `/usr/bin/python3` is
  Python `3.10.12`, `torch 2.8.0+cpu`, `torch_br 1.10.0.20800+br1xx`,
  `torch.supa.is_available()` is true, 8 `Li-SEE MR106` SUPA devices are
  visible, `compileall -q gsplat applications tools benchmark tests` passed,
  `pytest --collect-only` collected 187 tests, and
  `pytest -q tests/test_supa_runtime_probes.py -o
  cache_dir=.agents/temporary/pytest-cache
  --basetemp=.agents/temporary/pytest-tmp/takeover-runtime` passed (`1 passed`).
- 2026-07-22 current fixed quality gate v2 replaced the previous baseline
  chain. Command shape: `.agents/dataset/minecraft_dungle_pyramid_undistorted`,
  COLMAP data factor 4, `iterations=100`, `init_num_points=65535`,
  `train_crop_size=256`, `native=true`, `native_tiled=auto`, effective
  `native_alpha_threshold=1/255`, effective tiled backward mode `parallel`,
  `save_ply=false`, `seed=42`, `refine_profile=crop-balanced`,
  `refine_start_iter=25`, `refine_stop_iter=100`, `refine_every=25`,
  `reset_every=0`, `max_gaussians=65535`, and `eval_max_frames=8`. Metrics at
  `.agents/output/train/quality-gate-v2/reduce-training-chain-20260722/minecraft_undistorted_crop256_native_i100/metrics.json`: step `100`,
  Gaussians `65535`, final loss `0.2420420`, L1 `0.1822460`, SSIM loss
  `0.4812259`, eval PSNR `11.4581`, eval L1 `0.1984220`, eval MSE
  `0.0714811`, `projection_backend=supa_native`,
  `composite_backend=supa_native`, `composite_variant=tiled`,
  `tile_count_backend=supa_native`,
  `tiled_metadata_backend=supa_native_alpha_threshold`, step-time summary count
  `100`, median `0.3991s`, mean `0.4514s`, min `0.3513s`, max `5.6891s`,
  refinement events `0`.
- 2026-07-22 latest full focused validation for the reduced training chain
  passed after `source env.sh`: `compileall` across fallback, wrapper, trainer,
  benchmarks, gates, comparison tools, and tests;
  `tests/test_supa_multiview_trainer.py` (`50 passed`);
  `tests/test_supa_fallback.py` (`110 passed, 5 skipped`); selected native
  trainer/fallback tests (`9 passed` and `5 passed`);
  `tests/test_supa_benchmarks.py` (`3 passed`); and
  `tools/render_correctness_gate.py --output
  .agents/output/render_correctness/reduce-training-chain-20260722.json` passed
  with 4 cases passed and optional `garden_native_subset` skipped because the
  garden NPZ asset is absent.
- 2026-07-21 project-wide public-surface freeze validation passed after the CUDA
  removal and API tightening: full pytest passed with `176 passed, 5 skipped,
  2 xfailed, 1 xpassed`; `pytest --collect-only` collected 184 tests at that
  point; `compileall -q gsplat applications tools benchmark tests` passed; and
  source scans over active code/docs found no CUDA references outside
  `AGENTS.md`. No quality baseline was regenerated then because trainer/native
  algorithms did not change.
- High-risk historical regressions to preserve in future tests: full-resolution
  long tile ranges previously produced NaN/Inf pollution when unsafe tile-reduce
  paths bypassed range policy; mixed barrier paths in tile-reduce previously
  caused SUPA SPC crashes; exact alpha tiled metadata can produce very large
  gradients, so the native trainer profile defaults to the standard `1/255`
  alpha cutoff; `_SupaSafeAdam` must sanitize non-finite and square-overflowing
  updates; PLY export filters non-finite rows, so apparent PLY point-count drops
  should be checked against checkpoint tensors and `metrics.json` before
  blaming pruning.
