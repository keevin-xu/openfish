# openfish/npu — lessons (append-only)

## 2026-09-16
- The CPU oracle for `silu_mul` and `rmsnorm` is **torch** in `TxModel.cpp` (non-`USE_GPU` branch), not
  `nn_cpu.c` — `nn_cpu.c` only implements `rotary_emb`. openfish's GPU kernels are only reachable under `USE_GPU`.
- openfish `silu_mul` input is `[y ‖ gate]` per row; mlir-air `silu_and_mul` takes `(gate, up, out)` as
  separate tensors. A negative control (swapped layout, max_abs 4.18) proves the check can fail.
- SUP transformer T is chunk/12 = 1024 (5 conv layers), not chunk/6; the upsample x2 restores 2048 afterwards.
- Encoder input activations are non-contiguous (strides 524288,1,1024) — an NPU wrapper must `contiguous()` or
  honour strides.
- Regenerating the rotary sin/cos tables in numpy differs from torch's by ~6e-5 (float32 `pow`/`outer` at
  positions up to 2047). Treat the tables as model inputs (use torch's), not as something to recompute.
- `slorado -C` (gpu batch size) also governs CPU memory: sup at the default 512 exceeds 30 GiB.
- Build-flag trick without editing the Makefile: `CPPFLAGS=-DFOO make` (env var; the Makefile's `+=` appends).
  `make CPPFLAGS+=...` on the command line would *replace* the Makefile's include paths.

## 2026-09-16 (Phase 1, silu_mul)
- The installed `XRTRunner.run_test` locks `tempfile.gettempdir()/npu.lock`, i.e. `/tmp/npu.lock`, which another user
  owns on zhang-ryzen2. Set `TMPDIR=~/.npu-tmp` for mlir-air example runs instead of patching mlir-air.
- Don't take `$NPU_LOCK` inside a Python harness that is launched under `flock $NPU_LOCK`: the child inherits the
  locked fd and a second open+flock would deadlock. Lock from outside only.
- `XRTBackend.last_latency_us` is a **mean** over `n_perf_iters` (kernel-only, syncs excluded) — fine for registry
  comparability; our own G2 timing must report median/p95.
- The Python invoker allocates and maps new BOs on every call: 73 ms per call vs 4.3 ms of kernel time at N=16.8M.
  Real SUP activations are *easier* than randn for bf16 SiLU-Mul (8.9e-3 vs 1.0e-2) because |gate| is smaller.
- aircc compile of this 8-tile elementwise design takes ~0.2 s on the server (verified from a clean build dir).

## 2026-09-16 (Phase 2, silu_mul in slorado)
- C++ XRT: never pass `xrt::ext::bo` straight to `run.set_arg`; store/cast to `xrt::bo` (see debug_log).
- bf16 is not IEEE fp16: 1/8/7 bits. f32→bf16 = round the 23-bit mantissa to 7 bits nearest-even, carry into the
  exponent, overflow→inf. Validate host converters against `ml_dtypes` before trusting them.
- slorado pads every batch to `-C` chunks: even a 1-read run makes full 128×1024×4096 silu_mul calls (18 per batch).
- At N=268M per call, host-side layout split + f32→bf16 (8 threads) costs ~4× the NPU kernel time; the kernel is
  not the bottleneck at this granularity. Expected for Tier 1 (memory-bound op).
- make does not track -D flags: openfish's lib and slorado's objects must be rebuilt per variant
  (`npu/scripts/build_slorado_variants.sh`); keep variant binaries in `~/p0/bin`, run with cwd = slorado root.

## 2026-09-19 (iGPU base)
- ROCm on this box needs its own libtorch: the rocm6.4 wheel has no gfx1151 rocBLAS/hipBLASLt and bundles HIP .so.6.
  AMD's `torch-2.9.1+rocm7.2.1.lw` links the system ROCm 7.2.1 — link and run with `/opt/rocm/lib` on LD_LIBRARY_PATH,
  and never with `~/npu-env.sh` sourced (distro HSA 5.7.1 → "unrecognized id").
- The iGPU runs SUP in fp16; every NPU substitution costs a GPU→host copy + fp16→fp32→bf16 and the reverse.
- Load NPU kernels per K×N shape, not per op: 7 ops fit in 6 hw contexts (crf shares fc1's 512×4096 ELF).
- The CRF linear failed 30.8% of elements at registry tolerance with the bfp16 GEMM yet kept identity within
  0.0001 — the element-wise gate is far stricter than what basecalling needs; report both.
