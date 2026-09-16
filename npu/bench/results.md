# NPU microbench results (append-only)

## 2026-09-16 — bf16 GEMM (registry kernel) vs torch fp32 linear at slorado sup@v5.0.0 shapes

Env: zhang-ryzen2, openfish `6e74487`, mlir-air wheel `0.0.1.2026090504+6746658`, XRT 2.21.75, NPU FW 1.1.2.65.
NPU idle before runs; server load avg 0.02→3.0 during NPU runs, 0.47→3.95 during CPU runs (other users).
Raw logs: server `~/p0/gemm_sup_4096.txt`, `~/p0/cpu_linear.txt`.

NPU: `bench/gemm_sup.sh 4096` = `kernels/gemm_bf16` `make run`, high precision `--method auto`
(fused-cast TILE_M=64 if M·K·N ≥ 4e9 else drain TILE_M=32), tile_k_l2 256, tile_k_l1 32, tile_n 128, herd 8×4
(32 tiles), randn·(1/√K) inputs, 30 perf iters after 10 warmup, **kernel-only mean** (no host conversion/sync).
CPU: `bench/cpu_linear.py`, torch 2.14.0+cu130 (airenv, AVX512; not slorado's libtorch 2.9.0), fp32
`F.linear` no bias, median / p95 of 30 (5 at M=131072).

| op | M×K×N | NPU method | NPU correctness (rtol 1.6e-2, atol 1.5e-3) | NPU mean_rel_L1 | NPU kernel ms | CPU 8 thr median ms | CPU 16 thr median ms | NPU speedup vs 8 / 16 thr |
|---|---|---|---|---|---|---|---|---|
| fc1 | 4096×512×4096 | fused-cast | **FAIL** 633/16,777,216 (abs_err max 2.93e-3) | 9.709e-3 | 4.43 | 25.49 | 22.68 | 5.8× / 5.1× |
| fc2 | 4096×2048×512 | fused-cast | PASS (abs_err max 1.47e-3) | 9.709e-3 | 1.73 | 9.89 | 6.60 | 5.7× / 3.8× |
| wqkv | 4096×512×1536 | drain | **FAIL** 236/6,291,456 (abs_err max 2.44e-3) | 9.280e-3 | 1.99 | 7.00 | 4.10 | 3.5× / 2.1× |
| out_proj | 4096×512×512 | drain | **FAIL** 56/2,097,152 (abs_err max 2.01e-3) | 9.280e-3 | 0.70 | 2.30 | 1.43 | 3.3× / 2.0× |

CPU at the real batch (M = 131,072 rows per call, 8 / 16 threads): fc1 733 / 513 ms, fc2 346 / 283 ms,
wqkv 276 / 228 ms, out_proj 100 / 91 ms. NPU kernel-only extrapolation (32 launches of M=4096):
fc1 142 ms, fc2 56 ms, wqkv 64 ms, out_proj 22 ms — before host f32↔bf16 conversion and layout copies,
which for silu_mul cost ~4× the kernel.

Findings:
- **Correctness:** the failures are deterministic (identical counts on rerun), ~0.004% of elements, all small
  outputs (|ref| ≈ 0.002–0.025) with abs_err 1.5–2.9e-3 just above the fixed `atol = 1.5e-3`. mean_rel_L1 is
  in-family (9.3–9.7e-3). The registry calibrated atol at K = 8192 (input scale 1/√8192); at K = 512 inputs are
  4× larger, so the same relative bf16/BFP16 error is larger in absolute terms. Per the gate these are FAILs;
  no tolerance was changed. Next: validate against **real SUP weights/activations** (dumps have `ff_in`,
  `ff_fc1_weight`, `attn_in`, `attn_wqkv_weight`, …) and let the identity gate decide.
- **Speed:** kernel-only, the NPU GEMM is 2–6× faster than torch fp32 on the CPU at these shapes (best on fc1/fc2).
  Unlike silu_mul this op is compute-bound, so a real end-to-end win is plausible if host conversion is kept
  small (only the 512-wide side crosses when fc1→silu_mul→fc2 is fused).
