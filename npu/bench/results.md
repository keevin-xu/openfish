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

## 2026-09-16 — bf16 GEMM with real sup@v5.0.0 weights + activations (layer 0)

`test/test_gemm_npu.py` (openfish `58ea484` + drain fix), dump `~/p0/dump_sup8`, M = 4096 rows (first 4 chunks),
B = weight.T, same kernels/tiles as above. NPU kernel-only mean ×20; CPU torch fp32 8 threads median ×20 on the
same A. Raw: server `~/p0/gemm_real.txt`, `~/p0/gemm_real_attn.txt`.

| op | M×K×N | method | registry gate (rtol 1.6e-2, atol 1.5e-3) | outside tol | mean_rel_L1 vs fp32(A_bf16@B_bf16) | vs CPU dump mean_rel_L1 | vs CPU dump cosine | bf16-input floor mean_rel_L1 | NPU ms | CPU ms | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fc1 | 4096×512×4096 | fused-cast | FAIL | 1,686,580 (10.05%) | 7.819e-3 | 8.094e-3 | 0.99997199 | 2.094e-3 | 4.64 | 23.85 | 5.1× |
| fc2 | 4096×2048×512 | fused-cast | FAIL | 169,038 (8.06%) | 8.077e-3 | 8.396e-3 | 0.99996687 | 2.303e-3 | 1.60 | 9.78 | 6.1× |
| wqkv | 4096×512×1536 | drain | FAIL | 73,476 (1.17%) | 3.068e-3 | 3.565e-3 | 0.99999370 | 1.712e-3 | 2.04 | 6.58 | 3.2× |
| out_proj | 4096×512×512 | drain | FAIL | 676 (0.03%) | 6.642e-3 | 6.480e-3 | 0.99998275 | 2.318e-3 | 0.71 | 2.09 | 3.0× |

abs_err max: fc1 3.1e-2, fc2 2.0e-2, wqkv 1.6e-2, out_proj 7.8e-3.

Real tensors are heavy-tailed (|x| max / median): ff_in 49, fc1 weight 43, silu_mul out 479, fc2 weight 20,
attn_in 38, wqkv weight 56. **Hypothesis (unverified):** aie2p's GEMM emulates bf16 with BFP16 blocks of 8
sharing one exponent (`AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`), so small values in a block with an outlier lose
precision; randn inputs don't exercise this. The error beyond the bf16-input floor (≈2e-3) is ≈1–6e-3 mean.
Not a Phase-1 PASS for any linear; per-op cosine ≥ 0.99997 — the identity gate would have to decide.

## 2026-09-16 — root cause: BFP16 emulation in the aie2p GEMM microkernel (confirmed)

`aie_api/detail/aie2p/mmul_bf16_bf16.hpp`: with `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` (set by the registry
Makefile) the 8×8×8 mmul converts both operands to `v64bfp16ebs8` = **blocks of 8 elements sharing one 8-bit
exponent with 8-bit integer mantissas** (`aie_doc.hpp` block-vector table), documented as "to increase throughput
at the cost of accuracy". bf16 proper has a per-element exponent. Heavy-tailed real tensors (max/median 20–479)
put small values in blocks with outliers → few effective bits.

Experiment: same `mm_aie2p.cc`, same tiles/method, compiled with vs without that define (`test_gemm_npu.py
--mmul {bfp16,native}`), real SUP L0 data, M=4096. "fp32 intermediate" = the fused-cast f32 scratch, i.e. the
`bf16_in_fp32_out` datapath. Raw: server `~/p0/gemm_real_bfp16.txt`, `~/p0/gemm_real_native.txt`.

| op | mmul | registry gate | out vs fp32(A_bf16@B_bf16) mean_rel_L1 | fp32 intermediate vs same | vs CPU dump (bf16 out) | vs CPU dump (fp32 out) | cosine (bf16 out) | NPU ms | CPU 8thr ms | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| fc1 | bfp16 | FAIL 10.05% | 7.819e-3 | 7.248e-3 | 8.094e-3 | 7.390e-3 | 0.99997199 | 4.60 | 23.97 | 5.2× |
| fc1 | native | **PASS** | 2.819e-3 | **1.240e-7** (0 outside tol) | 3.276e-3 | **1.415e-3** | 0.99999375 | 20.25 | 23.36 | 1.2× |
| fc2 | bfp16 | FAIL 8.06% | 8.077e-3 | 7.546e-3 | 8.396e-3 | 7.740e-3 | 0.99996687 | 1.62 | 9.86 | 6.1× |
| fc2 | native | **PASS** | 2.820e-3 | **4.321e-7** (0 outside tol) | 3.432e-3 | **1.690e-3** | 0.99999329 | 9.50 | 9.84 | 1.0× |

Conclusion: the example is correct for what it was verified on, and the NPU arithmetic is exact (native fp32
intermediate matches to 1e-7); the accuracy loss is the BFP16 emulation, and so is the 4–6× speedup. Native bf16
with fp32 output is accurate to the bf16-input floor (~1.4–1.7e-3) but only ~CPU speed (8 threads) kernel-only.
