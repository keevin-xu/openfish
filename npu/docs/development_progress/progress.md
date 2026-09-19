# openfish/npu — progress (append-only)

Detailed session ledger (decisions, provenance, open questions):
`npu-adapting/9-16-npu-adapting-session/` (outside this repo).

## 2026-09-16 — Phase 0: CPU oracle (silu_mul template; covers all SUP kernels)

Env: zhang-ryzen2 (Ryzen AI MAX+ 395, 32 CPU, 30 GiB), slorado `ce74220`→`ba21b8e`, openfish `2e1a8cc`
(branch `npu-adapting` of keevin-xu/openfish), libtorch 2.9.0+rocm6.4 (CPU path),
input `test/PGXXXX230339/reads_1k.blow5` (md5 aec36d28647ad20cdf4b8d633f49fe54),
minimap2 2.24 + datamash 1.8 vs hg38noAlt.

1. Identity (gate item 3), `scripts/calculate_basecalling_accuracy.sh`:
   - fast@v5.0.0: median **0.9405915** (README 0.940374), n=1028 — PASS
   - hac@v5.0.0: median **0.9780595** (README 0.977594), n=1104 — PASS
   - sup@v5.0.0 `-C 128`: median **0.988506** (README 0.988561), n=1155 — PASS
     (default `-C 512` was SIGKILLed at 26.9 GB RSS; see debug_log)
2. Tensor dumps (gate item 4): slorado `ba21b8e` guarded `OPENFISH_DUMP` hooks; SUP layer 0, 4 of 128 rows,
   `~/p0/dump_sup` (28 files, 266 MB). Runtime shapes recorded in `../../TODO.md`.
3. References (gate item 5): `test/reference.py` (`silu_mul_ref`, `rmsnorm_ref`, `rotary_qkv_ref`, `gemm_ref`,
   plus `split_silu_mul_input` for the mlir-air (gate, up) order). `test/run_all.py` vs the dump:

   | check | shape | max_abs |
   |---|---|---|
   | silu_mul [y‖gate] | 4x1024x2048 | 3.576e-07 |
   | rmsnorm norm1 / norm2 | 4x1024x512 | 4.768e-07 / 7.153e-07 |
   | rotary q,k | 4x1024x3x8x64 | 0 |
   | gemm fc1 / fc2 | 4x1024x4096 / 4x1024x512 | 2.861e-06 / 6.199e-06 |
   | gemm wqkv / out_proj | 4x1024x3x8x64 / 4x1024x512 | 3.338e-06 / 1.013e-06 |
   | silu_mul swapped (negative control) | — | 4.183 (must differ) |

   All < 1e-5 → PASS.

**Phase 0: PASS** (all five gate items of `bc-phase-0-cpu-oracle`).

## 2026-09-16 — Phase 1: silu_mul kernel validation — PASS

Env: slorado `dbe9cab`, openfish `c1243fd` (npu-adapting), mlir-air wheel `0.0.1.2026090504+6746658`, XRT 2.21.75
userspace, NPU FW 1.1.2.65, amdxdna-dkms 7.0.0-rc1+git20260310. NPU idle before runs. Lock: `flock $NPU_LOCK`.
Dump: `~/p0/dump_sup8` (SUP L0, 8 rows; reference checks PASS on it too).

- Stock kernel (`kernels/silu_mul`, copied unmodified from mlir-air `silu_and_mul`): `make run` PASS, `make profile`
  4060.1 µs mean → 24.8 GB/s at N=16,777,216, 8×1 tiles, tile_n 4096 — reproduces the registry row.
- openfish harness `test/test_silu_mul_npu.py` (exit 0): randn PASS (1.024e-2); **real SUP activations through the
  `[y‖gate]` host split PASS (mean_rel_L1 8.858e-3, abs_err max 0.047)**; swapped negative control fails as required.
- Gate: (1) harness exists ✓ (2) element-wise PASS on full output, mean_rel_L1 recorded ✓ (3) 8/8 tiles, target for
  elementwise ✓ (4) registry row in `docs/registry_rows.md` (upstream via lab, not written into mlir-air) ✓.
- Host adapter cost at this N: f32→bf16 cast 11–12 ms + split 7 ms; one invoke incl. BO allocation and syncs 73 ms
  vs 4.3 ms kernel. Per-call BO allocation in the Python invoker dominates — the C backend must pre-allocate.

**Phase 1 (silu_mul): PASS.** Details: `phase1_kernels.md`.

## 2026-09-16 — Phase 2: silu_mul integrated into openfish/slorado — PASS

Backend: `src/nn_npu.{h,cpp}` (XRT full-ELF; device/context/kernel/BOs/run created once; process-lifetime
`flock($NPU_LOCK)`; bf16 conversion in C++, bit-exact vs ml_dtypes on 10M values; 16 launches of the verified
N=16,777,216 ELF per SUP call, zero-padded tail). Build `make npu=1` (slorado + openfish); call site
`TxModel.cpp` GatedMLP `#elif defined HAVE_NPU`. ELF harness re-validated first (identical PASS, 4110 µs).
Artifact `silu_mul_n16777216_t4096_h8x1.elf` sha256 7a2cbf77…b978. Commits: slorado `0565e1d`, openfish `ccbc046`.

- Smoke (reads_1.blow5): sequence identical to CPU. 18 calls / 288 launches; host in 5.2 s, kernel 1.3 s
  (4.6 ms/launch), host out 1.3 s.
- Gate (pre-registered): 18-layer, 1-row dumps of the first reads_1k batch (`-C 128`), CPU vs NPU binary:
  306/306 activations finite with whole-tensor cosine ≥ 0.99 (min 0.99897, L11 silu_mul out); 198 parameters
  bit-identical; everything upstream of L0 silu_mul bit-identical. enc_out cosine 0.99998 (L0) → 0.99964 (L17).
- Watch: per-position min cosine degrades with depth (enc_out 0.862 at L11; silu_mul out 0.765 at L14). Not a
  pre-registered gate; the Phase-3 identity score decides.

**Phase 2 (silu_mul): PASS.**

## 2026-09-16 — Phase 3: full basecaller with silu_mul on NPU — PASS

`~/p0/bin/slorado-npu basecaller -x cpu -C 128 sup@v5.0.0 reads_1k.blow5` (slorado `0565e1d`, openfish `ccbc046`):

| run | mean | q1 | median | q3 | n |
|---|---|---|---|---|---|
| CPU (Phase 0) | 0.95612597 | 0.9435065 | 0.988506 | 0.9956615 | 1155 |
| NPU silu_mul | 0.95459520 | 0.94265025 | 0.9881625 | 0.9957005 | 1164 |

Gate (pre-registered, median within ±0.001 of CPU): Δ = −0.000344 → **PASS**. Exact sequence match 402/1000
(40.2%), total bases −0.033%, mean identity −0.0015 (watch). Wall 25:39 vs 20:12 (shared machine, load 6–10);
`ff` 940 s vs 607 s. `[nn_npu]` 738 calls, 11,808 launches: host in 271.8 s, kernel 55.6 s (4.7 ms/launch),
host out 53.4 s. As expected for Tier 1, the NPU path is slower than torch CPU for this memory-bound op
(≈516 ms vs ≈65 ms per call, CPU figure inferred from the ff delta).

**silu_mul: Phases 0–3 PASS. G1 met.**

## 2026-09-16 — Tier 2 start: GEMM fc1+fc2 on NPU — Phase 2 + Phase 3 PASS

Root cause of registry-tolerance failures on real data: BFP16 emulation in the aie2p mmul (bench/results.md).
Integrated `openfish_linear_npu` (fused-cast ELF, M=4096, f32 accumulator output, resident bf16 weights,
`OPENFISH_NPU_OPS`), both mmul variants. Phase 2 (18 layers): bfp16 enc_out cosine ≥ 0.9998, native ≥ 0.99999.
Phase 3 (sup, reads_1k, -C 128, 16 host threads; pre-registered ±0.001):

| variant | median | Δ vs CPU | wall | ff | CPU time |
|---|---|---|---|---|---|
| CPU | 0.988506 | — | 20:12 | 607 s | 14,327 s |
| bfp16 | 0.988272 | −0.000234 PASS | 16:45 (−17%) | 411 s | 9,010 s |
| native | 0.988550 | +0.000044 PASS | 27:38 (+37%) | 1,010 s | 9,844 s |

bfp16 fc1: host in 21.6 s, kernel 102.7 s (4.3 ms/launch), host out 58.7 s; fc2: 62.0 / 40.8 / 14.2 s.
Shared machine (load 5–10). First end-to-end NPU speedup; Phase 1 passes only for native.

## 2026-09-19 — iGPU base + NPU (coverage study, research-meeting direction)

iGPU baseline (gfx1151, ROCm 7.2.1, libtorch 2.9.1+rocm7.2.1 in `~/torch-rocm72`, fp16): fast 5.1 s, hac 16.3 s,
sup 179.8 s on reads_1k, all within ±0.0002 of CPU identity. `npu=1` became an add-on to `rocm=1`
(`slorado-rocm-npu`): NPU hooks copy fp16 GPU tensors to host fp32 and back. Kernel targets and status:
`npu-adapting/9-16-npu-adapting-session/kernel-targets-igpu-npu.md`.

Per-op on the iGPU base (sup, reads_1k, -C 128; gate median ±0.001 of iGPU 0.9886105; Phase 2 18-layer cosine ≥ 0.99):

| NPU op(s) | Phase 1 (real data) | Phase 2 | median | Δ vs iGPU | time (iGPU 179.8 s) | energy (iGPU ≈15.2 kJ) |
|---|---|---|---|---|---|---|
| fc1+fc2 bfp16 | FAIL element-wise, cos ≥ 0.99997 | PASS | 0.988557 | −0.000054 | 734.1 s | 47.7 kJ |
| fc1+fc2 native | PASS | PASS | 0.988359 | −0.000252 | 1311.0 s | 61.6 kJ |
| silu_mul | PASS | PASS | 0.988640 | +0.000030 | 695.5 s | 54.1 kJ |
| fc1+fc2+silu_mul bfp16 | — | PASS | 0.988487 | −0.000124 | 1217.6 s | 76.8 kJ |
| upsample bfp16 | FAIL 13.1% (native PASS) | PASS | 0.988528 | −0.000083 | 187.2 s | 15.8 kJ |
| crf bfp16 | FAIL 30.8% (native PASS) | PASS | 0.988674 | +0.000064 | 213.7 s | 17.8 kJ |
| out_proj bfp16 | FAIL 0.01% (native PASS) | PASS | 0.988557 | −0.000054 | 280.3 s | 23.4 kJ |
| wqkv bfp16 | FAIL 1.1% (native PASS) | PASS | 0.988476 | −0.000135 | 346.6 s | 28.0 kJ |

Every substitution is correct at the identity gate; none is faster or cheaper than the iGPU alone (GPU↔host copies
+ fp16/fp32/bf16 conversion + NPU kernels slower than the iGPU's). Cost scales with call count per read.
Combined all-ops run (fc1, fc2, silu_mul, wqkv, out_proj, upsample, crf): Phase 2 PASS (bfp16 and native); Phase 3 running.
