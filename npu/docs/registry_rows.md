# Registry rows for upstream (mlir-air `kernel_registry`)

We do not edit mlir-air (CLAUDE.md). Rows below use the exact column schema of the
corresponding "tested shapes" table in `kernel_registry/supported_kernels.md` and
`details/<Kernel>_bf16.md`, so they can be pasted upstream via the lab (Jiajie Li).
Append-only.

## SiLU-and-Mul (BF16) — `details/SiLU_Mul_bf16.md` / `supported_kernels.md` §SiLU-and-Mul — tested shapes

| N | (as 2-D) | best config (hx/hy/tile_n) | latency | bandwidth | mean_rel_L1 | abs_err max | Status |
|---|---|---|---|---|---|---|---|
| 16777216 | 16384×1024 (slorado-sup@v5.0.0: B·T = 8·1024 rows × K=2048, harness uses 8 of 128 batch rows) | 8/1/4096 | 4060 µs | 24.8 GB/s | 1.0e-2 (randn) / 8.9e-3 (real SUP activations) | 0.125 (randn) / 0.047 (real) | ✅ (slorado-sup@v5.0.0 GatedMLP SiLU-Mul) |

> **slorado-sup@v5.0.0** (nanopore basecaller, openfish `c1243fd`): N = 16777216 is the registry's existing shape; the new
> information is real transformer activations. `L0` fc1 output of SUP v5.0.0 (8 of 128 chunks), split host-side from
> openfish's `[y ‖ gate]` row layout into `(gate, up)`: PASS at `rtol 1.6e-2, atol 8e-2`, `mean_rel_L1 = 8.858e-3`,
> `rel_err max = 0.200`, `abs_err max = 0.0469` — cleaner than randn because |gate| is smaller (fc1 |x| p99 ≈ 1.1).
> vs slorado's fp32 CPU output (no bf16 input rounding): `mean_rel_L1 = 9.289e-3`, `max_abs = 0.0611`, 0 elements
> outside tolerance. Latency from `make profile` (30 iters, 10 warmup, kernel-only mean); the openfish harness
> measured 4267 µs (23.6 GB/s) in a separate run. Production SUP call is N = 268,435,456 per layer (batch 128) —
> not yet measured (16 × this shape if chunked host-side).
