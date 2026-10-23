# NPU porting targets (iGPU base + NPU), ordered

Updated 2026-10-23. Order: **A** ports of mlir-air examples first, **B** zero-copy, **C** extra features last.
Model is SUP@v5.0.0 unless a row says hac/fast. Already on the NPU (gated or slotted in): fc1, fc2, silu_mul,
wqkv, out_proj, upsample, crf, rmsnorm, rope, conv (im2col + GEMM).

## Legend: what "Kind" means

| Kind | Meaning |
|---|---|
| **EXAMPLE** | Direct from an mlir-air `programming_examples/` kernel or builder. We copy it into `npu/kernels/`, set our shapes, build an ELF, and wire a call site. The NPU kernel logic itself is unchanged. |
| **EXAMPLE + MOD** | Starts from an mlir-air example, but the example does not cover our case. We must hand-write a change to the kernel or builder (noted in the row). |
| **HAND-CODED EXTRA** | No mlir-air example exists. A feature we write ourselves (host glue, scheduling, build logic). It adds speed or accuracy, not NPU coverage. |

## A. Ports of mlir-air examples

| # | Kind | Item | mlir-air example | slorado target | Status |
|---|---|---|---|---|---|
| A1 | EXAMPLE | FFN as one NPU call (fc1 → silu_mul → fc2) | `ffn_swiglu/prefill` | GatedMLP | blocked: upstream `build_ffn_module` raises ValueError at the installed mlir-air |
| A2 | EXAMPLE | RMSNorm → QKV GEMM → RoPE in one multi-launch ELF | `llms/shared/builders/rms_gemms_rope_multi.py` | norm2 (layer i) → wqkv + RoPE (layer i+1) | not started |
| A3 | EXAMPLE + MOD | out_proj → residual add → FFN. **Mod:** insert an RMSNorm after the add; scale the residual by α = 2.449 (deepnorm) | `llms/shared/builders/o_ffn_multi.py` | out_proj + norm1 + GatedMLP | not started |
| A4 | EXAMPLE + MOD | Windowed attention. **Mod:** the example only skips causal (future) blocks; change the skip rule to a ±127/128 band (~5 of 16 K-blocks per Q-block). The `dataflow_based` variant's mask argument is declared but never read, so it cannot be used as-is | `flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py` | MHA SDPA | not started |
| A5 | EXAMPLE | GEMM with fp32 output (replaces our fused-cast GEMM; should recover the accuracy lost by returning bf16) | `matrix_multiplication/bf16_in_fp32_out` | all GEMM call sites | not started |
| A6 | EXAMPLE | GEMM with bfp16 pre-converted weights (between the lossy bfp16 build and the slow native build) | `matrix_multiplication/bf16_x_bfp16` | all GEMM call sites | not started |
| A7 | EXAMPLE | Deepnorm residual α·x + y | `axpy` | `run_norm` residual (folded on the iGPU today) | not started |
| A8 | EXAMPLE | Conv-stack swish | `silu` | `ConvStack` activations | not started |
| A9 | EXAMPLE + MOD | hac/fast CRF tail: CRF linear (GEMM) + tanh. **Mod:** no tanh example; build tanh = 2·sigmoid(2x) − 1 | GEMM + `sigmoid` | `CRFModel.cpp` LinearCRF | not started |
| A10 | EXAMPLE | hac/fast conv stack through the existing im2col + GEMM path | GEMM (same as SUP conv) | `ConvStack` (fast/hac) | not started |
| A11 | EXAMPLE | Softmax (only useful if attention stays split up) | `softmax` | SDPA internals | low value |

## B. Infrastructure

| # | Kind | Item | How | Status |
|---|---|---|---|---|
| B1 | **HAND-CODED EXTRA** | Zero-copy iGPU↔NPU buffers: one buffer seen by both devices instead of iGPU buffer → host RAM → NPU buffer and back | Export the XRT buffer as dma-buf → `hipImportExternalMemory` (fallback: `hipHostRegister` on the XRT mapping); `torch::from_blob`; sync with HIP events + `bo.sync`. In `src/nn_npu.cpp` + the TxModel/CRFModel hooks | not started. Step 1: standalone import test. Est. 74.8 → ~33–40 s on the 50-read run |

## C. Extra features (last priority)

| # | Kind | Item | Why (measured) | Status |
|---|---|---|---|---|
| C1 | **HAND-CODED EXTRA** | Keep intermediate results in NPU buffers between chained NPU ops (no host round trip). Pattern from upstream skill `opt-buffer-object-reuse` | host pack + copy 18.45 s; silu_mul pack alone 2.93 s | not started |
| C2 | **HAND-CODED EXTRA** | Run iGPU and NPU at the same time: split each batch (~85% iGPU / 15% NPU), double-buffered async NPU launches | only route to beat the iGPU alone (9.8 s) | not started |
| C3 | **HAND-CODED EXTRA** | Fewer, bigger NPU launches (batch q and k for RoPE, raise the 8192-row cap) | RoPE: 9,216 launches, 13.1 s | not started |
| C4 | **HAND-CODED EXTRA** | Per-op GEMM build selection (native for conv, bfp16 elsewhere) | conv identity −0.019 (bfp16) vs −0.0012 (native) | not started |
| C5 | **HAND-CODED EXTRA** | Move remaining host packing (im2col, padding, silu split) into iGPU kernels | part of the 18.45 s host time | not started |
| C6 | **HAND-CODED EXTRA** | Individual identity gates for RMSNorm and RoPE | both slotted in without gates; all-10 run fails identity | not started |

## Checked and not portable

`conv1d_depthwise` (SUP convs are dense), `swiglu` (computes silu(x·gate)·(x·up), not SUP's silu(gate)·up, which
`silu_and_mul` already covers), `matrix_vector_multiplication` / `attention_decode` / `fused_decode*`
(token-by-token decode shapes), `dequant_awq` / q4 models (4-bit weights), `cascade_reduction`, `conv2d`,
beam-search decode.

Numbers: 50-read all-10 run (iGPU 9.8 s vs iGPU+NPU 74.8 s, NPU kernels 17.6 s). Ledger:
`npu-adapting/9-16-npu-adapting-session/results-ledger.md` R18–R20.
