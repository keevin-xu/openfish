#ifndef NN_NPU_H
#define NN_NPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// SiLU-and-Mul on the AMD XDNA2 NPU (mlir-air silu_and_mul, bf16).
// x: float32 [MN, 2K] rows of [y ‖ gate]; o: float32 [MN, K] = SiLU(gate) * y.
// Split host-side into launches of the compiled N; the last launch is zero-padded.
void silu_mul_npu(const float *x, float *o, uint64_t MN, uint64_t K);

// Whether op (e.g. "silu_mul", "fc1", "fc2") is listed in OPENFISH_NPU_OPS (default: none).
int npu_op_enabled(const char *op);

// Linear without bias on the NPU (bf16 GEMM, fp32 result): out[rows, N] = x[rows, K] @ weight[N, K].T.
// weight is cached on the device by pointer, so it must stay alive and unchanged.
// RMSNorm rows of N: y = x * rsqrt(mean(x^2) + 1e-5) * weight (residual/alpha folded in by the caller).
void rmsnorm_npu(const float *x, const float *weight, float *y, uint64_t rows, uint64_t N);

// Half-split RoPE on rows of head_dim; lut holds m rows of [cos(D/2), sin(D/2)] repeated every m rows.
void rope_npu(const float *x, const float *lut, float *y, uint64_t rows, uint64_t head_dim);

void linear_npu(const char *op, const float *x, float *out, const float *weight, uint64_t rows, uint64_t K, uint64_t N);

// bf16 host I/O variants (raw bf16 bit patterns), used when tensors come from the iGPU
void silu_mul_npu_bf16(const uint16_t *x, uint16_t *o, uint64_t MN, uint64_t K);
void linear_npu_bf16(const char *op, const uint16_t *x, uint16_t *out, const float *weight, uint64_t rows, uint64_t K, uint64_t N);
void rmsnorm_npu_bf16(const uint16_t *x, const float *weight, uint16_t *y, uint64_t rows, uint64_t N);
void rope_npu_bf16(const uint16_t *x, const float *lut, uint16_t *y, uint64_t rows, uint64_t head_dim);
// bf16 in, f32 out (fp32 accumulator; with an f32-out GEMM ELF this is the only valid output)
void linear_npu_bf16_f32(const char *op, const uint16_t *x, float *out, const float *weight, uint64_t rows, uint64_t K, uint64_t N);
// out = alpha * x + y (mlir-air axpy; alpha is compiled into the ELF and checked)
void axpy_npu_bf16(float alpha, const uint16_t *x, const uint16_t *y, uint16_t *out, uint64_t n);
// out = x * sigmoid(x) (mlir-air silu)
void silu_npu_bf16(const uint16_t *x, uint16_t *out, uint64_t n);
// normed = RMSNorm(x) * norm_w; qkv[rows, 3, d] = normed @ wqkv[3d, d].T with RoPE on the q and k thirds
// (mlir-air rms_gemms_rope_multi); lut[T * heads, head_dim] rows of [cos | sin] for (position, head); rows = N * T
void rms_qkv_rope_npu_bf16(const uint16_t *x, const float *norm_w, const float *wqkv, const float *lut, uint16_t *normed,
                           uint16_t *qkv, uint64_t rows, uint64_t T);

#ifdef __cplusplus
}
#endif

#endif // NN_NPU_H
