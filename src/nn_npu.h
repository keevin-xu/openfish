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

// Whether op (e.g. "silu_mul", "fc1", "fc2") is listed in OPENFISH_NPU_OPS (default "silu_mul").
int npu_op_enabled(const char *op);

// Linear without bias on the NPU (bf16 GEMM, fp32 result): out[rows, N] = x[rows, K] @ weight[N, K].T.
// weight is cached on the device by pointer, so it must stay alive and unchanged.
void linear_npu(const char *op, const float *x, float *out, const float *weight, uint64_t rows, uint64_t K, uint64_t N);

#ifdef __cplusplus
}
#endif

#endif // NN_NPU_H
