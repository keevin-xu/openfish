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

#ifdef __cplusplus
}
#endif

#endif // NN_NPU_H
