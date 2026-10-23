// bfp16ebs8 weight packing for the mlir-air bf16_x_bfp16 GEMM (kernels/gemm_bfp16w): a C++ port of
// pack_b_bfp16ebs8() in matmul_bf16_x_bfp16.py, byte-identical to it (npu/test/check_bfp16_pack.py).
// B[k, n] = weight[n, k] (bf16-rounded), packed into [N/tile_n, K/tile_k_l1, tile_bytes] where each 9-byte
// record is [shared exponent, 8 int8 mantissas] for 8 K-contiguous values of one output column n.
#ifndef NPU_BFP16_PACK_H
#define NPU_BFP16_PACK_H

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace npu_bfp16 {

constexpr uint64_t kBlock = 8, kRecordBytes = 9;

inline uint64_t tile_bytes(uint64_t tile_n, uint64_t tile_k_l1) { return tile_n * tile_k_l1 / kBlock * kRecordBytes; }

// One record from 8 float32 values (already bf16-rounded): floatToBfp16() order, truncate then shift.
inline void pack_record(const float *v, uint8_t *rec) {
    uint32_t bits[kBlock];
    int exp[kBlock];
    int max_exp = 0;
    for (uint64_t i = 0; i < kBlock; ++i) {
        std::memcpy(&bits[i], &v[i], 4);
        exp[i] = (int)((bits[i] >> 23) & 0xff);
        max_exp = std::max(max_exp, exp[i]);
    }
    rec[0] = (uint8_t)max_exp;
    for (uint64_t i = 0; i < kBlock; ++i) {
        const bool sign = bits[i] >> 31;
        const uint32_t mant = (bits[i] & 0x7fffffu) | (exp[i] ? 0x800000u : 0u);
        const uint32_t sm = sign ? (~mant + 1u) : mant;
        const int32_t b8 = (int8_t)(uint8_t)(sm >> 17);
        const int shift = max_exp - exp[i];
        const int8_t aligned = shift >= 32 ? (int8_t)(sign ? -1 : 0) : (int8_t)(b8 >> std::min(shift, 31));
        std::memcpy(&rec[1 + i], &aligned, 1);
    }
}

// weight[N, K] row-major float32 (already bf16-rounded by the caller) -> packed bytes (size (N/tn)*(K/tk)*tile_bytes).
// Record order per (nb, kb) tile: nbi (tile_n/8) x kbi (tile_k_l1/8) x n_i (8).
inline void pack_weight(const float *weight, uint64_t N, uint64_t K, uint64_t tile_n, uint64_t tile_k_l1, uint8_t *out,
                        uint64_t nb_begin, uint64_t nb_end) {
    const uint64_t Kb = K / tile_k_l1, NB_in = tile_n / kBlock, KB_in = tile_k_l1 / kBlock;
    const uint64_t tb = tile_bytes(tile_n, tile_k_l1);
    for (uint64_t nb = nb_begin; nb < nb_end; ++nb) {
        for (uint64_t kb = 0; kb < Kb; ++kb) {
            uint8_t *rec = out + (nb * Kb + kb) * tb;
            for (uint64_t nbi = 0; nbi < NB_in; ++nbi) {
                for (uint64_t kbi = 0; kbi < KB_in; ++kbi) {
                    for (uint64_t ni = 0; ni < kBlock; ++ni, rec += kRecordBytes) {
                        const uint64_t n = nb * tile_n + nbi * kBlock + ni;
                        const uint64_t k0 = kb * tile_k_l1 + kbi * kBlock;
                        pack_record(weight + n * K + k0, rec);
                    }
                }
            }
        }
    }
}

}  // namespace npu_bfp16

#endif  // NPU_BFP16_PACK_H
