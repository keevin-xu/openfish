// Packs a raw float32 [N, K] weight file with src/npu_bfp16_pack.h and writes the packed bytes (see check_bfp16_pack.py).
#include "../../src/npu_bfp16_pack.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s in.f32 N K tile_n tile_k_l1 out.u8\n", argv[0]);
        return 2;
    }
    const uint64_t N = std::strtoull(argv[2], nullptr, 10), K = std::strtoull(argv[3], nullptr, 10);
    const uint64_t tn = std::strtoull(argv[4], nullptr, 10), tk = std::strtoull(argv[5], nullptr, 10);
    std::vector<float> w(N * K);
    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(w.data(), 4, w.size(), f) != w.size()) return 1;
    fclose(f);
    std::vector<uint8_t> out((N / tn) * (K / tk) * npu_bfp16::tile_bytes(tn, tk));
    npu_bfp16::pack_weight(w.data(), N, K, tn, tk, out.data(), 0, N / tn);
    f = fopen(argv[6], "wb");
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    return 0;
}
