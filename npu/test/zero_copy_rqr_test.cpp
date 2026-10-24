// Debug: zero-copy rms_qkv_rope (development_progress/debug_log.md 2026-10-24) — the NPU's writes to iGPU buffers
// imported into XRT never reach the iGPU for this kernel, while the same mechanism works for the GEMM/silu_mul/axpy
// kernels (buffers >= 4 MiB; these outputs are 512 KiB).
//
// Part 1: for allocation sizes 256 KiB .. 64 MiB, does hipMalloc / hipExtMallocWithFlags(uncached) hand out a
//         sub-range of a larger allocation? (hipMemGetAddressRange base/size vs the returned pointer/request.) If so,
//         the dma-buf exported for the range may cover the whole parent allocation, and XRT imports it from offset 0.
// Part 2: the rms_qkv_rope ELF with random inputs; reference = all host XRT BOs. Then one arg at a time (x, normed,
//         v, q_roped, k_roped) replaced by an iGPU buffer imported into XRT, at the exact size and padded to 64 MiB;
//         outputs read back through the iGPU and compared bit-for-bit with the reference.
// Part 3: all five on imported buffers at once, (a) on a fresh xrt::run and (b) re-bound with set_arg on a run that was
//         already launched with host BOs (what nn_npu.cpp does); and (c) host BOs re-bound on a used run.
//
//   hipcc -x hip --offload-arch=gfx1151 -O2 -std=c++17 zero_copy_rqr_test.cpp -lxrt_coreutil -o zero_copy_rqr_test
//   LD_LIBRARY_PATH=/opt/rocm/lib flock -x -w 600 $HOME/.npu.lock zero_copy_rqr_test <artifacts dir>

#include <hip/hip_runtime.h>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_OK(x)                                                                                    \
    do {                                                                                             \
        hipError_t e_ = (x);                                                                         \
        if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); \
    } while (0)

static void *hip_alloc(uint64_t bytes, bool uncached) {
    void *p = nullptr;
    if (uncached) {
        HIP_OK(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached));
    } else {
        HIP_OK(hipMalloc(&p, bytes));
    }
    return p;
}

static xrt::bo import_bo(xrt::device &dev, void *p, uint64_t bytes) {
    int fd = -1;
    HIP_OK(hipMemGetHandleForAddressRange(&fd, (hipDeviceptr_t)p, bytes, hipMemRangeHandleTypeDmaBufFd, 0));
    return xrt::bo(dev, xrt::pid_type{getpid()}, (xrt::bo::export_handle)fd);
}

static uint16_t rnd_bf16(uint32_t &h, float scale) {
    h = h * 1664525u + 1013904223u;
    float f = ((int)(h >> 8 & 0xffff) - 32768) / 32768.0f * scale;
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <artifacts dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];

    // ---- part 1: sub-allocation
    printf("part 1: hipMalloc sub-allocation (pointer offset inside the underlying allocation)\n");
    printf("| request | kind | offset in parent | parent size |\n|---|---|---|---|\n");
    for (bool unc : {false, true}) {
        std::vector<void *> keep;
        for (uint64_t kb : {256, 512, 1024, 2048, 4096, 8192, 65536}) {
            for (int rep = 0; rep < 2; ++rep) {  // two in a row: a pool shows up as the second one at a nonzero offset
                void *p = hip_alloc(kb * 1024, unc);
                keep.push_back(p);
                hipDeviceptr_t base = nullptr;
                size_t size = 0;
                HIP_OK(hipMemGetAddressRange(&base, &size, (hipDeviceptr_t)p));
                printf("| %lu KiB #%d | %s | %ld | %lu KiB |\n", (unsigned long)kb, rep, unc ? "uncached" : "hipMalloc",
                       (long)((char *)p - (char *)base), (unsigned long)(size / 1024));
            }
        }
        for (void *p : keep) (void)hipFree(p);
    }

    // ---- part 2: rms_qkv_rope, one arg at a time on an imported iGPU buffer
    const uint64_t m = 512, d = 512, act = m * d * 2, w = d * d * 2;
    xrt::device dev(0);
    xrt::elf elf(dir + "/rms_qkv_rope.elf");
    xrt::hw_context ctx(dev, elf);
    xrt::ext::kernel kernel(ctx, "main:rms_gemms_rope");
    // args: 0 x_in, 1 norm_w, 2 normed, 3 wq, 4 q, 5 wk, 6 k, 7 wv, 8 v, 9 lut_q, 10 lut_k, 11 q_roped, 12 k_roped
    const uint64_t sizes[13] = {act, d * 2, act, w, act, w, act, w, act, act, act, act, act};
    std::vector<xrt::bo> host;
    uint32_t h = 4242;
    for (int i = 0; i < 13; ++i) {
        host.push_back(xrt::ext::bo(dev, sizes[i]));
        auto *p = host.back().map<uint16_t *>();
        const bool input = i == 0 || i == 1 || i == 3 || i == 5 || i == 7 || i == 9 || i == 10;
        for (uint64_t j = 0; j < sizes[i] / 2; ++j) p[j] = input ? rnd_bf16(h, i == 0 ? 2.0f : 0.1f) : 0;
        host.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    auto launch = [&](std::vector<xrt::bo> &args) {
        xrt::run run(kernel);
        for (int i = 0; i < 13; ++i) run.set_arg(i, args[i]);
        run.start();
        if (run.wait2(std::chrono::milliseconds(60000)) == std::cv_status::timeout) throw std::runtime_error("timeout");
    };
    launch(host);
    std::vector<std::vector<uint16_t>> ref(13);
    for (int i : {2, 8, 11, 12}) {
        host[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto *p = host[i].map<uint16_t *>();
        ref[i].assign(p, p + act / 2);
    }
    launch(host);  // determinism
    bool det = true;
    for (int i : {2, 8, 11, 12}) {
        host[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        det &= std::memcmp(host[i].map<uint16_t *>(), ref[i].data(), act) == 0;
    }
    printf("\npart 2: rms_qkv_rope (m %lu, d %lu); reference deterministic: %s\n", (unsigned long)m, (unsigned long)d,
           det ? "yes" : "NO");
    printf("| imported arg | iGPU allocation | outputs correct (normed, v, q_roped, k_roped) | note |\n|---|---|---|---|\n");
    const char *names[13] = {"x_in", "", "normed", "", "", "", "", "", "v", "", "", "q_roped", "k_roped"};
    for (int arg : {0, 2, 8, 11, 12}) {
        for (int variant = 0; variant < 3; ++variant) {
            const bool unc = variant == 1;
            const uint64_t alloc = variant == 2 ? 64ull << 20 : act;  // exact size, exact uncached, padded to 64 MiB
            std::string note;
            std::string res;
            void *dp = nullptr;
            try {
                dp = hip_alloc(alloc, unc);
                std::vector<xrt::bo> args = host;
                args[arg] = import_bo(dev, dp, alloc);
                // input: the same x values via the iGPU; output: a 0xFF marker the NPU must overwrite
                if (arg == 0) {
                    HIP_OK(hipMemcpy(dp, host[0].map<uint16_t *>(), act, hipMemcpyHostToDevice));
                } else {
                    HIP_OK(hipMemset(dp, 0xFF, act));
                }
                HIP_OK(hipDeviceSynchronize());
                args[arg].sync(XCL_BO_SYNC_BO_TO_DEVICE);
                for (int i : {2, 8, 11, 12}) {
                    if (i != arg) {
                        std::memset(host[i].map<uint8_t *>(), 0xFF, act);
                        host[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    }
                }
                launch(args);
                for (int i : {2, 8, 11, 12}) {
                    std::vector<uint16_t> got(act / 2);
                    args[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    if (i == arg) {
                        HIP_OK(hipMemcpy(got.data(), dp, act, hipMemcpyDeviceToHost));
                    } else {
                        std::memcpy(got.data(), host[i].map<uint16_t *>(), act);
                    }
                    uint64_t bad = 0, marker = 0;
                    for (uint64_t j = 0; j < act / 2; ++j) {
                        bad += got[j] != ref[i][j];
                        marker += got[j] == 0xFFFF;
                    }
                    res += std::string(bad ? "WRONG" : "ok") + (bad ? "(" + std::to_string(bad) + ", marker " + std::to_string(marker) + ")" : "") + " ";
                }
            } catch (const std::exception &e) {
                note = e.what();
            }
            if (dp) (void)hipFree(dp);
            printf("| %d %s | %s %lu KiB | %s | %s |\n", arg, names[arg], unc ? "uncached" : "hipMalloc", (unsigned long)(alloc >> 10),
                   res.c_str(), note.c_str());
        }
    }

    // ---- part 3: all five imported at once; fresh run vs set_arg re-binding on a used run
    printf("\npart 3: args 0, 2, 8, 11, 12 together\n| case | outputs correct (normed, v, q_roped, k_roped) |\n|---|---|\n");
    std::vector<void *> dps;
    std::vector<xrt::bo> args = host;
    for (int i : {0, 2, 8, 11, 12}) {
        void *dp = hip_alloc(act, false);
        dps.push_back(dp);
        args[i] = import_bo(dev, dp, act);
    }
    auto check = [&](const char *what, std::vector<xrt::bo> &a, bool imported, xrt::run *reuse) {
        // inputs and markers
        if (imported) {
            HIP_OK(hipMemcpy(dps[0], host[0].map<uint16_t *>(), act, hipMemcpyHostToDevice));
            for (int k = 1; k < 5; ++k) HIP_OK(hipMemset(dps[k], 0xFF, act));
            HIP_OK(hipDeviceSynchronize());
        } else {
            for (int i : {2, 8, 11, 12}) {
                std::memset(host[i].map<uint8_t *>(), 0xFF, act);
                host[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }
        }
        for (int i : {0, 2, 8, 11, 12}) a[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (reuse) {
            for (int i = 0; i < 13; ++i) reuse->set_arg(i, a[i]);
            reuse->start();
            if (reuse->wait2(std::chrono::milliseconds(60000)) == std::cv_status::timeout) throw std::runtime_error("timeout");
        } else {
            launch(a);
        }
        std::string res;
        int k = 1;
        for (int i : {2, 8, 11, 12}) {
            std::vector<uint16_t> got(act / 2);
            a[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            if (imported) {
                HIP_OK(hipMemcpy(got.data(), dps[k++], act, hipMemcpyDeviceToHost));
            } else {
                std::memcpy(got.data(), host[i].map<uint16_t *>(), act);
            }
            uint64_t bad = 0, marker = 0;
            for (uint64_t j = 0; j < act / 2; ++j) {
                bad += got[j] != ref[i][j];
                marker += got[j] == 0xFFFF;
            }
            res += std::string(bad ? "WRONG" : "ok") + (bad ? "(" + std::to_string(bad) + ", marker " + std::to_string(marker) + ")" : "") + " ";
        }
        printf("| %s | %s |\n", what, res.c_str());
    };
    check("(a) fresh run, all imported", args, true, nullptr);
    xrt::run used(kernel);
    for (int i = 0; i < 13; ++i) used.set_arg(i, host[i]);
    used.start();
    used.wait2(std::chrono::milliseconds(60000));
    check("(b) used run, set_arg to imported", args, true, &used);
    check("(c) same run, set_arg back to host", host, false, &used);
    check("(d) same run, set_arg to imported again", args, true, &used);
    xrt::run fresh_imported(kernel);
    for (int i = 0; i < 13; ++i) fresh_imported.set_arg(i, args[i]);
    check("(e) run created with imported, re-launched", args, true, &fresh_imported);
    for (void *p : dps) (void)hipFree(p);

    // ---- part 4: which combinations of imported args work (fresh run each)
    printf("\npart 4: subsets of imported args (fresh run each; others host BOs)\n| alloc | imported | normed | v | q_roped | k_roped |\n|---|---|---|---|---|---|\n");
    const std::vector<std::vector<int>> subsets = {{2, 8}, {0, 2}, {2, 8, 11, 12}, {0, 2, 8, 11, 12}};
    for (uint64_t alloc : {act, act * 2, 2ull << 20, 4ull << 20, 64ull << 20})
    for (const auto &sub : subsets) {
        std::vector<xrt::bo> a = host;
        std::vector<std::pair<int, void *>> imp;
        for (int i : sub) {
            void *dp = hip_alloc(alloc, false);
            imp.emplace_back(i, dp);
            a[i] = import_bo(dev, dp, alloc);
            if (i == 0) {
                HIP_OK(hipMemcpy(dp, host[0].map<uint16_t *>(), act, hipMemcpyHostToDevice));
            } else {
                HIP_OK(hipMemset(dp, 0xFF, act));
            }
        }
        HIP_OK(hipDeviceSynchronize());
        for (int i : {2, 8, 11, 12}) {
            std::memset(host[i].map<uint8_t *>(), 0xFF, act);
            host[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        for (int i : {0, 2, 8, 11, 12}) a[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        launch(a);
        std::string label, res;
        for (int i : sub) label += std::to_string(i) + " ";
        for (int i : {2, 8, 11, 12}) {
            std::vector<uint16_t> got(act / 2);
            a[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            void *dp = nullptr;
            for (auto &pr : imp) if (pr.first == i) dp = pr.second;
            if (dp) {
                HIP_OK(hipMemcpy(got.data(), dp, act, hipMemcpyDeviceToHost));
            } else {
                std::memcpy(got.data(), host[i].map<uint16_t *>(), act);
            }
            uint64_t bad = 0;
            for (uint64_t j = 0; j < act / 2; ++j) bad += got[j] != ref[i][j];
            res += std::string(" | ") + (bad ? "WRONG" : "ok") + (dp ? "*" : "");
        }
        printf("| %lu KiB | %s%s |\n", (unsigned long)(alloc >> 10), label.c_str(), res.c_str());
        for (auto &pr : imp) (void)hipFree(pr.second);
    }
    printf("(* = that output is an imported iGPU buffer)\n");
    return 0;
}
