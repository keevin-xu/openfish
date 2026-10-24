// Zero-copy iGPU <-> NPU feasibility test (porting_targets.md B1, step 1). Standalone: not linked into slorado.
//
// Question: can one buffer in the APU's shared memory be written by an iGPU (HIP) kernel, read and written by an
// NPU (XRT) kernel, and read back by the iGPU, with no copies? Four ways to share it are tried:
//   A xrt_export   XRT allocates (xrt::ext::bo), exports a dma-buf fd, HIP imports it (hipImportExternalMemory)
//   B xrt_hostreg  XRT allocates, HIP registers the BO's host mapping (hipHostRegister)
//   C hip_host     HIP allocates pinned host memory (hipHostMalloc), XRT wraps the pointer (xrt::ext::bo userptr)
//   D hip_export   HIP allocates device memory (hipMalloc), exports a dma-buf fd, XRT imports it
// Each is checked with the axpy NPU kernel (out = alpha*x + y, bf16, n elements; npu/kernels/axpy) on data the
// iGPU writes, with the output pre-filled by the iGPU with a NaN sentinel, over several rounds of fresh data.
// The copy path slorado uses today (iGPU -> host -> BO -> NPU -> BO -> host -> iGPU) is timed as the baseline.
//
// Build + run on zhang-ryzen2, in a shell WITHOUT ~/npu-env.sh (ROCm), see build_zero_copy_test.sh:
//   LD_LIBRARY_PATH=/opt/rocm/lib flock -x -w 600 $HOME/.npu.lock ~/p0/bin/zero_copy_test \
//       ~/slorado/openfish/npu/artifacts/axpy.json
// Exit 0 if at least one method passes.

#include <hip/hip_runtime.h>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>

#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_OK(x)                                                                                    \
    do {                                                                                             \
        hipError_t e_ = (x);                                                                         \
        if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); \
    } while (0)

using clock_type = std::chrono::steady_clock;

static double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

__host__ __device__ inline uint32_t fbits(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    return v.u;
}
__host__ __device__ inline uint16_t f2bf(float f) {
    uint32_t u = fbits(f);
    u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
    return (uint16_t)(u >> 16);
}
__host__ __device__ inline float bf2f(uint16_t h) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}
// deterministic values in [-4, 4] that change with the round's seed
__host__ __device__ inline float pattern(uint64_t i, uint32_t seed) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return ((int)(h % 2001u) - 1000) / 250.0f;
}

constexpr uint16_t kSentinel = 0x7fc0;  // bf16 NaN: any of these left in the output = the NPU result was not seen

__global__ void fill(uint16_t *x, uint16_t *y, uint16_t *out, uint64_t n, uint32_t seed) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = f2bf(pattern(i, seed));
    y[i] = f2bf(pattern(i, seed ^ 0x00abcdefu));
    out[i] = kSentinel;
}

__global__ void copy16(const uint16_t *src, uint16_t *dst, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

static std::string manifest_field(const std::string &json, const std::string &key) {
    std::smatch m;
    if (!std::regex_search(json, m, std::regex("\"" + key + "\"\\s*:\\s*\"?([^\",}\\n]+)\"?"))) {
        throw std::runtime_error("manifest has no field " + key);
    }
    return m[1].str();
}

// One buffer visible to both devices: the XRT BO the NPU kernel takes and the pointer an iGPU kernel takes.
struct shared_buf_t {
    xrt::bo bo;
    uint16_t *dptr = nullptr;
    std::function<void()> release;
};

struct method_t {
    const char *name, *how;
    std::function<shared_buf_t(xrt::device &, size_t)> make;
};

static std::vector<method_t> methods() {
    return {
        {"A xrt_export", "XRT BO -> dma-buf fd -> hipImportExternalMemory",
         [](xrt::device &dev, size_t bytes) {
             shared_buf_t s;
             s.bo = xrt::ext::bo(dev, bytes);
             const auto fd = s.bo.export_buffer();
             hipExternalMemoryHandleDesc d;
             std::memset(&d, 0, sizeof(d));
             d.type = hipExternalMemoryHandleTypeOpaqueFd;
             d.handle.fd = (int)fd;
             d.size = bytes;
             hipExternalMemory_t em;
             HIP_OK(hipImportExternalMemory(&em, &d));
             hipExternalMemoryBufferDesc bd;
             std::memset(&bd, 0, sizeof(bd));
             bd.size = bytes;
             void *p = nullptr;
             HIP_OK(hipExternalMemoryGetMappedBuffer(&p, em, &bd));
             s.dptr = (uint16_t *)p;
             s.release = [em, p] {
                 (void)hipFree(p);
                 (void)hipDestroyExternalMemory(em);
             };
             return s;
         }},
        {"B xrt_hostreg", "XRT BO host map -> hipHostRegister",
         [](xrt::device &dev, size_t bytes) {
             shared_buf_t s;
             s.bo = xrt::ext::bo(dev, bytes);
             void *h = s.bo.map();
             HIP_OK(hipHostRegister(h, bytes, hipHostRegisterMapped));
             void *p = nullptr;
             HIP_OK(hipHostGetDevicePointer(&p, h, 0));
             s.dptr = (uint16_t *)p;
             s.release = [h] { (void)hipHostUnregister(h); };
             return s;
         }},
        {"C hip_host", "hipHostMalloc -> xrt::ext::bo(userptr)",
         [](xrt::device &dev, size_t bytes) {
             shared_buf_t s;
             void *h = nullptr;
             HIP_OK(hipHostMalloc(&h, bytes, hipHostMallocMapped));
             try {
                 s.bo = xrt::ext::bo(dev, h, bytes);
             } catch (...) {
                 (void)hipHostFree(h);
                 throw;
             }
             void *p = nullptr;
             HIP_OK(hipHostGetDevicePointer(&p, h, 0));
             s.dptr = (uint16_t *)p;
             s.release = [h] { (void)hipHostFree(h); };
             return s;
         }},
        {"D hip_export", "hipMalloc -> dma-buf fd -> xrt::bo import",
         [](xrt::device &dev, size_t bytes) {
             shared_buf_t s;
             void *p = nullptr;
             HIP_OK(hipMalloc(&p, bytes));
             int fd = -1;
             try {
                 HIP_OK(hipMemGetHandleForAddressRange(&fd, (hipDeviceptr_t)p, bytes, hipMemRangeHandleTypeDmaBufFd, 0));
                 s.bo = xrt::bo(dev, xrt::pid_type{getpid()}, (xrt::bo::export_handle)fd);
             } catch (...) {
                 (void)hipFree(p);
                 throw;
             }
             s.dptr = (uint16_t *)p;
             s.release = [p] { (void)hipFree(p); };
             return s;
         }},
    };
}

struct check_t {
    uint64_t bad = 0, sentinel = 0;
    double max_err = 0;
    std::string first;  // the first few bad elements: index x y ref out
};

static check_t check(const std::vector<uint16_t> &out, uint64_t n, uint32_t seed, float alpha16) {
    check_t c;
    for (uint64_t i = 0; i < n; ++i) {
        if (out[i] == kSentinel) {
            ++c.sentinel;
            ++c.bad;
            continue;
        }
        const float x = bf2f(f2bf(pattern(i, seed))), y = bf2f(f2bf(pattern(i, seed ^ 0x00abcdefu)));
        const float ref = bf2f(f2bf(alpha16 * x + y));
        const double err = std::fabs((double)bf2f(out[i]) - ref);
        if (!(err <= 1e-8 + 1e-2 * std::fabs(ref))) {  // the axpy Phase-1 gate (rtol 1e-2, atol 1e-8)
            if (c.bad < 4) {
                char b[160];
                snprintf(b, sizeof(b), " [i=%lu x=%g y=%g ref=%g out=%g]", (unsigned long)i, x, y, ref, bf2f(out[i]));
                c.first += b;
            }
            ++c.bad;
        }
        if (err > c.max_err || std::isnan(err)) c.max_err = std::isnan(err) ? INFINITY : err;
    }
    return c;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <axpy.json manifest> [rounds=3] [timing iters=10]\n", argv[0]);
        return 2;
    }
    const std::string manifest = argv[1];
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 3;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 10;
    std::ifstream f(manifest);
    if (!f) {
        fprintf(stderr, "cannot read %s\n", manifest.c_str());
        return 2;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string json = ss.str();
    const std::string dir = manifest.substr(0, manifest.find_last_of('/'));
    const std::string elf_path = dir + "/" + manifest_field(json, "file");
    const uint64_t n = std::stoull(manifest_field(json, "n"));
    const float alpha16 = bf2f(f2bf(std::stof(manifest_field(json, "alpha"))));  // the kernel's bf16 constant
    const size_t bytes = n * sizeof(uint16_t);

    int ndev = 0;
    HIP_OK(hipGetDeviceCount(&ndev));
    hipDeviceProp_t prop;
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    printf("iGPU: %s (%s), %d HIP device(s); NPU kernel %s, n=%lu (%.1f MiB per buffer), alpha bf16 %.6f\n", prop.name,
           prop.gcnArchName, ndev, elf_path.c_str(), (unsigned long)n, bytes / 1048576.0, alpha16);

    xrt::device dev(0);
    xrt::elf elf(elf_path);
    xrt::hw_context ctx(dev, elf);
    xrt::ext::kernel kernel(ctx, manifest_field(json, "kernel_name"));

    const dim3 block(256), grid((unsigned)((n + 255) / 256));
    uint16_t *dscratch = nullptr;  // ordinary iGPU memory, for reading results back and for the baseline
    HIP_OK(hipMalloc(&dscratch, bytes));
    std::vector<uint16_t> host(n);

    auto npu_run = [&](xrt::bo &bx, xrt::bo &by, xrt::bo &bo) {
        xrt::run run(kernel);
        run.set_arg(0, bx);
        run.set_arg(1, by);
        run.set_arg(2, bo);
        run.start();
        if (run.wait2(std::chrono::milliseconds(60000)) == std::cv_status::timeout) {
            throw std::runtime_error("NPU launch timed out");
        }
    };

    // ---- baseline: the copy path the slorado hooks + nn_npu.cpp use today
    double base_move = 0, base_kernel = 0;
    {
        uint16_t *dx = nullptr, *dy = nullptr, *dout = nullptr;
        HIP_OK(hipMalloc(&dx, bytes));
        HIP_OK(hipMalloc(&dy, bytes));
        HIP_OK(hipMalloc(&dout, bytes));
        std::vector<uint16_t> hx(n), hy(n), hout(n);
        xrt::bo bx = xrt::ext::bo(dev, bytes), by = xrt::ext::bo(dev, bytes), bo = xrt::ext::bo(dev, bytes);
        auto *mx = bx.map<uint16_t *>(), *my = by.map<uint16_t *>(), *mo = bo.map<uint16_t *>();
        bool ok = true;
        for (int it = 0; it < iters + 1; ++it) {
            const uint32_t seed = 1000 + it;
            fill<<<grid, block>>>(dx, dy, dout, n, seed);
            HIP_OK(hipDeviceSynchronize());
            auto t0 = clock_type::now();
            HIP_OK(hipMemcpy(hx.data(), dx, bytes, hipMemcpyDeviceToHost));
            HIP_OK(hipMemcpy(hy.data(), dy, bytes, hipMemcpyDeviceToHost));
            std::memcpy(mx, hx.data(), bytes);
            std::memcpy(my, hy.data(), bytes);
            bx.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            by.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            const double in = ms_since(t0);
            auto t1 = clock_type::now();
            npu_run(bx, by, bo);
            const double k = ms_since(t1);
            auto t2 = clock_type::now();
            bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            std::memcpy(hout.data(), mo, bytes);
            HIP_OK(hipMemcpy(dout, hout.data(), bytes, hipMemcpyHostToDevice));
            const double out = ms_since(t2);
            if (it > 0) {  // the first iteration warms up
                base_move += in + out;
                base_kernel += k;
            }
            HIP_OK(hipMemcpy(host.data(), dout, bytes, hipMemcpyDeviceToHost));
            const check_t c = check(host, n, seed, alpha16);
            if (c.bad) {
                ok = false;
                printf("baseline copy path iteration %d: outside tol %lu, max abs err %.3g:%s\n", it, (unsigned long)c.bad,
                       c.max_err, c.first.c_str());
            }
        }
        printf("baseline copy path: %d iterations checked, %s\n", iters + 1, ok ? "all PASS" : "FAIL");
        base_move /= iters;
        base_kernel /= iters;
        printf("baseline copy path: data movement %.2f ms, NPU kernel %.2f ms per launch (3 x %.1f MiB)\n\n", base_move,
               base_kernel, bytes / 1048576.0);
        (void)hipFree(dx);
        (void)hipFree(dy);
        (void)hipFree(dout);
    }

    // iGPU read bandwidth from ordinary device memory, for comparison with the shared buffers
    double dev_read_ms = 0;
    {
        uint16_t *src = nullptr;
        HIP_OK(hipMalloc(&src, bytes));
        copy16<<<grid, block>>>(src, dscratch, n);
        HIP_OK(hipDeviceSynchronize());
        auto t0 = clock_type::now();
        for (int it = 0; it < iters; ++it) copy16<<<grid, block>>>(src, dscratch, n);
        HIP_OK(hipDeviceSynchronize());
        dev_read_ms = ms_since(t0) / iters;
        (void)hipFree(src);
    }

    struct row_t {
        std::string name, how, result, note;
        double move = NAN, kernel = NAN, gpu_copy = NAN;
    };
    std::vector<row_t> rows;
    bool any_pass = false;

    for (auto &m : methods()) {
        row_t r{m.name, m.how, "FAIL", ""};
        std::vector<shared_buf_t> bufs;
        try {
            for (int i = 0; i < 3; ++i) bufs.push_back(m.make(dev, bytes));
            auto &x = bufs[0], &y = bufs[1], &o = bufs[2];
            // correctness: fresh data every round; the iGPU writes x, y and the sentinel, the NPU computes, the
            // iGPU reads the output back through its own pointer
            bool sync_ok = true;
            check_t worst;
            for (int round = 0; round < rounds; ++round) {
                const uint32_t seed = 77 + 31 * round;
                fill<<<grid, block>>>(x.dptr, y.dptr, o.dptr, n, seed);
                HIP_OK(hipDeviceSynchronize());
                try {
                    x.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    y.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    o.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                } catch (const std::exception &e) {
                    sync_ok = false;
                    r.note = std::string("bo.sync to device: ") + e.what() + "; ";
                }
                npu_run(x.bo, y.bo, o.bo);
                try {
                    o.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                } catch (const std::exception &e) {
                    sync_ok = false;
                    r.note += std::string("bo.sync from device: ") + e.what() + "; ";
                }
                copy16<<<grid, block>>>(o.dptr, dscratch, n);
                HIP_OK(hipDeviceSynchronize());
                HIP_OK(hipMemcpy(host.data(), dscratch, bytes, hipMemcpyDeviceToHost));
                const check_t c = check(host, n, seed, alpha16);
                if (c.bad) r.note += "round " + std::to_string(round) + ":" + c.first + "; ";
                if (c.bad > worst.bad) worst = c;
                if (c.max_err > worst.max_err) worst.max_err = c.max_err;
            }
            bool pass = worst.bad == 0;
            r.result = pass ? "PASS" : "FAIL";
            char buf[256];
            snprintf(buf, sizeof(buf), "%d rounds: worst outside tol %lu/%lu, NPU result not seen %lu, max abs err %.3g%s",
                     rounds, (unsigned long)worst.bad, (unsigned long)n, (unsigned long)worst.sentinel, worst.max_err,
                     sync_ok ? "" : "; bo.sync unsupported");
            r.note += buf;
            if (pass) {
                // timing: everything between "iGPU finished writing" and "iGPU can read the result" except the kernel
                double move = 0, kern = 0, gcopy = 0;
                for (int it = 0; it < iters; ++it) {
                    fill<<<grid, block>>>(x.dptr, y.dptr, o.dptr, n, 5000 + it);
                    HIP_OK(hipDeviceSynchronize());
                    auto t0 = clock_type::now();
                    if (sync_ok) {
                        x.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                        y.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    }
                    const double in = ms_since(t0);
                    auto t1 = clock_type::now();
                    npu_run(x.bo, y.bo, o.bo);
                    kern += ms_since(t1);
                    auto t2 = clock_type::now();
                    if (sync_ok) o.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    move += in + ms_since(t2);
                    auto t3 = clock_type::now();
                    copy16<<<grid, block>>>(o.dptr, dscratch, n);
                    HIP_OK(hipDeviceSynchronize());
                    gcopy += ms_since(t3);
                    HIP_OK(hipMemcpy(host.data(), dscratch, bytes, hipMemcpyDeviceToHost));
                    const check_t c = check(host, n, 5000 + it, alpha16);
                    if (c.bad) {
                        r.result = "FAIL";
                        char b[96];
                        snprintf(b, sizeof(b), "; timing iteration %d: outside tol %lu", it, (unsigned long)c.bad);
                        r.note += b + c.first;
                    }
                }
                r.move = move / iters;
                r.kernel = kern / iters;
                r.gpu_copy = gcopy / iters;
                pass = r.result == "PASS";
            }
            any_pass |= pass;
        } catch (const std::exception &e) {
            r.result = "FAIL";
            r.note = std::string("setup/run error: ") + e.what();
        }
        for (auto &b : bufs) {
            if (b.release) b.release();
        }
        rows.push_back(r);
    }

    printf("| method | how | result | data movement ms (baseline %.2f) | NPU kernel ms (baseline %.2f) | iGPU read of output ms (device memory %.2f) | detail |\n",
           base_move, base_kernel, dev_read_ms);
    printf("|---|---|---|---|---|---|---|\n");
    for (auto &r : rows) {
        printf("| %s | %s | %s | %.2f | %.2f | %.2f | %s |\n", r.name.c_str(), r.how.c_str(), r.result.c_str(), r.move,
               r.kernel, r.gpu_copy, r.note.c_str());
    }
    (void)hipFree(dscratch);
    printf("RESULT: %s\n", any_pass ? "PASS (at least one zero-copy method works)" : "FAIL (no zero-copy method works)");
    return any_pass ? 0 : 1;
}
