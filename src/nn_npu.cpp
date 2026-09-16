// openfish NPU backend: AMD XDNA2 (NPU2) kernels compiled with mlir-air, invoked
// through the XRT full-ELF protocol (xrt::elf -> hw_context -> ext::kernel/ext::bo,
// see mlir-air python/air/backend/xrt.py:load). Host tensors are float32; the
// kernels take bf16, so values are converted on the way in and out.
//
// Environment:
//   OPENFISH_NPU_ARTIFACTS  directory with <kernel>.json manifests + artifacts (required)
//   OPENFISH_NPU_THREADS    host threads for layout/bf16 conversion (default 8)
//   NPU_LOCK                lock file taken with flock(LOCK_EX) for the process lifetime
//                           (unless OPENFISH_NPU_NO_FLOCK=1); do not also wrap the
//                           process in `flock $NPU_LOCK`, that would deadlock.

#include "nn_npu.h"

#include <openfish/openfish_error.h>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

double seconds_since(clock_type::time_point t0) {
    return std::chrono::duration<double>(clock_type::now() - t0).count();
}

// IEEE float32 -> bfloat16 (1 sign, 8 exponent, 7 mantissa bits): round the
// mantissa to nearest-even, carry into the exponent, overflow to inf. Bit-exact
// with ml_dtypes.bfloat16 (verified on 10M values incl. denormals, inf, nan).
inline uint16_t f32_to_bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    uint32_t sign = b >> 31;
    uint32_t ex = (b >> 23) & 0xff;
    uint32_t man = b & 0x7fffff;
    uint32_t q = man >> 16;
    if (ex == 0xff) {
        q = man ? std::max<uint32_t>(q, 1) : 0;  // nan keeps a nonzero mantissa
    } else {
        uint32_t rem = man & 0xffff;
        if (rem > 0x8000 || (rem == 0x8000 && (q & 1))) {
            ++q;
        }
        if (q == 0x80) {
            q = 0;
            ++ex;
        }
        if (ex >= 0xff) {
            ex = 0xff;
            q = 0;
        }
    }
    return (uint16_t)((sign << 15) | (ex << 7) | q);
}

inline float bf16_to_f32(uint16_t h) {
    uint32_t b = ((uint32_t)(h >> 15) << 31) | ((uint32_t)((h >> 7) & 0xff) << 23) | ((uint32_t)(h & 0x7f) << 16);
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

void parallel_for(uint64_t n, int n_threads, const std::function<void(uint64_t, uint64_t)> &fn) {
    if (n_threads <= 1 || n < (1u << 16)) {
        fn(0, n);
        return;
    }
    std::vector<std::thread> threads;
    uint64_t step = (n + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; ++t) {
        uint64_t a = t * step;
        uint64_t b = std::min(n, a + step);
        if (a >= b) {
            break;
        }
        threads.emplace_back(fn, a, b);
    }
    for (auto &th : threads) {
        th.join();
    }
}

std::string manifest_field(const std::string &json, const std::string &key) {
    std::smatch m;
    std::regex re("\"" + key + "\"\\s*:\\s*\"?([^\",}]*)\"?");
    if (!std::regex_search(json, m, re)) {
        throw std::runtime_error("manifest missing field '" + key + "'");
    }
    return m[1];
}

struct npu_stats_t {
    uint64_t calls = 0;
    uint64_t launches = 0;
    uint64_t elements = 0;
    double t_in = 0;      // layout split + f32->bf16 + sync to device
    double t_kernel = 0;  // run.start() .. wait2()
    double t_out = 0;     // sync from device + bf16->f32
};

// One compiled 3-tensor elementwise kernel (in0, in1, out), resident for the process lifetime.
struct elementwise_kernel_t {
    std::string name;
    std::string path;
    uint64_t n = 0;
    xrt::device device;
    xrt::elf elf;
    xrt::hw_context ctx;
    xrt::ext::kernel kernel;
    xrt::ext::bo bo_in0, bo_in1, bo_out;
    uint16_t *in0 = nullptr, *in1 = nullptr, *out = nullptr;
    xrt::run run;
    npu_stats_t stats;

    elementwise_kernel_t(const std::string &kernel_id, const std::string &dir, const std::string &json)
        : name(kernel_id),
          path(dir + "/" + manifest_field(json, "file")),
          n(std::stoull(manifest_field(json, "n"))),
          device(0),
          elf(path),
          ctx(device, elf),
          kernel(ctx, manifest_field(json, "kernel_name")),
          bo_in0(device, n * sizeof(uint16_t)),
          bo_in1(device, n * sizeof(uint16_t)),
          bo_out(device, n * sizeof(uint16_t)),
          in0(bo_in0.map<uint16_t *>()),
          in1(bo_in1.map<uint16_t *>()),
          out(bo_out.map<uint16_t *>()),
          run(kernel) {
        run.set_arg(0, bo_in0);
        run.set_arg(1, bo_in1);
        run.set_arg(2, bo_out);
    }
};

std::mutex g_mutex;
elementwise_kernel_t *g_silu_mul = nullptr;  // intentionally never freed: avoids XRT static-destruction order at exit
int g_lock_fd = -1;
int g_threads = 8;

void print_stats() {
    const elementwise_kernel_t *k = g_silu_mul;
    if (!k || !k->stats.calls) {
        return;
    }
    const npu_stats_t &s = k->stats;
    fprintf(stderr,
            "[nn_npu] %s: %lu calls, %lu launches, %lu elements; host in %.3f s, kernel %.3f s (%.1f ms/launch), host out %.3f s\n",
            k->name.c_str(), (unsigned long)s.calls, (unsigned long)s.launches, (unsigned long)s.elements, s.t_in,
            s.t_kernel, 1e3 * s.t_kernel / std::max<uint64_t>(s.launches, 1), s.t_out);
}

void take_process_lock() {
    const char *lock = std::getenv("NPU_LOCK");
    const char *skip = std::getenv("OPENFISH_NPU_NO_FLOCK");
    if (!lock || (skip && std::strcmp(skip, "1") == 0)) {
        fprintf(stderr, "[nn_npu] not taking an NPU lock (NPU_LOCK unset or OPENFISH_NPU_NO_FLOCK=1)\n");
        return;
    }
    g_lock_fd = open(lock, O_RDWR | O_CREAT, 0644);
    if (g_lock_fd < 0) {
        throw std::runtime_error(std::string("cannot open NPU_LOCK ") + lock + ": " + strerror(errno));
    }
    fprintf(stderr, "[nn_npu] waiting for lock %s\n", lock);
    if (flock(g_lock_fd, LOCK_EX) != 0) {
        throw std::runtime_error(std::string("flock failed on ") + lock + ": " + strerror(errno));
    }
    fprintf(stderr, "[nn_npu] holding lock %s for the process lifetime\n", lock);
}

elementwise_kernel_t *load_silu_mul() {
    const char *dir = std::getenv("OPENFISH_NPU_ARTIFACTS");
    if (!dir) {
        throw std::runtime_error("OPENFISH_NPU_ARTIFACTS is not set");
    }
    if (const char *t = std::getenv("OPENFISH_NPU_THREADS")) {
        g_threads = std::max(1, std::atoi(t));
    }
    const std::string manifest = std::string(dir) + "/silu_mul.json";
    std::ifstream in(manifest);
    if (!in) {
        throw std::runtime_error("cannot read " + manifest);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string json = ss.str();
    if (manifest_field(json, "format") != "elf" || manifest_field(json, "dtype") != "bf16") {
        throw std::runtime_error(manifest + ": expected format elf, dtype bf16");
    }
    take_process_lock();
    auto t0 = clock_type::now();
    auto *k = new elementwise_kernel_t("silu_mul", dir, json);
    struct stat st;
    stat(k->path.c_str(), &st);
    fprintf(stderr,
            "[nn_npu] loaded silu_mul: %s (%ld bytes, sha256 %s), kernel %s, N=%lu, tile_n %s, herd %s, %d host threads, %.3f s\n",
            k->path.c_str(), (long)st.st_size, manifest_field(json, "sha256").c_str(),
            manifest_field(json, "kernel_name").c_str(), (unsigned long)k->n, manifest_field(json, "tile_n").c_str(),
            manifest_field(json, "herd").c_str(), g_threads, seconds_since(t0));
    std::atexit(print_stats);
    return k;
}

}  // namespace

void silu_mul_npu(const float *x, float *o, uint64_t MN, uint64_t K) {
    std::lock_guard<std::mutex> guard(g_mutex);
    try {
        if (!g_silu_mul) {
            g_silu_mul = load_silu_mul();
        }
        elementwise_kernel_t &k = *g_silu_mul;
        const uint64_t total = MN * K;
        const uint64_t n = k.n;
        const uint64_t launches = (total + n - 1) / n;
        static bool warned_padding = false;
        if (total % n && !warned_padding) {
            fprintf(stderr, "[nn_npu] silu_mul: %lu elements (MN=%lu, K=%lu) is not a multiple of N=%lu; zero-padding the last launch\n",
                    (unsigned long)total, (unsigned long)MN, (unsigned long)K, (unsigned long)n);
            warned_padding = true;
        }

        for (uint64_t l = 0; l < launches; ++l) {
            const uint64_t begin = l * n;
            const uint64_t count = std::min(n, total - begin);

            auto t0 = clock_type::now();
            // openfish row layout [y ‖ gate] -> kernel args (gate, up)
            parallel_for(count, g_threads, [&](uint64_t a, uint64_t b) {
                for (uint64_t i = a; i < b; ++i) {
                    const uint64_t e = begin + i;
                    const float *row = x + (e / K) * 2 * K;
                    const uint64_t col = e % K;
                    k.in0[i] = f32_to_bf16(row[K + col]);
                    k.in1[i] = f32_to_bf16(row[col]);
                }
            });
            if (count < n) {
                std::memset(k.in0 + count, 0, (n - count) * sizeof(uint16_t));
                std::memset(k.in1 + count, 0, (n - count) * sizeof(uint16_t));
            }
            k.bo_in0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            k.bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto t1 = clock_type::now();

            k.run.start();
            if (k.run.wait2(std::chrono::milliseconds(60000)) == std::cv_status::timeout) {
                throw std::runtime_error("silu_mul launch timed out after 60 s");
            }
            auto t2 = clock_type::now();

            k.bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            parallel_for(count, g_threads, [&](uint64_t a, uint64_t b) {
                for (uint64_t i = a; i < b; ++i) {
                    o[begin + i] = bf16_to_f32(k.out[i]);
                }
            });
            auto t3 = clock_type::now();

            k.stats.t_in += std::chrono::duration<double>(t1 - t0).count();
            k.stats.t_kernel += std::chrono::duration<double>(t2 - t1).count();
            k.stats.t_out += std::chrono::duration<double>(t3 - t2).count();
        }
        k.stats.calls += 1;
        k.stats.launches += launches;
        k.stats.elements += total;
    } catch (const std::exception &e) {
        OPENFISH_ERROR("silu_mul_npu: %s", e.what());
        exit(EXIT_FAILURE);
    }
}
