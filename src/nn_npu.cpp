// openfish NPU backend: AMD XDNA2 (NPU2) kernels compiled with mlir-air, invoked
// through the XRT full-ELF protocol (xrt::elf -> hw_context -> ext::kernel/ext::bo,
// see mlir-air python/air/backend/xrt.py:load). Host tensors are float32; the
// kernels take bf16, so values are converted on the way in and out.
//
// Environment:
//   OPENFISH_NPU_OPS        comma list of ops to run on the NPU (default: none);
//                           known: silu_mul, fc1, fc2
//   OPENFISH_NPU_ARTIFACTS  directory with silu_mul.json + its ELF
//   OPENFISH_NPU_GEMM_ARTIFACTS  directory with gemm_k<K>_n<N>.json + ELFs (fc1/fc2)
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
#include <unordered_map>
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
    // Held as xrt::bo: passing an xrt::ext::bo to run.set_arg() picks the scalar
    // template overload and fails with "patch_value() only supports 64-bit values".
    xrt::bo bo_in0, bo_in1, bo_out;
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
          bo_in0(xrt::ext::bo(device, n * sizeof(uint16_t))),
          bo_in1(xrt::ext::bo(device, n * sizeof(uint16_t))),
          bo_out(xrt::ext::bo(device, n * sizeof(uint16_t))),
          in0(bo_in0.map<uint16_t *>()),
          in1(bo_in1.map<uint16_t *>()),
          out(bo_out.map<uint16_t *>()),
          run(kernel) {
        run.set_arg(0, bo_in0);
        run.set_arg(1, bo_in1);
        run.set_arg(2, bo_out);
    }
};

// GEMM C[m,n] = A[m,k] @ B[k,n] from the registry fused-cast ELF (args A_bf16, B_bf16, C_f32, C_bf16).
// The f32 scratch C is read back as the result (full-precision accumulator, no bf16 round trip).
// B = weight.T is converted to bf16 once per weight tensor and kept resident.
struct gemm_kernel_t {
    std::string name;
    std::string path;
    uint64_t m = 0, k = 0, n = 0;
    xrt::device device;
    xrt::elf elf;
    xrt::hw_context ctx;
    xrt::ext::kernel kernel;
    xrt::bo bo_a, bo_c32, bo_c16;
    uint16_t *a = nullptr;
    float *c32 = nullptr;
    xrt::run run;
    std::unordered_map<const float *, xrt::bo> weights;
    npu_stats_t stats;

    gemm_kernel_t(const std::string &op, const std::string &dir, const std::string &json)
        : name(op),
          path(dir + "/" + manifest_field(json, "file")),
          m(std::stoull(manifest_field(json, "m"))),
          k(std::stoull(manifest_field(json, "k"))),
          n(std::stoull(manifest_field(json, "n"))),
          device(0),
          elf(path),
          ctx(device, elf),
          kernel(ctx, manifest_field(json, "kernel_name")),
          bo_a(xrt::ext::bo(device, m * k * sizeof(uint16_t))),
          bo_c32(xrt::ext::bo(device, m * n * sizeof(float))),
          bo_c16(xrt::ext::bo(device, m * n * sizeof(uint16_t))),
          a(bo_a.map<uint16_t *>()),
          c32(bo_c32.map<float *>()),
          run(kernel) {
        run.set_arg(0, bo_a);
        run.set_arg(2, bo_c32);
        run.set_arg(3, bo_c16);
    }
};

std::mutex g_mutex;
// Kernels are intentionally never freed: avoids XRT static-destruction order at exit.
elementwise_kernel_t *g_silu_mul = nullptr;
std::unordered_map<std::string, gemm_kernel_t *> g_gemm;
std::vector<std::pair<std::string, const npu_stats_t *>> g_stats;
int g_lock_fd = -1;
int g_threads = 8;
bool g_common_ready = false;

void print_stats() {
    for (const auto &it : g_stats) {
        const npu_stats_t &s = *it.second;
        if (!s.calls) {
            continue;
        }
        fprintf(stderr,
                "[nn_npu] %s: %lu calls, %lu launches, %lu elements; host in %.3f s, kernel %.3f s (%.1f ms/launch), host out %.3f s\n",
                it.first.c_str(), (unsigned long)s.calls, (unsigned long)s.launches, (unsigned long)s.elements, s.t_in,
                s.t_kernel, 1e3 * s.t_kernel / std::max<uint64_t>(s.launches, 1), s.t_out);
    }
}

std::string read_file(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot read " + path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
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

void init_common() {
    if (g_common_ready) {
        return;
    }
    if (const char *t = std::getenv("OPENFISH_NPU_THREADS")) {
        g_threads = std::max(1, std::atoi(t));
    }
    take_process_lock();
    std::atexit(print_stats);
    g_common_ready = true;
}

elementwise_kernel_t *load_silu_mul() {
    const char *dir = std::getenv("OPENFISH_NPU_ARTIFACTS");
    if (!dir) {
        throw std::runtime_error("OPENFISH_NPU_ARTIFACTS is not set");
    }
    const std::string manifest = std::string(dir) + "/silu_mul.json";
    const std::string json = read_file(manifest);
    if (manifest_field(json, "format") != "elf" || manifest_field(json, "dtype") != "bf16") {
        throw std::runtime_error(manifest + ": expected format elf, dtype bf16");
    }
    init_common();
    auto t0 = clock_type::now();
    auto *k = new elementwise_kernel_t("silu_mul", dir, json);
    struct stat st;
    stat(k->path.c_str(), &st);
    fprintf(stderr,
            "[nn_npu] loaded silu_mul: %s (%ld bytes, sha256 %s), kernel %s, N=%lu, tile_n %s, herd %s, %d host threads, %.3f s\n",
            k->path.c_str(), (long)st.st_size, manifest_field(json, "sha256").c_str(),
            manifest_field(json, "kernel_name").c_str(), (unsigned long)k->n, manifest_field(json, "tile_n").c_str(),
            manifest_field(json, "herd").c_str(), g_threads, seconds_since(t0));
    g_stats.emplace_back(k->name, &k->stats);
    return k;
}

gemm_kernel_t *load_gemm(const std::string &op, uint64_t K, uint64_t N) {
    const char *dir = std::getenv("OPENFISH_NPU_GEMM_ARTIFACTS");
    if (!dir) {
        throw std::runtime_error("OPENFISH_NPU_GEMM_ARTIFACTS is not set");
    }
    const std::string manifest = std::string(dir) + "/gemm_k" + std::to_string(K) + "_n" + std::to_string(N) + ".json";
    const std::string json = read_file(manifest);
    if (manifest_field(json, "format") != "elf" || manifest_field(json, "method") != "fused-cast") {
        throw std::runtime_error(manifest + ": expected format elf, method fused-cast");
    }
    init_common();
    auto t0 = clock_type::now();
    auto *g = new gemm_kernel_t(op, dir, json);
    if (g->k != K || g->n != N) {
        throw std::runtime_error(manifest + ": shape mismatch");
    }
    fprintf(stderr,
            "[nn_npu] loaded %s: %s (sha256 %s), kernel %s, M=%lu K=%lu N=%lu, mmul %s, tiles m/kl2/kl1/n %s/%s/%s/%s herd %s, %.3f s\n",
            op.c_str(), g->path.c_str(), manifest_field(json, "sha256").c_str(), manifest_field(json, "kernel_name").c_str(),
            (unsigned long)g->m, (unsigned long)K, (unsigned long)N, manifest_field(json, "mmul").c_str(),
            manifest_field(json, "tile_m").c_str(), manifest_field(json, "tile_k_l2").c_str(),
            manifest_field(json, "tile_k_l1").c_str(), manifest_field(json, "tile_n").c_str(),
            manifest_field(json, "herd").c_str(), seconds_since(t0));
    g_stats.emplace_back(op, &g->stats);
    return g;
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

int npu_op_enabled(const char *op) {
    static const std::vector<std::string> ops = [] {
        std::vector<std::string> v;
        const char *env = std::getenv("OPENFISH_NPU_OPS");
        std::stringstream ss(env ? env : "");
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) {
                v.push_back(item);
            }
        }
        std::string joined;
        for (const auto &o : v) {
            joined += (joined.empty() ? "" : ",") + o;
        }
        fprintf(stderr, "[nn_npu] ops on NPU: %s\n", joined.empty() ? "(none)" : joined.c_str());
        return v;
    }();
    return std::find(ops.begin(), ops.end(), op) != ops.end();
}

void linear_npu(const char *op, const float *x, float *out, const float *weight, uint64_t rows, uint64_t K, uint64_t N) {
    std::lock_guard<std::mutex> guard(g_mutex);
    try {
        auto it = g_gemm.find(op);
        if (it == g_gemm.end()) {
            it = g_gemm.emplace(op, load_gemm(op, K, N)).first;
        }
        gemm_kernel_t &g = *it->second;
        if (g.k != K || g.n != N) {
            throw std::runtime_error(std::string(op) + ": runtime shape " + std::to_string(K) + "x" + std::to_string(N) +
                                     " != compiled " + std::to_string(g.k) + "x" + std::to_string(g.n));
        }

        auto t0 = clock_type::now();
        auto w = g.weights.find(weight);
        if (w == g.weights.end()) {
            // B[k, n] = weight[n, k].T, bf16, resident for the process lifetime
            xrt::bo bo_b = xrt::ext::bo(g.device, K * N * sizeof(uint16_t));
            uint16_t *b = bo_b.map<uint16_t *>();
            parallel_for(K, g_threads, [&](uint64_t lo, uint64_t hi) {
                for (uint64_t i = lo; i < hi; ++i) {
                    for (uint64_t j = 0; j < N; ++j) {
                        b[i * N + j] = f32_to_bf16(weight[j * K + i]);
                    }
                }
            });
            bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            w = g.weights.emplace(weight, bo_b).first;
        }
        g.run.set_arg(1, w->second);

        const uint64_t m = g.m;
        const uint64_t launches = (rows + m - 1) / m;
        g.stats.t_in += seconds_since(t0);
        for (uint64_t l = 0; l < launches; ++l) {
            const uint64_t begin = l * m;
            const uint64_t count = std::min(m, rows - begin);

            auto t1 = clock_type::now();
            parallel_for(count * K, g_threads, [&](uint64_t lo, uint64_t hi) {
                for (uint64_t i = lo; i < hi; ++i) {
                    g.a[i] = f32_to_bf16(x[begin * K + i]);
                }
            });
            if (count < m) {
                std::memset(g.a + count * K, 0, (m - count) * K * sizeof(uint16_t));
            }
            g.bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto t2 = clock_type::now();

            g.run.start();
            if (g.run.wait2(std::chrono::milliseconds(60000)) == std::cv_status::timeout) {
                throw std::runtime_error(std::string(op) + " launch timed out after 60 s");
            }
            auto t3 = clock_type::now();

            g.bo_c32.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            // parallel copy: the destination is a fresh torch tensor, so first-touch page faults dominate
            parallel_for(count * N, g_threads, [&](uint64_t lo, uint64_t hi) {
                std::memcpy(out + begin * N + lo, g.c32 + lo, (hi - lo) * sizeof(float));
            });
            auto t4 = clock_type::now();

            g.stats.t_in += std::chrono::duration<double>(t2 - t1).count();
            g.stats.t_kernel += std::chrono::duration<double>(t3 - t2).count();
            g.stats.t_out += std::chrono::duration<double>(t4 - t3).count();
        }
        g.stats.calls += 1;
        g.stats.launches += launches;
        g.stats.elements += rows * N;
    } catch (const std::exception &e) {
        OPENFISH_ERROR("linear_npu(%s): %s", op, e.what());
        exit(EXIT_FAILURE);
    }
}
