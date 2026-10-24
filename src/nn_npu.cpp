// openfish NPU backend: AMD XDNA2 (NPU2) kernels compiled with mlir-air, invoked
// through the XRT full-ELF protocol (xrt::elf -> hw_context -> ext::kernel/ext::bo,
// see mlir-air python/air/backend/xrt.py:load). Host tensors are float32; the
// kernels take bf16, so values are converted on the way in and out.
//
// Environment:
//   OPENFISH_NPU_OPS        comma list of ops to run on the NPU (default: none);
//                           known: silu_mul, fc1, fc2, wqkv, out_proj, upsample, crf, rmsnorm, rope, conv,
//                           residual (axpy), swish (silu), rms_qkv_rope (fused norm2 + next layer's wqkv + RoPE)
//                           (GEMM ops share one loaded ELF per K x N shape, e.g. fc1 and crf)
//   OPENFISH_NPU_ARTIFACTS  directory with silu_mul.json + its ELF
//   OPENFISH_NPU_GEMM_ARTIFACTS  directory with gemm_k<K>_n<N>.json + ELFs (fc1/fc2)
//   OPENFISH_NPU_THREADS    host threads for layout/bf16 conversion (default 8)
//   NPU_LOCK                lock file taken with flock(LOCK_EX) for the process lifetime
//                           (unless OPENFISH_NPU_NO_FLOCK=1); do not also wrap the
//                           process in `flock $NPU_LOCK`, that would deadlock.

#include "nn_npu.h"
#include "npu_bfp16_pack.h"

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
#include <cmath>
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
#include <type_traits>
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

inline uint16_t to_bf16(float v) { return f32_to_bf16(v); }
inline uint16_t to_bf16(uint16_t v) { return v; }

inline float bf16_to_f32(uint16_t h);
inline void store_bf16(float &d, uint16_t h) { d = bf16_to_f32(h); }
inline void store_bf16(uint16_t &d, uint16_t h) { d = h; }

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
    uint64_t guard_retried = 0, guard_polls = 0, guard_timeouts = 0;  // stale-output guard (below)
    double guard_wait = 0;
};

// Stale-output guard (development_progress/debug_log.md 2026-10-24): after run.wait() + bo.sync(FROM_DEVICE), the
// last few hundred elements of an output BO can still read as the previous launch's values for up to ~1 ms (rare
// with XRT alone, frequent with HIP active in the process). Before a launch, guard_arm() fills the output's last
// kGuardBytes with 0xFF (NaN in bf16 and f32) and syncs that range; after the launch and sync, guard_wait() re-syncs
// that range until no all-0xFF element is left. The CPU's last write there is the marker, so any other value read is
// the NPU's new data. OPENFISH_NPU_GUARD=0 turns it off.
constexpr uint64_t kGuardBytes = 4096;
constexpr double kGuardTimeout = 0.05;  // s; a kernel that really outputs all-0xFF NaNs there would otherwise spin

bool guard_on() {
    static const bool on = [] {
        const char *e = std::getenv("OPENFISH_NPU_GUARD");
        const bool v = !(e && std::string(e) == "0");
        fprintf(stderr, "[nn_npu] stale-output guard %s\n", v ? "on" : "off (OPENFISH_NPU_GUARD=0)");
        return v;
    }();
    return on;
}

void guard_arm(xrt::bo &bo, void *map, uint64_t bytes) {
    if (!guard_on()) return;
    const uint64_t g = std::min(bytes, kGuardBytes), off = bytes - g;
    std::memset(static_cast<uint8_t *>(map) + off, 0xFF, g);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, g, off);
}

bool guard_marked(const void *p, uint64_t g, uint64_t elem_bytes) {
    if (elem_bytes == 4) {
        const uint32_t *q = static_cast<const uint32_t *>(p);
        for (uint64_t i = 0; i < g / 4; ++i) {
            if (q[i] == 0xFFFFFFFFu) return true;
        }
        return false;
    }
    const uint16_t *q = static_cast<const uint16_t *>(p);
    for (uint64_t i = 0; i < g / 2; ++i) {
        if (q[i] == 0xFFFFu) return true;
    }
    return false;
}

// Call after the output BO's bo.sync(FROM_DEVICE), before reading it.
void guard_wait(xrt::bo &bo, const void *map, uint64_t bytes, uint64_t elem_bytes, npu_stats_t &stats) {
    if (!guard_on()) return;
    const uint64_t g = std::min(bytes, kGuardBytes), off = bytes - g;
    const uint8_t *tail = static_cast<const uint8_t *>(map) + off;
    if (!guard_marked(tail, g, elem_bytes)) return;
    auto t0 = clock_type::now();
    stats.guard_retried += 1;
    while (guard_marked(tail, g, elem_bytes)) {
        if (seconds_since(t0) > kGuardTimeout) {
            stats.guard_timeouts += 1;
            break;
        }
        bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, g, off);
        stats.guard_polls += 1;
    }
    stats.guard_wait += seconds_since(t0);
}

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

// GEMM C[m,n] = A[m,k] @ B[k,n], one of two ELF kinds (manifest "method"):
//   fused-cast: registry bf16_in_bf16_out GEMM + cast launch, args (A_bf16, B_bf16, C_f32, C_bf16)
//   f32-out:    bf16_in_fp32_out GEMM, one launch, args (A_bf16, B_bf16, C_f32); no bf16 output exists
//   bfp16w:     bf16_x_bfp16 GEMM, one launch, args (A_bf16, B_bfp16ebs8 packed, C_bf16); no f32 output exists
// The f32 C is read back for an f32 result (full-precision accumulator, no bf16 round trip).
// B = weight.T is converted (bf16, or bf16 then bfp16ebs8-packed) once per weight tensor and kept resident.
struct gemm_kernel_t {
    std::string name;
    std::string path;
    std::string method;
    uint64_t m = 0, k = 0, n = 0;
    uint64_t tile_n = 0, tile_k_l1 = 0;  // bfp16w weight packing
    xrt::device device;
    xrt::elf elf;
    xrt::hw_context ctx;
    xrt::ext::kernel kernel;
    xrt::bo bo_a, bo_c32, bo_c16;
    uint16_t *a = nullptr;
    float *c32 = nullptr;
    uint16_t *c16 = nullptr;
    xrt::run run;
    std::unordered_map<const float *, xrt::bo> weights;
    npu_stats_t stats;

    gemm_kernel_t(const std::string &op, const std::string &dir, const std::string &json)
        : name(op),
          path(dir + "/" + manifest_field(json, "file")),
          method(manifest_field(json, "method")),
          m(std::stoull(manifest_field(json, "m"))),
          k(std::stoull(manifest_field(json, "k"))),
          n(std::stoull(manifest_field(json, "n"))),
          device(0),
          elf(path),
          ctx(device, elf),
          kernel(ctx, manifest_field(json, "kernel_name")),
          bo_a(xrt::ext::bo(device, m * k * sizeof(uint16_t))),
          a(bo_a.map<uint16_t *>()),
          run(kernel) {
        run.set_arg(0, bo_a);
        if (method == "bfp16w") {
            tile_n = std::stoull(manifest_field(json, "tile_n"));
            tile_k_l1 = std::stoull(manifest_field(json, "tile_k_l1"));
            bo_c16 = xrt::ext::bo(device, m * n * sizeof(uint16_t));
            c16 = bo_c16.map<uint16_t *>();
            run.set_arg(2, bo_c16);
            return;
        }
        bo_c32 = xrt::ext::bo(device, m * n * sizeof(float));
        c32 = bo_c32.map<float *>();
        run.set_arg(2, bo_c32);
        if (method == "fused-cast") {
            bo_c16 = xrt::ext::bo(device, m * n * sizeof(uint16_t));
            c16 = bo_c16.map<uint16_t *>();
            run.set_arg(3, bo_c16);
        }
    }
};

std::mutex g_mutex;
// Kernels are intentionally never freed: avoids XRT static-destruction order at exit.
elementwise_kernel_t *g_silu_mul = nullptr;
std::unordered_map<std::string, gemm_kernel_t *> g_gemm;  // keyed by "<K>x<N>"
std::unordered_map<std::string, npu_stats_t> g_op_stats;   // per op name (node addresses are stable)
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
                "[nn_npu] %s: %lu calls, %lu launches, %lu elements; host in %.3f s, kernel %.3f s (%.1f ms/launch), host out %.3f s; "
                "stale-output guard: %lu launches waited (%lu re-syncs, %.4f s, %lu timeouts)\n",
                it.first.c_str(), (unsigned long)s.calls, (unsigned long)s.launches, (unsigned long)s.elements, s.t_in,
                s.t_kernel, 1e3 * s.t_kernel / std::max<uint64_t>(s.launches, 1), s.t_out, (unsigned long)s.guard_retried,
                (unsigned long)s.guard_polls, s.guard_wait, (unsigned long)s.guard_timeouts);
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
    const std::string method = manifest_field(json, "method");
    if (manifest_field(json, "format") != "elf" || (method != "fused-cast" && method != "f32-out" && method != "bfp16w")) {
        throw std::runtime_error(manifest + ": expected format elf, method fused-cast, f32-out or bfp16w");
    }
    init_common();
    auto t0 = clock_type::now();
    auto *g = new gemm_kernel_t(op, dir, json);
    if (g->k != K || g->n != N) {
        throw std::runtime_error(manifest + ": shape mismatch");
    }
    fprintf(stderr,
            "[nn_npu] loaded gemm %lux%lu for %s: %s (%s, sha256 %s), kernel %s, M=%lu, mmul %s, tiles m/kl2/kl1/n %s/%s/%s/%s herd %s, %.3f s\n",
            (unsigned long)K, (unsigned long)N, op.c_str(), g->path.c_str(), method.c_str(), manifest_field(json, "sha256").c_str(),
            manifest_field(json, "kernel_name").c_str(), (unsigned long)g->m, manifest_field(json, "mmul").c_str(),
            manifest_field(json, "tile_m").c_str(), manifest_field(json, "tile_k_l2").c_str(),
            manifest_field(json, "tile_k_l1").c_str(), manifest_field(json, "tile_n").c_str(),
            manifest_field(json, "herd").c_str(), seconds_since(t0));
    return g;
}

}  // namespace

template <typename In, typename Out>
static void silu_mul_impl(const In *x, Out *o, uint64_t MN, uint64_t K) {
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
                    const In *row = x + (e / K) * 2 * K;
                    const uint64_t col = e % K;
                    k.in0[i] = to_bf16(row[K + col]);
                    k.in1[i] = to_bf16(row[col]);
                }
            });
            if (count < n) {
                std::memset(k.in0 + count, 0, (n - count) * sizeof(uint16_t));
                std::memset(k.in1 + count, 0, (n - count) * sizeof(uint16_t));
            }
            k.bo_in0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            k.bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            guard_arm(k.bo_out, k.out, n * sizeof(uint16_t));
            auto t1 = clock_type::now();

            k.run.start();
            if (k.run.wait2(std::chrono::milliseconds(60000)) == std::cv_status::timeout) {
                throw std::runtime_error("silu_mul launch timed out after 60 s");
            }
            auto t2 = clock_type::now();

            k.bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            guard_wait(k.bo_out, k.out, n * sizeof(uint16_t), 2, k.stats);
            parallel_for(count, g_threads, [&](uint64_t a, uint64_t b) {
                for (uint64_t i = a; i < b; ++i) {
                    store_bf16(o[begin + i], k.out[i]);
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

template <typename In, typename Out>
static void linear_impl(const char *op, const In *x, Out *out, const float *weight, uint64_t rows, uint64_t K, uint64_t N) {
    std::lock_guard<std::mutex> guard(g_mutex);
    try {
        const std::string key = std::to_string(K) + "x" + std::to_string(N);
        auto it = g_gemm.find(key);
        if (it == g_gemm.end()) {
            it = g_gemm.emplace(key, load_gemm(op, K, N)).first;
        }
        gemm_kernel_t &g = *it->second;
        auto st = g_op_stats.find(op);
        if (st == g_op_stats.end()) {
            st = g_op_stats.emplace(op, npu_stats_t()).first;
            g_stats.emplace_back(op, &st->second);
        }
        npu_stats_t &stats = st->second;
        if (g.k != K || g.n != N) {
            throw std::runtime_error(std::string(op) + ": runtime shape " + std::to_string(K) + "x" + std::to_string(N) +
                                     " != compiled " + std::to_string(g.k) + "x" + std::to_string(g.n));
        }

        auto t0 = clock_type::now();
        auto w = g.weights.find(weight);
        if (w == g.weights.end() && g.method == "bfp16w") {
            // B[k, n] = weight[n, k].T rounded to bf16, then bfp16ebs8-packed (example pack_b_bfp16ebs8), resident
            std::vector<float> wr(K * N);
            parallel_for(K * N, g_threads, [&](uint64_t lo, uint64_t hi) {
                for (uint64_t i = lo; i < hi; ++i) wr[i] = bf16_to_f32(f32_to_bf16(weight[i]));
            });
            const uint64_t nb = N / g.tile_n;
            xrt::bo bo_b = xrt::ext::bo(g.device, nb * (K / g.tile_k_l1) * npu_bfp16::tile_bytes(g.tile_n, g.tile_k_l1));
            uint8_t *b = bo_b.map<uint8_t *>();
            parallel_for(nb, std::min<int>(g_threads, (int)nb), [&](uint64_t lo, uint64_t hi) {
                npu_bfp16::pack_weight(wr.data(), N, K, g.tile_n, g.tile_k_l1, b, lo, hi);
            });
            bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            w = g.weights.emplace(weight, bo_b).first;
        }
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
        stats.t_in += seconds_since(t0);
        for (uint64_t l = 0; l < launches; ++l) {
            const uint64_t begin = l * m;
            const uint64_t count = std::min(m, rows - begin);

            auto t1 = clock_type::now();
            parallel_for(count * K, g_threads, [&](uint64_t lo, uint64_t hi) {
                for (uint64_t i = lo; i < hi; ++i) {
                    g.a[i] = to_bf16(x[begin * K + i]);
                }
            });
            if (count < m) {
                std::memset(g.a + count * K, 0, (m - count) * K * sizeof(uint16_t));
            }
            g.bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            // guard the output this call reads (the fp32 accumulator, or the bf16 result)
            if constexpr (std::is_same<Out, float>::value) {
                if (g.c32) guard_arm(g.bo_c32, g.c32, m * N * sizeof(float));
            } else {
                if (g.c16) guard_arm(g.bo_c16, g.c16, m * N * sizeof(uint16_t));
            }
            auto t2 = clock_type::now();

            g.run.start();
            if (g.run.wait2(std::chrono::milliseconds(60000)) == std::cv_status::timeout) {
                throw std::runtime_error(std::string(op) + " launch timed out after 60 s");
            }
            auto t3 = clock_type::now();

            if constexpr (std::is_same<Out, float>::value) {
                if (!g.c32) {
                    throw std::runtime_error(std::string(op) + ": f32 output requested but the GEMM ELF is bfp16w "
                                             "(bf16 output only; unset OPENFISH_NPU_GEMM_OUT)");
                }
                g.bo_c32.sync(XCL_BO_SYNC_BO_FROM_DEVICE);  // fp32 accumulator
                guard_wait(g.bo_c32, g.c32, m * N * sizeof(float), 4, stats);
                parallel_for(count * N, g_threads, [&](uint64_t lo, uint64_t hi) {
                    std::memcpy(out + begin * N + lo, g.c32 + lo, (hi - lo) * sizeof(float));
                });
            } else {
                if (!g.c16) {
                    throw std::runtime_error(std::string(op) + ": bf16 output requested but the GEMM ELF is f32-out "
                                             "(set OPENFISH_NPU_GEMM_OUT=f32)");
                }
                g.bo_c16.sync(XCL_BO_SYNC_BO_FROM_DEVICE);  // bf16 output of the cast launch
                guard_wait(g.bo_c16, g.c16, m * N * sizeof(uint16_t), 2, stats);
                parallel_for(count * N, g_threads, [&](uint64_t lo, uint64_t hi) {
                    std::memcpy(out + begin * N + lo, g.c16 + lo, (hi - lo) * sizeof(uint16_t));
                });
            }
            auto t4 = clock_type::now();

            stats.t_in += std::chrono::duration<double>(t2 - t1).count();
            stats.t_kernel += std::chrono::duration<double>(t3 - t2).count();
            stats.t_out += std::chrono::duration<double>(t4 - t3).count();
        }
        stats.calls += 1;
        stats.launches += launches;
        stats.elements += rows * N;
    } catch (const std::exception &e) {
        OPENFISH_ERROR("linear_npu(%s): %s", op, e.what());
        exit(EXIT_FAILURE);
    }
}

// ---- generic 3-buffer bf16 kernels (rmsnorm: X, W, Y; rope: src, lut, dst), manifest <name>.json in
// OPENFISH_NPU_ARTIFACTS with element counts a, b, c and rows-per-launch m.
namespace {

struct kernel3_t {
    std::string name;
    uint64_t m = 0, n = 0, a = 0, b = 0, c = 0;
    xrt::device device;
    xrt::elf elf;
    xrt::hw_context ctx;
    xrt::ext::kernel kernel;
    xrt::bo bo_a, bo_b, bo_c;
    uint16_t *pa = nullptr, *pb = nullptr, *pc = nullptr;
    xrt::run run;
    const void *b_key = nullptr;  // host pointer whose contents are currently in bo_b

    kernel3_t(const std::string &nm, const std::string &dir, const std::string &json)
        : name(nm),
          m(std::stoull(manifest_field(json, "m"))),
          n(std::stoull(manifest_field(json, "n"))),
          a(std::stoull(manifest_field(json, "a"))),
          b(std::stoull(manifest_field(json, "b"))),
          c(std::stoull(manifest_field(json, "c"))),
          device(0),
          elf(dir + "/" + manifest_field(json, "file")),
          ctx(device, elf),
          kernel(ctx, manifest_field(json, "kernel_name")),
          bo_a(xrt::ext::bo(device, a * sizeof(uint16_t))),
          bo_b(xrt::ext::bo(device, b * sizeof(uint16_t))),
          bo_c(xrt::ext::bo(device, c * sizeof(uint16_t))),
          pa(bo_a.map<uint16_t *>()),
          pb(bo_b.map<uint16_t *>()),
          pc(bo_c.map<uint16_t *>()),
          run(kernel) {
        run.set_arg(0, bo_a);
        run.set_arg(1, bo_b);
        run.set_arg(2, bo_c);
    }
};

std::unordered_map<std::string, kernel3_t *> g_k3;

kernel3_t &get_kernel3(const std::string &name) {
    auto it = g_k3.find(name);
    if (it != g_k3.end()) {
        return *it->second;
    }
    const char *dir = std::getenv("OPENFISH_NPU_ARTIFACTS");
    if (!dir) {
        throw std::runtime_error("OPENFISH_NPU_ARTIFACTS is not set");
    }
    const std::string json = read_file(std::string(dir) + "/" + name + ".json");
    init_common();
    auto *k = new kernel3_t(name, dir, json);
    fprintf(stderr, "[nn_npu] loaded %s: %s/%s (sha256 %s), kernel %s, m=%lu n=%lu\n", name.c_str(), dir,
            manifest_field(json, "file").c_str(), manifest_field(json, "sha256").c_str(),
            manifest_field(json, "kernel_name").c_str(), (unsigned long)k->m, (unsigned long)k->n);
    return *g_k3.emplace(name, k).first->second;
}

npu_stats_t &op_stats(const char *op) {
    auto st = g_op_stats.find(op);
    if (st == g_op_stats.end()) {
        st = g_op_stats.emplace(op, npu_stats_t()).first;
        g_stats.emplace_back(op, &st->second);
    }
    return st->second;
}

// Row-chunked launch: x[rows, n] -> y[rows, n]; buffer b holds `bconst` (b elements), uploaded when it changes.
template <typename In, typename Out>
void run_rows(kernel3_t &k, npu_stats_t &stats, const In *x, Out *y, uint64_t rows, const float *bconst) {
    auto t0 = clock_type::now();
    if (k.b_key != bconst) {
        parallel_for(k.b, g_threads, [&](uint64_t lo, uint64_t hi) {
            for (uint64_t i = lo; i < hi; ++i) k.pb[i] = f32_to_bf16(bconst[i]);
        });
        k.bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        k.b_key = bconst;
    }
    stats.t_in += seconds_since(t0);
    const uint64_t launches = (rows + k.m - 1) / k.m;
    for (uint64_t l = 0; l < launches; ++l) {
        const uint64_t begin = l * k.m;
        const uint64_t count = std::min(k.m, rows - begin);
        auto t1 = clock_type::now();
        parallel_for(count * k.n, g_threads, [&](uint64_t lo, uint64_t hi) {
            for (uint64_t i = lo; i < hi; ++i) k.pa[i] = to_bf16(x[begin * k.n + i]);
        });
        if (count < k.m) std::memset(k.pa + count * k.n, 0, (k.m - count) * k.n * sizeof(uint16_t));
        k.bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        guard_arm(k.bo_c, k.pc, k.m * k.n * sizeof(uint16_t));
        auto t2 = clock_type::now();
        k.run.start();
        if (k.run.wait2(std::chrono::milliseconds(60000)) == std::cv_status::timeout) {
            throw std::runtime_error(k.name + " launch timed out after 60 s");
        }
        auto t3 = clock_type::now();
        k.bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        guard_wait(k.bo_c, k.pc, k.m * k.n * sizeof(uint16_t), 2, stats);
        parallel_for(count * k.n, g_threads, [&](uint64_t lo, uint64_t hi) {
            for (uint64_t i = lo; i < hi; ++i) store_bf16(y[begin * k.n + i], k.pc[i]);
        });
        auto t4 = clock_type::now();
        stats.t_in += std::chrono::duration<double>(t2 - t1).count();
        stats.t_kernel += std::chrono::duration<double>(t3 - t2).count();
        stats.t_out += std::chrono::duration<double>(t4 - t3).count();
    }
    stats.calls += 1;
    stats.launches += launches;
    stats.elements += rows * k.n;
}

}  // namespace

template <typename T>
static void rmsnorm_impl(const T *x, const float *weight, T *y, uint64_t rows, uint64_t N) {
    std::lock_guard<std::mutex> guard(g_mutex);
    try {
        kernel3_t &k = get_kernel3("rmsnorm_n" + std::to_string(N));
        if (k.n != N || k.b != N) throw std::runtime_error("rmsnorm shape mismatch");
        run_rows(k, op_stats("rmsnorm"), x, y, rows, weight);
    } catch (const std::exception &e) {
        OPENFISH_ERROR("rmsnorm_npu: %s", e.what());
        exit(EXIT_FAILURE);
    }
}

template <typename T>
static void rope_impl(const T *x, const float *lut, T *y, uint64_t rows, uint64_t head_dim) {
    std::lock_guard<std::mutex> guard(g_mutex);
    try {
        kernel3_t &k = get_kernel3("rope_d" + std::to_string(head_dim));
        if (k.n != head_dim || rows % k.m) {
            throw std::runtime_error("rope: rows " + std::to_string(rows) + " not a multiple of compiled " +
                                     std::to_string(k.m) + " (LUT repeats every m rows)");
        }
        run_rows(k, op_stats("rope"), x, y, rows, lut);
    } catch (const std::exception &e) {
        OPENFISH_ERROR("rope_npu: %s", e.what());
        exit(EXIT_FAILURE);
    }
}

// ---- generic 1-D bf16 elementwise kernels (air.api examples: axpy = (x, y, out), silu = (x, out)), manifest
// <name>.json in OPENFISH_NPU_ARTIFACTS with n elements per launch and the number of inputs.
namespace {

struct eltwise_kernel_t {
    std::string name, json;
    uint64_t n = 0;
    int inputs = 0;
    xrt::device device;
    xrt::elf elf;
    xrt::hw_context ctx;
    xrt::ext::kernel kernel;
    xrt::bo bo_in[2], bo_out;
    uint16_t *in[2] = {nullptr, nullptr}, *out = nullptr;
    xrt::run run;

    eltwise_kernel_t(const std::string &nm, const std::string &dir, const std::string &js)
        : name(nm),
          json(js),
          n(std::stoull(manifest_field(js, "n"))),
          inputs(std::stoi(manifest_field(js, "inputs"))),
          device(0),
          elf(dir + "/" + manifest_field(js, "file")),
          ctx(device, elf),
          kernel(ctx, manifest_field(js, "kernel_name")),
          run(kernel) {
        if (inputs < 1 || inputs > 2) throw std::runtime_error(nm + ": manifest inputs must be 1 or 2");
        for (int i = 0; i < inputs; ++i) {
            bo_in[i] = xrt::ext::bo(device, n * sizeof(uint16_t));
            in[i] = bo_in[i].map<uint16_t *>();
            run.set_arg(i, bo_in[i]);
        }
        bo_out = xrt::ext::bo(device, n * sizeof(uint16_t));
        out = bo_out.map<uint16_t *>();
        run.set_arg(inputs, bo_out);
    }
};

std::unordered_map<std::string, eltwise_kernel_t *> g_ew;

eltwise_kernel_t &get_eltwise(const std::string &name) {
    auto it = g_ew.find(name);
    if (it != g_ew.end()) {
        return *it->second;
    }
    const char *dir = std::getenv("OPENFISH_NPU_ARTIFACTS");
    if (!dir) {
        throw std::runtime_error("OPENFISH_NPU_ARTIFACTS is not set");
    }
    const std::string json = read_file(std::string(dir) + "/" + name + ".json");
    init_common();
    auto *k = new eltwise_kernel_t(name, dir, json);
    fprintf(stderr, "[nn_npu] loaded %s: %s/%s (sha256 %s), kernel %s, n=%lu, inputs %d\n", name.c_str(), dir,
            manifest_field(json, "file").c_str(), manifest_field(json, "sha256").c_str(),
            manifest_field(json, "kernel_name").c_str(), (unsigned long)k->n, k->inputs);
    return *g_ew.emplace(name, k).first->second;
}

// out[i] = f(a[i][, b[i]]) over total elements, n per launch, the last launch zero-padded.
void run_eltwise(eltwise_kernel_t &k, npu_stats_t &stats, const uint16_t *a, const uint16_t *b, uint16_t *o, uint64_t total) {
    const uint16_t *src[2] = {a, b};
    const uint64_t launches = (total + k.n - 1) / k.n;
    for (uint64_t l = 0; l < launches; ++l) {
        const uint64_t begin = l * k.n;
        const uint64_t count = std::min(k.n, total - begin);
        auto t0 = clock_type::now();
        for (int i = 0; i < k.inputs; ++i) {
            parallel_for(count, g_threads, [&](uint64_t lo, uint64_t hi) {
                std::memcpy(k.in[i] + lo, src[i] + begin + lo, (hi - lo) * sizeof(uint16_t));
            });
            if (count < k.n) std::memset(k.in[i] + count, 0, (k.n - count) * sizeof(uint16_t));
            k.bo_in[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        guard_arm(k.bo_out, k.out, k.n * sizeof(uint16_t));
        auto t1 = clock_type::now();
        k.run.start();
        if (k.run.wait2(std::chrono::milliseconds(60000)) == std::cv_status::timeout) {
            throw std::runtime_error(k.name + " launch timed out after 60 s");
        }
        auto t2 = clock_type::now();
        k.bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        guard_wait(k.bo_out, k.out, k.n * sizeof(uint16_t), 2, stats);
        parallel_for(count, g_threads, [&](uint64_t lo, uint64_t hi) {
            std::memcpy(o + begin + lo, k.out + lo, (hi - lo) * sizeof(uint16_t));
        });
        auto t3 = clock_type::now();
        stats.t_in += std::chrono::duration<double>(t1 - t0).count();
        stats.t_kernel += std::chrono::duration<double>(t2 - t1).count();
        stats.t_out += std::chrono::duration<double>(t3 - t2).count();
    }
    stats.calls += 1;
    stats.launches += launches;
    stats.elements += total;
}

// ---- fused RMSNorm -> Q/K/V GEMMs -> RoPE(Q, K) (mlir-air llms rms_gemms_rope_multi, 13 args, m rows per launch):
//   0 x_in[m,d] 1 norm_w[d] 2 normed[m,d] 3 wq[d,d] 4 q 5 wk[d,d] 6 k 7 wv[d,d] 8 v 9 lut_q[m*d] 10 lut_k 11 q_roped 12 k_roped
// Weights (B = W.T per q/k/v slice of wqkv) and the norm weight are bf16-resident per host pointer; the LUT rows
// for positions t0..t0+m-1 are one buffer per t0 (rows of a launch never straddle a chunk: T % m == 0).
struct rqr_kernel_t {
    uint64_t m = 0, d = 0, heads = 0, hd = 0;
    xrt::device device;
    xrt::elf elf;
    xrt::hw_context ctx;
    xrt::ext::kernel kernel;
    xrt::bo bo[13];
    uint16_t *p[13] = {};
    xrt::run run;
    std::unordered_map<const float *, std::vector<xrt::bo>> wqkv;
    std::unordered_map<const float *, xrt::bo> normw;
    const float *lut_key = nullptr;
    uint64_t lut_T = 0;
    std::vector<xrt::bo> luts;

    rqr_kernel_t(const std::string &dir, const std::string &json)
        : m(std::stoull(manifest_field(json, "m"))),
          d(std::stoull(manifest_field(json, "n"))),
          heads(std::stoull(manifest_field(json, "heads"))),
          hd(std::stoull(manifest_field(json, "head_dim"))),
          device(0),
          elf(dir + "/" + manifest_field(json, "file")),
          ctx(device, elf),
          kernel(ctx, manifest_field(json, "kernel_name")),
          run(kernel) {
        if (std::stoi(manifest_field(json, "nargs")) != 13 || heads * hd != d) {
            throw std::runtime_error("rms_qkv_rope manifest: expected 13 args and heads*head_dim == n");
        }
        for (int i : {0, 2, 4, 6, 8, 11, 12}) {
            bo[i] = xrt::ext::bo(device, m * d * sizeof(uint16_t));
            p[i] = bo[i].map<uint16_t *>();
            run.set_arg(i, bo[i]);
        }
    }
};

rqr_kernel_t *g_rqr = nullptr;

xrt::bo bf16_bo(xrt::device &dev, uint64_t n, const std::function<float(uint64_t)> &val) {
    xrt::bo b = xrt::ext::bo(dev, n * sizeof(uint16_t));
    uint16_t *ptr = b.map<uint16_t *>();
    parallel_for(n, g_threads, [&](uint64_t lo, uint64_t hi) {
        for (uint64_t i = lo; i < hi; ++i) ptr[i] = f32_to_bf16(val(i));
    });
    b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return b;
}

}  // namespace

void axpy_npu_bf16(float alpha, const uint16_t *x, const uint16_t *y, uint16_t *out, uint64_t n) {
    std::lock_guard<std::mutex> guard(g_mutex);
    try {
        eltwise_kernel_t &k = get_eltwise("axpy");
        // the kernel holds alpha as a bf16 constant: the caller's alpha must round to the same bf16 value
        // (SUP's deepnorm alpha is 2.4494897 in fp32 and 2.449219 in the iGPU's fp16 model; both are bf16 2.453125)
        const float compiled = std::stof(manifest_field(k.json, "alpha"));
        if (k.inputs != 2 || f32_to_bf16(compiled) != f32_to_bf16(alpha)) {
            throw std::runtime_error("alpha " + std::to_string(alpha) + " does not round to the compiled bf16 alpha " +
                                     std::to_string(compiled));
        }
        run_eltwise(k, op_stats("residual"), x, y, out, n);
    } catch (const std::exception &e) {
        OPENFISH_ERROR("axpy_npu: %s", e.what());
        exit(EXIT_FAILURE);
    }
}

void silu_npu_bf16(const uint16_t *x, uint16_t *out, uint64_t n) {
    std::lock_guard<std::mutex> guard(g_mutex);
    try {
        eltwise_kernel_t &k = get_eltwise("silu");
        if (k.inputs != 1) throw std::runtime_error("silu manifest: expected 1 input");
        run_eltwise(k, op_stats("swish"), x, nullptr, out, n);
    } catch (const std::exception &e) {
        OPENFISH_ERROR("silu_npu: %s", e.what());
        exit(EXIT_FAILURE);
    }
}

void rms_qkv_rope_npu_bf16(const uint16_t *x, const float *norm_w, const float *wqkv, const float *lut, uint16_t *normed,
                           uint16_t *qkv, uint64_t rows, uint64_t T) {
    std::lock_guard<std::mutex> guard(g_mutex);
    try {
        if (!g_rqr) {
            const char *dir = std::getenv("OPENFISH_NPU_ARTIFACTS");
            if (!dir) throw std::runtime_error("OPENFISH_NPU_ARTIFACTS is not set");
            const std::string json = read_file(std::string(dir) + "/rms_qkv_rope.json");
            init_common();
            g_rqr = new rqr_kernel_t(dir, json);
            fprintf(stderr, "[nn_npu] loaded rms_qkv_rope: %s/%s (sha256 %s), kernel %s, m=%lu d=%lu heads=%lu, gemm %s\n", dir,
                    manifest_field(json, "file").c_str(), manifest_field(json, "sha256").c_str(),
                    manifest_field(json, "kernel_name").c_str(), (unsigned long)g_rqr->m, (unsigned long)g_rqr->d,
                    (unsigned long)g_rqr->heads, manifest_field(json, "gemm_method").c_str());
        }
        rqr_kernel_t &k = *g_rqr;
        npu_stats_t &stats = op_stats("rms_qkv_rope");
        const uint64_t m = k.m, d = k.d;
        if (T % m || rows % T) {
            throw std::runtime_error("T " + std::to_string(T) + " must be a multiple of the compiled rows " +
                                     std::to_string(m) + " and rows a multiple of T");
        }
        auto t0 = clock_type::now();
        if (k.lut_key != lut || k.lut_T != T) {
            k.luts.clear();
            const uint64_t per = m * d;  // m positions x heads rows x head_dim
            for (uint64_t t = 0; t < T; t += m) {
                const float *src = lut + t * d;
                k.luts.push_back(bf16_bo(k.device, per, [&](uint64_t i) { return src[i]; }));
            }
            k.lut_key = lut;
            k.lut_T = T;
        }
        auto nw = k.normw.find(norm_w);
        if (nw == k.normw.end()) {
            nw = k.normw.emplace(norm_w, bf16_bo(k.device, d, [&](uint64_t i) { return norm_w[i]; })).first;
        }
        auto w = k.wqkv.find(wqkv);
        if (w == k.wqkv.end()) {
            std::vector<xrt::bo> v;
            for (uint64_t sl = 0; sl < 3; ++sl) {  // B[e, j] = W[sl*d + j, e]
                const float *ws = wqkv + sl * d * d;
                v.push_back(bf16_bo(k.device, d * d, [&](uint64_t i) { return ws[(i % d) * d + i / d]; }));
            }
            w = k.wqkv.emplace(wqkv, std::move(v)).first;
        }
        k.run.set_arg(1, nw->second);
        k.run.set_arg(3, w->second[0]);
        k.run.set_arg(5, w->second[1]);
        k.run.set_arg(7, w->second[2]);
        stats.t_in += seconds_since(t0);

        const uint64_t launches = rows / m;
        for (uint64_t l = 0; l < launches; ++l) {
            const uint64_t begin = l * m;
            auto t1 = clock_type::now();
            const xrt::bo &lb = k.luts[(begin % T) / m];
            k.run.set_arg(9, lb);
            k.run.set_arg(10, lb);
            parallel_for(m * d, g_threads, [&](uint64_t lo, uint64_t hi) {
                std::memcpy(k.p[0] + lo, x + begin * d + lo, (hi - lo) * sizeof(uint16_t));
            });
            k.bo[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
            for (int i : {2, 11, 12, 8}) guard_arm(k.bo[i], k.p[i], m * d * sizeof(uint16_t));
            auto t2 = clock_type::now();
            k.run.start();
            if (k.run.wait2(std::chrono::milliseconds(60000)) == std::cv_status::timeout) {
                throw std::runtime_error("rms_qkv_rope launch timed out after 60 s");
            }
            auto t3 = clock_type::now();
            for (int i : {2, 11, 12, 8}) {
                k.bo[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                guard_wait(k.bo[i], k.p[i], m * d * sizeof(uint16_t), 2, stats);
            }
            parallel_for(m, g_threads, [&](uint64_t lo, uint64_t hi) {
                for (uint64_t r = lo; r < hi; ++r) {
                    const uint64_t row = begin + r;
                    std::memcpy(normed + row * d, k.p[2] + r * d, d * sizeof(uint16_t));
                    std::memcpy(qkv + row * 3 * d, k.p[11] + r * d, d * sizeof(uint16_t));
                    std::memcpy(qkv + row * 3 * d + d, k.p[12] + r * d, d * sizeof(uint16_t));
                    std::memcpy(qkv + row * 3 * d + 2 * d, k.p[8] + r * d, d * sizeof(uint16_t));
                }
            });
            auto t4 = clock_type::now();
            stats.t_in += std::chrono::duration<double>(t2 - t1).count();
            stats.t_kernel += std::chrono::duration<double>(t3 - t2).count();
            stats.t_out += std::chrono::duration<double>(t4 - t3).count();
        }
        stats.calls += 1;
        stats.launches += launches;
        stats.elements += rows * d;
    } catch (const std::exception &e) {
        OPENFISH_ERROR("rms_qkv_rope_npu: %s", e.what());
        exit(EXIT_FAILURE);
    }
}

// float32 host I/O (CPU build) and bf16 host I/O (iGPU build: tensors cast fp16 -> bf16 on the GPU)
void silu_mul_npu(const float *x, float *o, uint64_t MN, uint64_t K) { silu_mul_impl(x, o, MN, K); }
void silu_mul_npu_bf16(const uint16_t *x, uint16_t *o, uint64_t MN, uint64_t K) { silu_mul_impl(x, o, MN, K); }
void linear_npu(const char *op, const float *x, float *out, const float *w, uint64_t rows, uint64_t K, uint64_t N) {
    linear_impl(op, x, out, w, rows, K, N);
}
void linear_npu_bf16(const char *op, const uint16_t *x, uint16_t *out, const float *w, uint64_t rows, uint64_t K, uint64_t N) {
    linear_impl(op, x, out, w, rows, K, N);
}
void linear_npu_bf16_f32(const char *op, const uint16_t *x, float *out, const float *w, uint64_t rows, uint64_t K, uint64_t N) {
    linear_impl(op, x, out, w, rows, K, N);
}
void rmsnorm_npu(const float *x, const float *w, float *y, uint64_t rows, uint64_t N) { rmsnorm_impl(x, w, y, rows, N); }
void rmsnorm_npu_bf16(const uint16_t *x, const float *w, uint16_t *y, uint64_t rows, uint64_t N) { rmsnorm_impl(x, w, y, rows, N); }
void rope_npu(const float *x, const float *lut, float *y, uint64_t rows, uint64_t d) { rope_impl(x, lut, y, rows, d); }
void rope_npu_bf16(const uint16_t *x, const float *lut, uint16_t *y, uint64_t rows, uint64_t d) { rope_impl(x, lut, y, rows, d); }
