// Diagnose the stale-output race (development_progress/debug_log.md 2026-10-24): after run.wait() returns and the
// output BO is synced from the device, is all of the output visible to the CPU, and if not, when does it land?
// XRT only (no HIP), same BO/launch pattern as nn_npu.cpp.
//
// Per launch: fill the observed output BO with 0xFF bytes (NaN in bf16 and f32), sync it to the device, run, wait,
// then sync from the device and scan the whole output for sentinel elements (= not yet written, as seen by the CPU).
// If any remain, poll only that range until they are gone:
//   mode sync    bo.sync(FROM_DEVICE) on the range before each scan (what a retry in nn_npu.cpp would do)
//   mode nosync  re-read the map without syncing (data appearing anyway = late DMA write, not a stale CPU cache)
// Inputs: every other argument is filled once with bf16 values in [-1, 1).
//
//   g++ -O2 -std=c++17 tail_race_test.cpp -lxrt_coreutil -o tail_race_test
//   flock -x -w 600 $HOME/.npu.lock tail_race_test <elf> <kernel_name> <launches> <out_arg> <out_elem_bytes> \
//       <arg0_bytes> [<arg1_bytes> ...]
// Exit 0 if no launch showed sentinel elements after wait + sync, 1 otherwise.

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;

static double us_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::micro>(clock_type::now() - t0).count();
}

struct scan_t {
    uint64_t count = 0, lo = 0, hi = 0;  // sentinel elements and their index range [lo, hi]
};

static scan_t scan(const uint8_t *p, uint64_t elem_bytes, uint64_t lo, uint64_t hi) {
    scan_t s;
    for (uint64_t i = lo; i <= hi; ++i) {
        const uint8_t *e = p + i * elem_bytes;
        bool all = true;
        for (uint64_t b = 0; b < elem_bytes; ++b) all &= e[b] == 0xFF;
        if (all) {
            if (!s.count) s.lo = i;
            s.hi = i;
            ++s.count;
        }
    }
    return s;
}

static double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5))];
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s <elf> <kernel_name> <launches> <out_arg> <out_elem_bytes> <arg0_bytes> [...]\n", argv[0]);
        return 2;
    }
    const std::string elf_path = argv[1], kname = argv[2];
    const int launches = std::atoi(argv[3]);
    const int out_arg = std::atoi(argv[4]);
    const uint64_t eb = std::strtoull(argv[5], nullptr, 10);
    std::vector<uint64_t> sizes;
    for (int i = 6; i < argc; ++i) sizes.push_back(std::strtoull(argv[i], nullptr, 10));
    const uint64_t out_bytes = sizes.at(out_arg), n = out_bytes / eb;

    xrt::device dev(0);
    xrt::elf elf(elf_path);
    xrt::hw_context ctx(dev, elf);
    xrt::ext::kernel kernel(ctx, kname);
    std::vector<xrt::bo> bos;
    for (size_t i = 0; i < sizes.size(); ++i) {
        bos.push_back(xrt::ext::bo(dev, sizes[i]));
        if ((int)i == out_arg) continue;
        auto *p = bos.back().map<uint16_t *>();
        uint32_t h = 12345u + (uint32_t)i;
        for (uint64_t j = 0; j < sizes[i] / 2; ++j) {
            h = h * 1664525u + 1013904223u;
            float f = ((int)(h >> 8 & 0xffff) - 32768) / 32768.0f;
            uint32_t u;
            std::memcpy(&u, &f, 4);
            p[j] = (uint16_t)(u >> 16);
        }
        bos.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    xrt::run run(kernel);
    for (size_t i = 0; i < bos.size(); ++i) run.set_arg((int)i, bos[i]);
    auto &out = bos[out_arg];
    auto *op = out.map<uint8_t *>();
    printf("%s (%s): %zu args, observing arg %d (%lu elements x %lu bytes = %.2f MiB), %d launches per mode\n",
           elf_path.c_str(), kname.c_str(), sizes.size(), out_arg, (unsigned long)n, (unsigned long)eb, out_bytes / 1048576.0,
           launches);

    bool clean = true;
    for (int mode = 0; mode < 2; ++mode) {
        int hits = 0, never = 0;
        std::vector<double> elems, from_end, clear_us, polls, kern_us;
        for (int l = 0; l < launches + 1; ++l) {
            std::memset(op, 0xFF, out_bytes);
            out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto tk = clock_type::now();
            run.start();
            if (run.wait2(std::chrono::milliseconds(60000)) == std::cv_status::timeout) {
                fprintf(stderr, "launch timed out\n");
                return 2;
            }
            const double k = us_since(tk);
            auto t0 = clock_type::now();
            out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            scan_t s = scan(op, eb, 0, n - 1);
            if (l == 0) continue;  // warm-up
            kern_us.push_back(k);
            if (!s.count) continue;
            ++hits;
            elems.push_back((double)s.count);
            from_end.push_back((double)(n - s.lo));
            // poll the range until it is written (or 100 ms)
            const uint64_t page = 4096, b0 = s.lo * eb / page * page;
            const uint64_t b1 = std::min(out_bytes, ((s.hi + 1) * eb + page - 1) / page * page);
            int p = 0;
            scan_t r = s;
            while (r.count && us_since(t0) < 100000) {
                if (mode == 0) out.sync(XCL_BO_SYNC_BO_FROM_DEVICE, b1 - b0, b0);
                r = scan(op, eb, s.lo, s.hi);
                ++p;
            }
            if (r.count) ++never;
            clear_us.push_back(us_since(t0));
            polls.push_back(p);
        }
        clean &= hits == 0;
        printf("  mode %-6s: %d/%d launches had unwritten output after wait + sync", mode ? "nosync" : "sync", hits, launches);
        if (hits) {
            printf("; elements median %.0f max %.0f; first unwritten element median %.0f / max %.0f from the end; "
                   "visible after median %.1f us / max %.1f us (%.0f / %.0f polls)%s",
                   pct(elems, 0.5), pct(elems, 1), pct(from_end, 0.5), pct(from_end, 1), pct(clear_us, 0.5), pct(clear_us, 1),
                   pct(polls, 0.5), pct(polls, 1), never ? "" : "");
            if (never) printf("; %d never became visible within 100 ms", never);
        }
        printf("; kernel (start..wait) median %.0f us\n", pct(kern_us, 0.5));
    }
    printf("RESULT: %s\n", clean ? "CLEAN" : "RACE");
    return clean ? 0 : 1;
}
