#!/usr/bin/env python3
"""CPU baseline: torch fp32 linear (what slorado's CPU path runs) at sup@v5.0.0 shapes.

Median / p95 over --iters after warmup, at several thread counts. GFLOPS = 2*M*K*N / t.
Note: airenv torch (not slorado's libtorch 2.9.0+rocm6.4), same op (aten::linear, no bias).
"""
import argparse
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, nargs="+", default=[4096, 131072])
ap.add_argument("--threads", type=int, nargs="+", default=[8, 16])
ap.add_argument("--iters", type=int, default=30)
args = ap.parse_args()

shapes = [("fc1", 512, 4096), ("fc2", 2048, 512), ("wqkv", 512, 1536), ("out_proj", 512, 512)]
print(f"torch {torch.__version__}, {torch.backends.cpu.get_cpu_capability()}")
print(f"{'op':<9} {'M':>7} {'K':>5} {'N':>5} {'thr':>4} {'median_ms':>10} {'p95_ms':>9} {'GFLOPS':>8}")
for m in args.m:
    iters = args.iters if m <= 8192 else 5
    for name, k, n in shapes:
        x = torch.randn(m, k)
        w = torch.randn(n, k) / k ** 0.5
        for thr in args.threads:
            torch.set_num_threads(thr)
            for _ in range(3):
                torch.nn.functional.linear(x, w)
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                torch.nn.functional.linear(x, w)
                ts.append(time.perf_counter() - t0)
            med, p95 = np.median(ts), np.percentile(ts, 95)
            print(f"{name:<9} {m:>7} {k:>5} {n:>5} {thr:>4} {med*1e3:>10.2f} {p95*1e3:>9.2f} {2*m*k*n/med/1e9:>8.1f}", flush=True)
