#!/usr/bin/env python3
"""Phase-1 harness: silu_mul on NPU2 vs reference.py, in openfish's [y ‖ gate] layout.

One compile, one load, three invocations of the registry silu_and_mul kernel
(kernels/silu_mul, N elements, bf16, herd 8x1):
  1. randn N(0,1) gate/up (the registry protocol)                    -> must PASS
  2. real SUP activations: slorado dump L0_ff_fc1_out [rows,1024,4096],
     split host-side from [y ‖ gate] into (gate, up)                  -> must PASS
  3. negative control: case 2 with gate and up swapped                -> must FAIL
Gate: element-wise |out-ref| <= atol + rtol*|ref| on the full output
(registry SiLU_Mul_bf16: rtol 1.6e-2, atol 8e-2). Exit 0 only if all three hold.

Run on the server, serialized on the private NPU lock:
  source ~/npu-env.sh && cd ~/slorado/openfish/npu
  OPENFISH_DUMP_DIR=~/p0/dump_sup8 flock -x -w 1800 $NPU_LOCK python3 test/test_silu_mul_npu.py [--perf-iters 30]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
KERNEL_DIR = HERE.parent / "kernels" / "silu_mul"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KERNEL_DIR))

import reference as ref  # noqa: E402

RTOL, ATOL = 1.6e-2, 8e-2


def n_fail(out_f32, ref_f32):
    return int((np.abs(out_f32 - ref_f32) > ATOL + RTOL * np.abs(ref_f32)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16777216)
    ap.add_argument("--tile-n", type=int, default=4096)
    ap.add_argument("--herd-x", type=int, default=8)
    ap.add_argument("--herd-y", type=int, default=1)
    ap.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
    ap.add_argument("--perf-iters", type=int, default=0)
    ap.add_argument("--dump-dir", default=os.environ.get("OPENFISH_DUMP_DIR"))
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()
    n = args.n

    # Real activations first, so a missing/short dump fails before compiling.
    dump = Path(args.dump_dir or "") / f"L{args.layer}_ff_fc1_out.npy"
    if not dump.exists():
        print(f"FAIL: missing {dump} (set OPENFISH_DUMP_DIR)")
        return 2
    x = np.load(dump)  # [rows, T, 2K] float32, [y ‖ gate]
    if x[..., : x.shape[-1] // 2].size != n:
        print(f"FAIL: dump {x.shape} gives {x[..., : x.shape[-1] // 2].size} elements per half, need N={n} "
              f"(re-dump with OPENFISH_DUMP_ROWS = N / (T*K))")
        return 2
    cpu_out = np.load(dump.with_name(f"L{args.layer}_ff_silu_mul_out.npy"))

    subprocess.run(["make", "-s", "-C", str(KERNEL_DIR), "compile-kernel"], check=True)
    os.chdir(KERNEL_DIR / "build_peano")  # aircc picks silu_and_mul.o up from cwd

    from air.backend.xrt import XRTBackend
    from air.backend.xrt_runner import XRTRunner
    from silu_and_mul import build_module

    print(f"silu_mul NPU: N={n} tile_n={args.tile_n} herd={args.herd_x}x{args.herd_y} "
          f"format={args.output_format} rtol={RTOL} atol={ATOL}")
    module = build_module(n, args.tile_n, bfloat16, herd_x=args.herd_x, herd_y=args.herd_y)
    backend = XRTBackend(target_device="npu2", omit_while_true_loop=False,
                         output_format=args.output_format, instance_name="silu_and_mul")
    t0 = time.perf_counter()
    artifact = backend.compile(module)
    print(f"compile: {time.perf_counter() - t0:.1f} s -> {artifact.output_binary}")
    checker = XRTRunner(report_precision=True)  # same precision print + isclose as the registry harness

    ok = True
    invoke = backend.load(artifact)
    try:
        # ---- case 1: randn (registry protocol)
        rng = np.random.default_rng(0)
        gate = rng.standard_normal(n, dtype=np.float32).astype(bfloat16)
        up = rng.standard_normal(n, dtype=np.float32).astype(bfloat16)
        expected = (ref.silu(gate.astype(np.float32)) * up.astype(np.float32)).astype(bfloat16)
        backend.n_perf_iters = args.perf_iters
        t0 = time.perf_counter()
        out = invoke(gate, up, np.zeros(n, bfloat16))[2]
        wall1 = time.perf_counter() - t0
        backend.n_perf_iters = 0
        print("== case 1: randn N(0,1)")
        c1 = checker._check_outputs([out], [expected], rtol=RTOL, atol=ATOL)
        print("case 1:", "PASS" if c1 else "FAIL")
        if args.perf_iters:
            lat = backend.last_latency_us
            print(f"perf: kernel-only mean latency {lat:.1f} us over {args.perf_iters} iters "
                  f"(10 warmup) -> {3 * n * 2 / (lat * 1e-6) / 1e9:.1f} GB/s")
        else:
            print(f"invoke wall (BO alloc + copy + sync + run, single shot): {wall1 * 1e3:.1f} ms")
        ok &= c1

        # ---- case 2: real SUP activations, openfish layout through the host adapter
        t0 = time.perf_counter()
        x_bf = x.astype(bfloat16)
        t_cast = time.perf_counter() - t0
        t0 = time.perf_counter()
        k = x.shape[-1] // 2
        g2 = np.ascontiguousarray(x_bf[..., k:]).reshape(n)
        u2 = np.ascontiguousarray(x_bf[..., :k]).reshape(n)
        t_split = time.perf_counter() - t0
        expected2 = ref.silu_mul_ref(x_bf.astype(np.float32)).astype(bfloat16).reshape(n)
        t0 = time.perf_counter()
        out2 = invoke(g2, u2, np.zeros(n, bfloat16))[2]
        wall2 = time.perf_counter() - t0
        print(f"== case 2: slorado SUP layer {args.layer} ff_fc1_out {tuple(x.shape)} via [y|gate] split")
        c2 = checker._check_outputs([out2], [expected2], rtol=RTOL, atol=ATOL)
        print("case 2:", "PASS" if c2 else "FAIL")
        print(f"host adapter: f32->bf16 cast {t_cast * 1e3:.0f} ms, split {t_split * 1e3:.0f} ms, "
              f"invoke wall {wall2 * 1e3:.0f} ms")
        o2 = out2.astype(np.float32)
        c = cpu_out.reshape(n)
        err = np.abs(o2 - c)
        print(f"info: NPU vs slorado CPU fp32 output: mean_rel_L1={err.mean() / np.abs(c).mean():.3e} "
              f"max_abs={err.max():.3e} n_outside_tol={n_fail(o2, c)}")
        ok &= c2

        # ---- case 3: negative control (swap gate/up) must fail
        out3 = invoke(u2, g2, np.zeros(n, bfloat16))[2]
        nf = n_fail(out3.astype(np.float32), expected2.astype(np.float32))
        c3 = nf > 0
        print(f"== case 3: swapped gate/up -> {nf} elements outside tol: {'PASS (fails as it must)' if c3 else 'FAIL (control did not fail)'}")
        ok &= c3
    finally:
        backend.unload()

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
