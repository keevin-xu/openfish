# openfish/npu — AMD XDNA2 (NPU2) backend research workspace

Porting openfish/slorado kernels to the Strix Halo NPU via mlir-air.
Everything here runs on the server (`zhang-ryzen2`) after `source ~/npu-env.sh`.
Plan and gates: `npu-adapting/9-16-npu-adapting-context/05-kernel-plan.md`.
Phase status per kernel: [`TODO.md`](TODO.md).

```
npu/
├── build.sh                 builds every kernels/<k>/ (make artifact) into artifacts/
├── kernels/<kernel>/        AIR .py + microkernel .cc + Makefile (adapted from mlir-air examples)
├── artifacts/               compiled xclbin/elf + insts (gitignored)
├── test/reference.py        FP32 numpy references in openfish layouts (the Phase-1 oracle)
├── test/check_reference_vs_dump.py   reference.py vs slorado CPU tensor dumps
├── test/run_all.py          one command, exit code: every check under test/
├── bench/                   npu-microbench outputs
└── docs/development_progress/{progress,LESSONS,debug_log,phase_timing}.md  (append-only)
```

## Commands (server)

```bash
source ~/npu-env.sh && cd ~/slorado/openfish/npu
./build.sh
OPENFISH_DUMP_DIR=~/p0/dump_sup python3 test/run_all.py
```

## Tensor dumps (CPU oracle activations)

Built from slorado's guarded hooks in `thirdparty/dorado/TxModel.cpp`:

```bash
cd ~/slorado && rm -f build/TxModel.o && CPPFLAGS=-DOPENFISH_DUMP make -j16 cxx11_abi=1 \
  && cp slorado ~/p0/slorado-dump && rm -f build/TxModel.o && make -j16 cxx11_abi=1
mkdir -p ~/p0/dump_sup && OPENFISH_DUMP_DIR=$HOME/p0/dump_sup OPENFISH_DUMP_EXIT=1 \
  ~/p0/slorado-dump basecaller -x cpu -C 128 models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0 \
  test/PGXXXX230339/reads_1k.blow5 -o /dev/null
```

Optional: `OPENFISH_DUMP_LAYERS=n` (default 1), `OPENFISH_DUMP_ROWS=r` batch rows kept (default 4).
