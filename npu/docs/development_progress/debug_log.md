# openfish/npu — debug log (append-only)

## 2026-09-16
- **sup CPU run SIGKILLed** at default `-C 512`: max RSS 26.9 GB on a 30 GiB host, probable OOM (dmesg not
  readable). Rerun with `-C 128` (README sup batch): PASS at 12.2 GB peak. Not fully confirmed whether the kill
  was the OOM killer or a manual `kill -9`.
- **Accuracy script** needs `<ref.fa> <fastq>` and minimap2 + datamash; none were on the server. Installed
  per-user in `~/tools` (no sudo).
- **`git submodule sync` reverted a `git config submodule.openfish.url` override** — sync copies from
  `.gitmodules`. Fixed by pointing `.gitmodules` at the fork.

## 2026-09-16 (Phase 2)
- `patch_value() only supports 64-bit values or less` on the first launch: `xrt::run::set_arg`'s forwarding
  template matched `xrt::ext::bo` better than the `const xrt::bo&` overload, so the BO was patched as a 32-byte
  scalar. Fix: hold BOs as `xrt::bo` built from `xrt::ext::bo(...)`.
- `git add include/openfish/openfish.h` refused (openfish `.gitignore` has `include`, tracked file) → needed `-f`.

### 2026-10-24 — stale tail in NPU output after run.wait() (copy path)
- Found by `npu/test/zero_copy_test.cpp` baseline (axpy ELF, n 4,194,304 bf16, ext::bo + bo.sync(FROM_DEVICE) + memcpy from the map — the nn_npu.cpp pattern): 10/31 launches have 8–328 wrong elements, all within the last 512 elements, each equal to the previous launch's result at that index.
- 2 ms sleep after `run.wait2()` before `bo.sync(FROM_DEVICE)`: 0/31. sync, 2 ms, sync again: 0/31. Zero-copy reads by the iGPU (methods B/C/D): 0/35 each.
- Interpretation: the last output DMA can complete after the run reports done (or the first sync's cache invalidation runs too early). Not yet fixed in nn_npu.cpp; open question how often it hits real ops and whether it moved past identity numbers.

### 2026-10-24 (cont.) — race diagnostic
- Runtime sequence awaits every output DMA task: not a missing wait in our builds.
- `npu/test/tail_race_test.cpp` (XRT only): with the nn_npu.cpp pattern, 1 / ≈ 3,400 launches over 9 kernel kinds (rope_d64: last 392 elements = previous launch's, correct after 1.05 ms). With the CPU writing + syncing the output before each launch (sentinel modes): 0 / ≈ 6,800.
- Same pattern with HIP active (`zero_copy_test` baseline, axpy): 5/41 launches, already wrong in the CPU's read of the BO.
- Candidate fix: before each launch write a sentinel into the output's last few KiB and sync that range; after wait + sync, poll that range until the sentinel is gone. Sound as a detector: the CPU's last write there is the sentinel, so anything else read is the NPU's new data.

### 2026-10-24 (cont.) — guard implemented; race absent in slorado
- `src/nn_npu.cpp`: guard_arm()/guard_wait() around every launch (tail 4 KiB marker + re-sync, per-op counters, OPENFISH_NPU_GUARD=0 off).
- comb12 (50 reads, 12 ops, native f32-out GEMMs), binary 8822a258: guard on ×2 → 0 waits in 14,752 launches each; guard off; all three bit-identical to each other and to the 10-23 run (653d6379). The race reproduced by the standalone tests did not occur in slorado here. Open: why the standalone pattern triggers it (fresh 8 MiB input memcpy + sync right before each launch, HIP active) and slorado's does not.
