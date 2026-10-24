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
