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

### 2026-10-24 — zero-copy rms_qkv_rope: NPU outputs never visible to the iGPU
- Setup: shared buffers = iGPU allocation exported as dma-buf, imported with `xrt::bo(device, pid, fd)`, bound as args 0 (x) and 2/8/11/12 (normed, v, q_roped, k_roped) of the 13-arg `rms_gemms_rope` ELF; args 4/6 (q, k intermediates) stay host BOs; LUTs and weights host BOs.
- Symptom: the guard's 0xFF marker stays in all four outputs for the full 50 ms on every launch (run crawls); with OPENFISH_NPU_GUARD=0 the run completes but basecalls are wrong (3 reads: 0/3 exact).
- Not the cause: the strided q|k|v copies (replaced by contiguous copies, same result); iGPU caching (uncached allocation, same result).
- Same mechanism works for the GEMM ELFs (bit-identical over 2,304+ launches per run), silu_mul and axpy (bit-identical).
- Next: standalone repro with this ELF (zero_copy_test-style: import one output at a time to find which arg breaks; try a larger/aligned buffer; check XRT logs for the patching of imported BOs).

### 2026-10-24 (cont.) — zero-copy rms_qkv_rope: root cause = imported iGPU buffers under 2 MiB
`npu/test/zero_copy_rqr_test.cpp` (logs `~/p0/zero_copy_rqr_test{,_b,_c,_d}.log`), rms_qkv_rope ELF, reference = host BOs:
- hipMalloc does not sub-allocate (every allocation 256 KiB–64 MiB is its own, offset 0): not a dma-buf range problem.
- One imported iGPU buffer (any of x, normed, v, q_roped, k_roped; 512 KiB or 64 MiB; cached or uncached): all outputs correct.
- Two or more imported buffers of 512 KiB or 1 MiB: every imported output wrong (never written: still the 0xFF marker).
  Fresh `xrt::run` or `set_arg` re-binding behave the same.
- The same combinations with allocations of 2, 4 or 64 MiB: all correct.
- Fix: `hip_shared_alloc` rounds sizes up to 2 MiB (openfish `round zero-copy shared allocations up to 2 MiB`). 3 reads:
  zero-copy rms_qkv_rope bit-identical to the copy path, 18.7 → 13.8 s, guard 0 waits. The GEMM/silu_mul/axpy buffers
  (4–64 MiB) were never affected; rope_d64 (1 MiB) would have been.
- Likely a 2 MiB mapping granularity in the import of iGPU (VRAM) dma-bufs into the NPU's address space (driver side);
  not isolated further. Worth reporting upstream (amdxdna / XRT) with this test.
