#ifndef NN_HIP_SHARED_H
#define NN_HIP_SHARED_H

// iGPU-side helpers for the zero-copy iGPU <-> NPU buffers in nn_npu.cpp (rocm=1 npu=1 builds); implemented in
// nn_hip.c so that nn_npu.cpp stays a plain XRT translation unit.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// hipMalloc `bytes` (rounded up to 4 KiB) and export the allocation as a dma-buf fd. Returns 0 on success.
int hip_shared_alloc(uint64_t bytes, void **dptr, int *dmabuf_fd);

// Wait for all prior iGPU work, copy (any direction, hipMemcpyDefault), wait again.
void hip_copy_sync(void *dst, const void *src, uint64_t bytes);

// hipMemset, then wait.
void hip_memset_sync(void *dst, int value, uint64_t bytes);

// Asynchronous variants on the default stream (ordered after the caller's iGPU work); hip_sync() waits for all of it.
void hip_sync(void);
void hip_copy_async(void *dst, const void *src, uint64_t bytes);
void hip_copy2d_async(void *dst, uint64_t dpitch, const void *src, uint64_t spitch, uint64_t width, uint64_t height);
void hip_memset_async(void *dst, int value, uint64_t bytes);

#ifdef __cplusplus
}
#endif

#endif  // NN_HIP_SHARED_H
