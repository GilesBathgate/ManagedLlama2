#pragma once

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

constexpr int MAX_SEQ_LEN_SMEM_KERNEL = 8192; // 8k is the max sequence length supported by the kernel that uses shared memory

// utility function to load from memory (try different cache hints)
#define USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS 1
#define USE_LDCS_FOR_WEIGHT_LOADS 0

__forceinline__ __device__ uint4 loadFromMem(const uint4* ptr) {
    uint4 ret;
#if USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS
    asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
#elif USE_LDCS_FOR_WEIGHT_LOADS
    ret = __ldcs(ptr);
#else
    ret = *ptr;
#endif
    return ret;
}

__forceinline__ __device__ uint32_t loadFromMem(const uint32_t* ptr) {
    uint32_t ret;
#if USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS
    asm volatile("ld.global.L1::no_allocate.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#elif USE_LDCS_FOR_WEIGHT_LOADS
    ret = __ldcs(ptr);
#else
    ret = *ptr;
#endif
    return ret;
}

__forceinline__ __device__ half loadFromMem(const half* ptr) {
    half ret;
#if USE_NO_CACHE_ALLOCATE_FOR_WEIGHT_LOADS
    uint16_t temp;
    asm volatile("ld.global.L1::no_allocate.u16 %0, [%1];" : "=h"(temp) : "l"(ptr));
    ret = __ushort_as_half(temp);
#elif USE_LDCS_FOR_WEIGHT_LOADS
    ret = __ldcs(ptr);
#else
    ret = *ptr;
#endif
    return ret;
}

inline int div_ceil(int a, int b) {
    return (a - 1) / b + 1;
}
