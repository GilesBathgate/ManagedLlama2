#include "vecadd_kernel.cuh"

__global__ void VecAdd(const float* A, const float* B, float* C, int N)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < N)
    C[i] = A[i] + B[i];
}

#include <assert.h>
#include <stdio.h>

template <typename T>
inline __device__ void memory_alignment(T* base, T* offset_ptr, size_t offset)
{
    T* memory1 = base + offset;
    T* memory2 = offset_ptr;

    if (memory1 != memory2)
        printf("%p != %p\n diff: %lu\n", memory1, memory2, (size_t)(memory1 - memory2));

    assert(memory1 == memory2);
}

__global__ void memory_alignment_int_kernel(int* base, int* offset_ptr, size_t offset)
{
    return memory_alignment<int>(base, offset_ptr, offset);
}

__global__ void memory_alignment_half_kernel(half* base, half* offset_ptr, size_t offset)
{
    return memory_alignment<half>(base, offset_ptr, offset);
}
