#include <cuda_fp16.h>

extern "C"  {
__global__ void VecAdd(const float* A, const float* B, float* C, int N);

__global__ void memory_alignment_int_kernel(int* base, int* offset_ptr, size_t offset);

__global__ void memory_alignment_half_kernel(half* base, half* offset_ptr, size_t offset);
}
