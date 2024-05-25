#include "convert_kernel.cuh"

template <typename A, typename B>
inline __device__ void convert(A* output, const B* input, const int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        output[index] = input[index];
}

__global__ void convert_kernel(float* output, const half* input, const int size)
{
    return convert<float, half>(output, input, size);
}
