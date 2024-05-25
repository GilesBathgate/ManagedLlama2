#include "rmsnorm_kernel.cuh"

#include "common.cuh"
#include <cub/block/block_reduce.cuh>

using BlockReduce = cub::BlockReduce<float, 1024>;
using BlockStorage = BlockReduce::TempStorage;

template <typename T>
inline __device__ void rmsnorm(T* output, const T* input, const T* gamma, const int size, const float eps)
{
    float local_variance = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        const int index = blockIdx.x * size + i;
        const float value = (float)input[index];
        local_variance += (value * value);
    }

    __shared__ BlockStorage storage;
    float variance = BlockReduce(storage).Sum(local_variance);

    __shared__ float rms;
    if (threadIdx.x == 0)
    {
        variance /= size;
        rms = rsqrtf(variance + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        const int index = blockIdx.x * size + i;
        const float value = (float)input[index];
        const float weight = (float)gamma[index];
        output[index] = (value * rms) * weight;
    }
}

__global__ void rmsnorm_kernel(half* output, const half* input, const half* gamma, const int size, const float eps)
{
    return rmsnorm<half>(output, input, gamma, size, eps);
}
