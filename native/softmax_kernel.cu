#include "softmax_kernel.cuh"

#include <cub/block/block_reduce.cuh>

constexpr int max_seqlen_shared_mem = 8192;

using BlockReduce = cub::BlockReduce<float, 1024>;
using BlockStorage = BlockReduce::TempStorage;

template <typename T>
inline __device__ void softmax(T* vector, const int size)
{
    // load input to shared memory
    __shared__ float buffer[max_seqlen_shared_mem];
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        buffer[i] = (float)vector[blockIdx.x * size + i];
    __syncthreads();

    // find max value (for numerical stability)
    float max_val = threadIdx.x < size ? buffer[threadIdx.x] : 0;
    for (int i = threadIdx.x + blockDim.x; i < size; i += blockDim.x)
        if (buffer[i] > max_val)
            max_val = buffer[i];

    __shared__ float shared;
    __shared__ BlockStorage storage;
    max_val = BlockReduce(storage).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) shared = max_val;
    __syncthreads();
    max_val = shared;

    // exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        buffer[i] = expf(buffer[i] - max_val);
        sum += buffer[i];
    }

    sum = BlockReduce(storage).Sum(sum);
    if (threadIdx.x == 0) shared = sum;
    __syncthreads();
    sum = shared;

    // normalize and write the result
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        vector[blockIdx.x * size + i] = (T)(buffer[i] / sum);
}

__global__ void softmax_kernel(half* vector, const int size)
{
    return softmax<half>(vector, size);
}
