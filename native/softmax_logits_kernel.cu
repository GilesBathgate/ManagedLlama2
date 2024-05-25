#include "softmax_logits_kernel.cuh"

#include <cub/block/block_reduce.cuh>

using BlockReduce = cub::BlockReduce<float, 1024>;
using BlockStorage = BlockReduce::TempStorage;

template <typename T>
inline __device__ void softmax_logits(T* logits, const int size, const float temperature, int *indices)
{
    for (int t = threadIdx.x; t < size; t += blockDim.x)
    {
        // first just write the indices array
        indices[t] = t;

        // divide by temperature
        float val = (float)logits[t];
        val /= temperature;
        logits[t] = (half)val;
    }
    __syncthreads();

    // Compute the softmax
    __shared__ float shared_val;
    __shared__ BlockStorage storage;
    // find max value (for numerical stability)
    float max_val = threadIdx.x < size ? ((float)logits[threadIdx.x]) : -FLT_MAX;
    for (int i = threadIdx.x + blockDim.x; i < size; i += blockDim.x)
        if ((float)logits[i] > max_val)
            max_val = logits[i];

    max_val = BlockReduce(storage).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float v = expf(float(logits[i]) - max_val);
        logits[i] = (half)v;
        sum += v;
    }

    sum = BlockReduce(storage).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    for (int t = threadIdx.x; t < size; t += blockDim.x)
        logits[t] = (half)(float(logits[t]) / sum);
}

__global__ void softmax_logits_kernel(half* logits, const int size, const float temperature, int *indices)
{
    return softmax_logits<half>(logits, size, temperature, indices);
}