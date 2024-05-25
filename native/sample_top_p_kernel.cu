#include "sample_top_p_kernel.cuh"

#include <cub/block/block_reduce.cuh>

using BlockReduce = cub::BlockReduce<float, 1024>;
using BlockStorage = BlockReduce::TempStorage;

template <typename T>
inline __device__ void sample_top_p(T* logits, int* indices, const int size, const float threshold, int* result, const int pos)
{
    int min_index = size - 1;

    for (int t = threadIdx.x; t < size; t += blockDim.x)
        if ((float)(logits[t]) >= threshold && t < min_index)
                min_index = t;

    __shared__ BlockStorage storage;
    const int min_index_global = BlockReduce(storage).Reduce(min_index, cub::Min());

    if (threadIdx.x == 0)
        result[pos] = indices[min_index_global];
}

__global__ void sample_top_p_kernel(half* logits, int* indices, const int size, const float threshold, int* result, const int pos)
{
    return sample_top_p<half>(logits, indices, size, threshold, result, pos);
}