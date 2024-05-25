#include "cumulative_sum_kernel.cuh"

#include <cub/block/block_scan.cuh>

using BlockScan = cub::BlockScan<float, 1024>;
using ScanStorage = BlockScan::TempStorage;

template <typename T>
inline __device__ void cumulative_sum(T* values, const int size)
{
    constexpr int items = 32;

    const int index = blockIdx.x * blockDim.x * items + threadIdx.x * items;

    if (index >= size) return;

    float local_values[items];
    for(int i = 0; i < items; ++i)
        local_values[i] = (float)values[index + i];

    __shared__ ScanStorage storage;
    BlockScan(storage).InclusiveSum(local_values, local_values);
    __syncthreads();

    for(int i = 0; i < items; ++i)
        values[index + i] = (T)local_values[i];
}

__global__ void cumulative_sum_kernel(half* values, const int size)
{
    return cumulative_sum<half>(values, size);
}