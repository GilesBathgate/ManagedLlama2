#include "sort_kernel.cuh"

#include <cub/block/block_radix_sort.cuh>

constexpr int items = 125;

using BlockRadixSort = cub::BlockRadixSort<half, 256, items, int>;
using SortStorage = BlockRadixSort::TempStorage;

__device__ SortStorage storage;

template <typename K, typename V>
inline __device__ void sort(K* keys, V* values, const int size)
{
    const int index = blockIdx.x * blockDim.x * items + threadIdx.x * items;

    if (index >= size) return;

    K local_keys[items];
    V local_values[items];

    for(int i = 0; i < items; ++i)
    {
        const int j = index + i;
        local_keys[i] = keys[j];
        local_values[i] = values[j];
    }

    BlockRadixSort(storage).SortDescending(local_keys, local_values);
    __syncthreads();

    for(int i = 0; i < items; ++i)
    {
        const int j = index + i;
        keys[j] = local_keys[i];
        values[j] = local_values[i];
    }
}

__global__ void sort_kernel(half* keys, int* values, const int size)
{
    return sort<half, int>(keys, values, size);
}