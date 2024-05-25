#include "argmax_kernel.cuh"

#include <cub/block/block_reduce.cuh>

using BlockReduce = cub::BlockReduce<float, 1024>;
using BlockStorage = BlockReduce::TempStorage;

template <typename T>
inline __device__ void argmax(const T* input, const int size, int* result, const int pos, const bool write_token)
{
    const int index = blockDim.x + threadIdx.x;

    // find local max value and its position
    float max_val = threadIdx.x < size ? (float)input[threadIdx.x] : -INFINITY;
    int   max_pos = threadIdx.x < size ? threadIdx.x : 0;
    for (int i = index; i < size; i += blockDim.x) {
        if ((float)input[i] > max_val) {
            max_val = input[i];
            max_pos = i;
        }
    }

    // find the global max value
    __shared__ float shared_val;
    __shared__ BlockStorage storage;
    float global_max_val = BlockReduce(storage).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = global_max_val;
    __syncthreads();
    global_max_val = shared_val;

    // possibility of race condition here, so we first write it to shared memory variable and then have just one thread to update the pointers.
    __shared__ int global_max_pos;
    if (max_val == global_max_val)
        global_max_pos = max_pos;
    __syncthreads();

    // write next token to the current token location
    if (threadIdx.x == 0 && write_token)
        result[pos] = global_max_pos;
}

__global__ void argmax_kernel(const half* input, const int size, int* result, const int pos, const bool write_token)
{
    return argmax<half>(input, size, result, pos, write_token);
}