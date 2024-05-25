#include "argmax_kernel.cuh"

__global__ void argmax_kernel(half* __restrict__ x, int size, int* result, volatile int* pPos, int* pPosGpu, bool write_token) {
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    int tid = threadIdx.x;
    int step = blockDim.x;

    // find local max value and its position
    float max_val = tid < size ? (float)x[tid] : -INFINITY;
    int   max_pos = tid < size ? tid : 0;
    for (int i = tid + step; i < size; i += step) {
        if ((float)x[i] > max_val) {
            max_val = x[i];
            max_pos = i;
        }
    }

    // find the global max value
    float global_max_val;
    global_max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = global_max_val;
    __syncthreads();
    global_max_val = shared_val;

    // possibility of race condition here, so we first write it to shared memory variable and then have just one thread to update the pointers.
    __shared__ int global_max_pos;
    if (max_val == global_max_val) {
        global_max_pos = max_pos;
    }
    __syncthreads();

    // write next token to the current token location
    if (threadIdx.x == 0) {
        int token_pos = *pPos;
        token_pos++;

        if (write_token)
            result[token_pos] = global_max_pos;

        // update the token indices (unblocks the CPU)
        *pPos = token_pos;
        *pPosGpu = token_pos;
    }
}
