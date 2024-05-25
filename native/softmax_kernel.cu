#include "softmax_kernel.cuh"

__global__ void softmax_kernel(half* __restrict__ arr, int num_heads, int* pPos) {
    __shared__ float att[MAX_SEQ_LEN_SMEM_KERNEL];
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    int size = *pPos + 1;

    // load input to shared memory
    for (int t = tid; t < size; t += step)
        att[t] = (float)arr[h * size + t];
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    // find max value (for numerical stability)
    float max_val = tid < size ? att[tid] : 0;
    for (int i = tid + step; i < size; i += step)
        if (att[i] > max_val)
            max_val = att[i];

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        att[i] = expf(att[i] - max_val);
        sum += att[i];
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        arr[h * size + t] = (half)(att[t] / sum);
}
