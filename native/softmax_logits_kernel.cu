#include "softmax_logits_kernel.cuh"

// This is used for Top-P sampling. We do the following:
// 1. Divide the logits by temperature
// 2. Compute softmax
// 3. Write the indices in an array
__global__ void softmax_logits_kernel(half* __restrict__ logits, int size, float temperature, int *indices) {
    int tid = threadIdx.x;
    int step = blockDim.x;


    for (int t = tid; t < size; t += step)
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
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    // find max value (for numerical stability)
    float max_val = tid < size ? ((float)logits[tid]) : -FLT_MAX;
    for (int i = tid + step; i < size; i += step)
        if ((float)logits[i] > max_val)
            max_val = logits[i];

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        float v = expf(float(logits[i]) - max_val);
        logits[i] = (half)v;
        sum += v;
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        logits[t] = (half)(float(logits[t]) / sum);
}

