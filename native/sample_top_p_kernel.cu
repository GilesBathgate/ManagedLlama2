#include "sample_top_p_kernel.cuh"

// find the index in the array that crosses top-p threshold
__global__ void sample_top_p_kernel(half* sorted_logits_prefix_sum, int* indices, int n, float top_p_threshold, int* result, volatile int* pPos, int* pPosGpu)
{
    int tid = threadIdx.x;
    int step = blockDim.x;

    int min_index = n - 1;

    for (int t = tid; t < n; t += step) {
        if ((float)(sorted_logits_prefix_sum[t]) >= top_p_threshold) {
            if (t < min_index) {
                min_index = t;
            }
        }
    }

    // find the min across the block
    using BlockReduce = cub::BlockReduce<int, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    int min_index_global = BlockReduce(temp).Reduce(min_index, cub::Min());
    if (threadIdx.x == 0)
    {
        int token_pos = *pPos;
        token_pos++;
        result[token_pos] = indices[min_index_global];

        // update the token indices
        *pPos = token_pos;
        *pPosGpu = token_pos;
    }
}
