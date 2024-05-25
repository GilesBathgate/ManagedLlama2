#include "vec_mat_kernel.cuh"

// Here we make use of shared memory to achieve better memory access pattern, and transpose a 32x32 chunk of the matrix on the fly
// Again used only by the MHA block
__global__ void vec_mat_kernel(half* op, const half* __restrict__ ip, const half* __restrict__ wt, int N, int* pPos, int w_stride, int op_stride, int w_row_stride, int kv_mul) {
    int K = *pPos + 1;
    const half* __restrict__ input = ip + blockIdx.y * K;
    const half* __restrict__ weight = wt + (blockIdx.y / kv_mul) * w_stride;
    half* output = op + blockIdx.y * op_stride;

    int start_n = blockIdx.x * 32;
    int i = start_n + threadIdx.y;

    // 2x for double buffering
    // +2 to avoid shared memory bank conflicts
    __shared__ half loaded_fragment[2][32][32 + 2];

    // OOB check
    if (i >= N)
        return;

    // load the first 32x32 fragment
    int n = start_n + threadIdx.x;
    int k = threadIdx.y;
    int offset = k * w_row_stride + n;
    loaded_fragment[0][threadIdx.y][threadIdx.x] = ((n < N) && (k < K)) ? weight[offset] : (half)0.0;

    float sum = 0;
    // Loop over the matrix row and vector elements
    for (int e = 0; ;) {
        __syncthreads();    // wait for the load

        int start_k = e * 32;
        if (start_k >= K) break;
        k = start_k + threadIdx.x;
        int buf_i = e & 1;
        sum += float(loaded_fragment[buf_i][threadIdx.x][threadIdx.y]) * ((k < K) ? (float)input[k] : 0.0f);

        // load for the next iteration
        e++;
        start_k = e * 32;
        buf_i = e & 1;
        n = start_n + threadIdx.x;
        k = start_k + threadIdx.y;
        int offset = k * w_row_stride + n;
        loaded_fragment[buf_i][threadIdx.y][threadIdx.x] = ((n < N) && (k < K)) ? weight[offset] : (half)0.0;
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    if (threadIdx.x == 0)
        output[i] = (half)sum;
}
