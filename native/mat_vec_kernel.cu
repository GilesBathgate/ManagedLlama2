#include "mat_vec_kernel.cuh"

// Only used for the final linear layer to get logits (for most other layers we use the INT4 version below)
__global__ void mat_vec_kernel(half* op, const half* ip, const half* wt, int n, int d, int numSerialLoads,
    int ip_stride, int w_stride, int op_stride, int w_row_stride, float alpha) {
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;
    const half* __restrict__ input = ip + blockIdx.y * ip_stride;
    const half* __restrict__ weight = wt + blockIdx.y * w_stride;
    half* output = op + blockIdx.y * op_stride;

    float sum = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + threadIdx.x) * 8;
        if (j < n) {
            half w[8];
            half ip[8];
            *((uint4*)(&w)) = loadFromMem((uint4*)(&weight[index * w_row_stride + j]));
            *((uint4*)(&ip)) = *((uint4*)(&input[j]));
            for (int el = 0; el < 8; el++)
                sum += float(w[el]) * float(ip[el]);
        }
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);
    sum *= alpha;

    if (threadIdx.x == 0)
        output[index] = (half)sum;
}

// Simpler version of the above - handles non multiple of 8 dimensions too (used only by MHA block)
__global__ void mat_vec_kernel_simple(half* op, half* ip, half* wt, int n, int numSerialElements,
    int ip_stride, int w_stride, int w_row_stride, float alpha, int* pPos, int kv_mul) {

    int op_stride = *pPos + 1;
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= op_stride)
        return;

    const half* __restrict__ input = ip + blockIdx.y * ip_stride;
    const half* __restrict__ weight = wt + (blockIdx.y / kv_mul) * w_stride;
    half* output = op + blockIdx.y * op_stride;

    float sum = 0;
    for (int i = 0; i < numSerialElements; i++) {
        int j = i * 32 + threadIdx.x;
        if (j < n)
            sum += ((float)weight[index * w_row_stride + j]) * ((float)input[j]);
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);
    sum *= alpha;

    if (threadIdx.x == 0)
        output[index] = (half)sum;
}

// hardcoded for group-count = 128
__forceinline__ __device__ float get_mat_vec_int4(int index, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height) {

    float sum = 0;
    for (int ygq = 0; ygq * 128 + threadIdx.x * 4 < packed_weights_height; ygq++) {   // each iteration of this loop covers 8 x 128 elements in y dimension of weight matrix (weight matrix is column major)
        uint32_t packed_q_z = loadFromMem(&q_zeros[index * packed_zeros_height + ygq]);

        // load weights in one go (32 elements from weight matrix loaded by each thread in one read)
        uint32_t loaded_packed_wts[4];
        *((uint4*)(&loaded_packed_wts[0])) = loadFromMem((uint4*)(&q_weight[index * packed_weights_height + ygq * 128 + threadIdx.x * 4]));

        int group_y = ygq * 8 + (threadIdx.x / 4);
        float q_z = (float)(packed_q_z >> (4 * (threadIdx.x / 4)) & 0xF);
        float scale = (float)loadFromMem(&scales[index * scales_height + group_y]);
        int y_base = ygq * 1024 + threadIdx.x * 32;

        for (int qi = 0; qi < 4; qi++) {                 // each iteration of this loop covers 256 elements in y dimension of weight matrix
            int ys = y_base + qi * 8;
            if (ys < inputElements) {
                uint32_t packed_q_w = loaded_packed_wts[qi];
                half ip[8];
                *((uint4*)(&ip)) = *((uint4*)(&input[ys]));

                for (int i = 0; i < 8; i++) {
                    float q_wt = (float)(packed_q_w & 0xF);
                    float w = (q_wt - q_z) * scale;
                    sum += w * float(ip[i]);
                    packed_q_w = (packed_q_w >> 4);
                }
            }
        }
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    return sum;
}


__device__ void mat_vec_int4(half* __restrict__ output, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, bool accum, int loff, int* pPos)
{
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= opElements)
        return;


    float sum = get_mat_vec_int4(index, input, q_weight, q_zeros, scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height);

    if (threadIdx.x == 0) {
        if (loff != -1) {
            output += loff + (*pPos * opElements);
        }

        if (accum)
            sum += (float)output[index];
        output[index] = (half)sum;
    }
}

__global__ void mat_vec_kernel_int4(half* __restrict__ output, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, bool accum, int loff, int* pPos)
{
    mat_vec_int4(output, input, q_weight, q_zeros, scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height, accum, loff, pPos);
}

__global__ void qkv_matvec_kernel(half* __restrict__ q, half* __restrict__ key_cache, half* __restrict__ value_cache, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ q_scales,
    const uint32_t* __restrict__ k_weight, const uint32_t* __restrict__ k_zeros, const half* __restrict__ k_scales,
    const uint32_t* __restrict__ v_weight, const uint32_t* __restrict__ v_zeros, const half* __restrict__ v_scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, int loff, int* pPos)
{
    if (blockIdx.y == 0)
        mat_vec_int4(q, input, q_weight, q_zeros, q_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height, false, -1, nullptr);
    else if (blockIdx.y == 1)
        mat_vec_int4(key_cache, input, k_weight, k_zeros, k_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height, false, loff, pPos);
    else // if (blockIdx.y == 2)
        mat_vec_int4(value_cache, input, v_weight, v_zeros, v_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height, false, loff, pPos);
}

__global__ void ffn_matvec_silu_kernel(half* __restrict__ output, const half* __restrict__ input,
    const uint32_t* __restrict__ g_weight, const uint32_t* __restrict__ g_zeros, const half* __restrict__ g_scales,
    const uint32_t* __restrict__ u_weight, const uint32_t* __restrict__ u_zeros, const half* __restrict__ u_scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height) {

    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= opElements)
        return;

    float g_val = get_mat_vec_int4(index, input, g_weight, g_zeros, g_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height);
    float u_val = get_mat_vec_int4(index, input, u_weight, u_zeros, u_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height);

    // apply silu and write the result
    if (threadIdx.x == 0) {
        float val = g_val;
        val *= 1.0f / (1.0f + expf(-val));
        val *= u_val;
        output[index] = (half)val;
    }
}
