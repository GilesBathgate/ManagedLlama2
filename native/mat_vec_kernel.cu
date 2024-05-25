#include "mat_vec_kernel.cuh"

#include "common.cuh"
#include <cub/warp/warp_reduce.cuh>

using WarpReduce = cub::WarpReduce<float>;
using WarpStorage = WarpReduce::TempStorage;

template <typename T>
inline __device__ void mat_vec(T* output, const T* vector, const T* matrix, const int rows, const int cols)
{
    const int col_index = blockIdx.x * blockDim.y + threadIdx.y;

    if (col_index >= cols) return;

    const int index = col_index * rows + blockIdx.y;

    float sum = 0.0f;
    for (int row_index = threadIdx.x; row_index < rows; row_index += blockDim.x) {
        const int mat_index = index + row_index;
        const int vec_index = row_index + blockIdx.y;
        sum += (float)matrix[mat_index] * (float)vector[vec_index];
    }

    __shared__ WarpStorage storage;
    sum = WarpReduce(storage).Sum(sum);

    if (threadIdx.x == 0)
        output[col_index + blockIdx.y] = (T)sum;
}

__global__ void mat_vec_kernel(half* output, const half* vector, const half* matrix, const int rows, const int cols)
{
    return mat_vec<half>(output, vector, matrix, rows, cols);
}

template <typename T>
inline __device__ void mat_vec_strided(T* output, const T* vector, const T* matrix, const int rows, const int cols,
    const int v_stride, const int m_col_stride, const int m_row_stride, const int o_stride, const float alpha)
{
    const int col_index = blockIdx.x * blockDim.y + threadIdx.y;

    if (col_index >= cols) return;

    const int index = col_index * m_row_stride + blockIdx.y * m_col_stride;

    float sum = 0.0f;
    for (int row_index = threadIdx.x; row_index < rows; row_index += blockDim.x) {
        const int mat_index = index + row_index;
        const int vec_index = row_index + blockIdx.y * v_stride;
        sum += (float)matrix[mat_index] * (float)vector[vec_index];
    }

    __shared__ WarpStorage storage;
    sum = WarpReduce(storage).Sum(sum);

    if (threadIdx.x == 0) {
        sum *= alpha;
        output[col_index + blockIdx.y * o_stride] = (T)sum;
    }
}

__global__ void mat_vec_strided_kernel(half* output, const half* vector, const half* matrix, const int rows, const int cols,
    const int v_stride, const int m_col_stride, const int m_row_stride, const int o_stride, const float alpha)
{
    return mat_vec_strided<half>(output, vector, matrix, rows, cols, v_stride, m_col_stride, m_row_stride, o_stride, alpha);
}

template <typename T, typename Block = uint4>
__forceinline__ __device__ void load_mem(T* dest, const T* src) {
    *((Block*)dest) = __ldcs((Block*)src);
}

template <typename T>
inline __device__ float sum_mat_vec_int4(const T* input, const uint32_t* weights, const uint32_t* zeros, const half* scales,
    const int rows, const int cols, const int zeros_size, const int scales_size, const int weights_size)
{
    const int col_index = blockIdx.x * blockDim.y + threadIdx.y;

    if (col_index >= cols) return;

    constexpr int n_bits = 4;
    constexpr int i_size = 8;
    constexpr int w_size = 4;
    constexpr int z_size = 32;
    constexpr uint32_t mask = 0xF;
    constexpr int group_size = 128;

    const int z = threadIdx.x * z_size;
    const int s = threadIdx.x / w_size;
    const int w = threadIdx.x * w_size;

    const int z_stride = col_index * zeros_size;
    const int s_stride = col_index * scales_size;
    const int w_stride = col_index * weights_size;

    float sum = 0;
    for (int g = 0; g * group_size + w < weights_size; ++g) {
        const int g_stride = g * group_size;

        const int z_index = z_stride + g;
        const float zero = (float)(zeros[z_index] >> (s * n_bits) & mask);

        const int s_index = s_stride + g * i_size + s;
        const float scale = scales[s_index];

        const int w_index = w_stride + g_stride + w;
        uint32_t packed_weights[4];
        load_mem(packed_weights, &weights[w_index]);

        const int index = g_stride * i_size + z;
        for (int q = 0; q < w_size; ++q) {
            const int row_index = index + q * i_size;
            if (row_index >= rows) continue;

            uint32_t weight = packed_weights[q];

            T ip[8];
            load_mem(ip, &input[row_index]);

            for (int i = 0; i < i_size; ++i) {
                const int element = (int)(weight & mask);
                sum += (float)ip[i] * (element - zero) * scale;
                weight >>= n_bits;
            }
        }
    }

    __shared__ WarpStorage storage;
    sum = WarpReduce(storage).Sum(sum);

    return sum;
}

__global__ void mat_vec_residual_int4_kernel(half* output, const half* input, const uint32_t* weights, const uint32_t* zeros, const half* scales,
    const int rows, const int cols, const int zeros_size, const int scales_size, const int weights_size)
{
    float sum = sum_mat_vec_int4<half>(input, weights, zeros, scales, rows, cols, zeros_size, scales_size, weights_size);

    if (threadIdx.x == 0) {
        const int col_index = blockIdx.x * blockDim.y + threadIdx.y;
        if (col_index >= cols) return;

        sum += (float)output[col_index]; // Residual accumulation

        output[col_index] = (half)sum;
    }
}

__device__ void mat_vec_int4(half* __restrict__ output, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, bool accum, int loff, int* pPos)
{
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= opElements)
        return;


    float sum = sum_mat_vec_int4(input, q_weight, q_zeros, scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height);

    if (threadIdx.x == 0) {
        if (loff != -1) {
            output += loff + (*pPos * opElements);
        }

        if (accum)
            sum += (float)output[index];
        output[index] = (half)sum;
    }
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

    float g_val = sum_mat_vec_int4(input, g_weight, g_zeros, g_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height);
    float u_val = sum_mat_vec_int4(input, u_weight, u_zeros, u_scales, inputElements, opElements, packed_zeros_height, scales_height, packed_weights_height);

    // apply silu and write the result
    if (threadIdx.x == 0) {
        float val = g_val;
        val *= 1.0f / (1.0f + expf(-val));
        val *= u_val;
        output[index] = (half)val;
    }
}
