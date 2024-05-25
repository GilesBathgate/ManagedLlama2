#include "mat_vec_kernel.cuh"

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

template <typename T>
inline __device__ void mat_vec_residual_int4(T* output, const T* input, const uint32_t* weights, const uint32_t* zeros, const half* scales,
    const int rows, const int cols, const int zeros_size, const int scales_size, const int weights_size)
{
    float sum = sum_mat_vec_int4<T>(input, weights, zeros, scales, rows, cols, zeros_size, scales_size, weights_size);

    if (threadIdx.x == 0) {
        const int col_index = blockIdx.x * blockDim.y + threadIdx.y;
        if (col_index >= cols) return;

        sum += (float)output[col_index]; // Residual accumulation

        output[col_index] = (T)sum;
    }
}

__global__ void mat_vec_residual_int4_kernel(half* output, const half* input, const uint32_t* weights, const uint32_t* zeros, const half* scales,
    const int rows, const int cols, const int zeros_size, const int scales_size, const int weights_size)
{
    return mat_vec_residual_int4<half>(output, input, weights, zeros, scales, rows, cols, zeros_size, scales_size, weights_size);
}

template <typename T>
inline __device__ void mat_vec_int4(T* output, const T* input, const uint32_t* weights, const uint32_t* zeros, const half* scales,
    const int rows, const int cols, const int zeros_size, const int scales_size, const int weights_size)
{
    float sum = sum_mat_vec_int4<T>(input, weights, zeros, scales, rows, cols, zeros_size, scales_size, weights_size);

    if (threadIdx.x == 0) {
        const int col_index = blockIdx.x * blockDim.y + threadIdx.y;
        if (col_index >= cols) return;
        output[col_index] = (T)sum;
    }
}

template <typename T>
inline __device__ void qkv_mat_vec(T*  q_output, T*  k_output, T*  v_output, const T*  input,
    const uint32_t* q_weight, const uint32_t* q_zeros, const half*  q_scales,
    const uint32_t* k_weight, const uint32_t* k_zeros, const half*  k_scales,
    const uint32_t* v_weight, const uint32_t* v_zeros, const half*  v_scales,
    const int rows, const int cols, int zeros_size, int scales_size, int weights_size)
{
    if (blockIdx.y == 0)
        mat_vec_int4<T>(q_output, input, q_weight, q_zeros, q_scales, rows, cols, zeros_size, scales_size, weights_size);
    else if (blockIdx.y == 1)
        mat_vec_int4<T>(k_output, input, k_weight, k_zeros, k_scales, rows, cols, zeros_size, scales_size, weights_size);
    else // if (blockIdx.y == 2)
        mat_vec_int4<T>(v_output, input, v_weight, v_zeros, v_scales, rows, cols, zeros_size, scales_size, weights_size);
}

__global__ void qkv_mat_vec_kernel(half* q_output, half* k_output, half* v_output, const half* input,
    const uint32_t* q_weight, const uint32_t* q_zeros, const half*  q_scales,
    const uint32_t* k_weight, const uint32_t* k_zeros, const half*  k_scales,
    const uint32_t* v_weight, const uint32_t* v_zeros, const half*  v_scales,
    const int rows, const int cols, const int zeros_size, const int scales_size, const int weights_size)
{
    return qkv_mat_vec<half>(q_output, k_output, v_output, input,
                             q_weight, q_zeros, q_scales, k_weight, k_zeros, k_scales, v_weight, v_zeros, v_scales,
                             rows, cols, zeros_size, scales_size, weights_size);
}


template <typename T>
inline __device__ void mat_vec_swiglu(T* output, const T* input,
    const uint32_t* g_weight, const uint32_t* g_zeros, const half* g_scales,
    const uint32_t* u_weight, const uint32_t* u_zeros, const half* u_scales,
    const int rows, const int cols, int zeros_size, int scales_size, int weights_size)
{
    float g_val = sum_mat_vec_int4<T>(input, g_weight, g_zeros, g_scales, rows, cols, zeros_size, scales_size, weights_size);
    float u_val = sum_mat_vec_int4<T>(input, u_weight, u_zeros, u_scales, rows, cols, zeros_size, scales_size, weights_size);

    if (threadIdx.x == 0) {
        const int col_index = blockIdx.x * blockDim.y + threadIdx.y;
        if (col_index >= cols) return;

        float val = g_val;
        val *= 1.0f / (1.0f + expf(-val));
        val *= u_val;
        output[col_index] = (T)val;
    }
}

__global__ void mat_vec_swiglu_kernel(half* output, const half* input,
    const uint32_t* g_weight, const uint32_t* g_zeros, const half* g_scales,
    const uint32_t* u_weight, const uint32_t* u_zeros, const half* u_scales,
    const int rows, const int cols, const int zeros_size, const int scales_size, const int weights_size)
{
    return mat_vec_swiglu<half>(output, input,
                                       g_weight, g_zeros, g_scales, u_weight, u_zeros, u_scales,
                                       rows, cols, zeros_size, scales_size, weights_size);
}
