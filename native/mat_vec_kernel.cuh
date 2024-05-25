#pragma once

#include <cstdint>
#include <cuda_fp16.h>

extern "C" __global__ void mat_vec_kernel(half* output, const half* vector, const half* matrix, const int rows, const int cols);

extern "C" __global__ void mat_vec_strided_kernel(half* output, const half* vector, const half* matrix, const int rows, const int cols,
    const int v_stride, const int m_col_stride, const int m_row_stride, const int o_stride, const float alpha);

extern "C" __global__ void mat_vec_residual_int4_kernel(half* output, const half* input, const uint32_t* weights, const uint32_t* zeros, const half* scales,
    const int rows, const int cols, const int zeros_size, const int scales_size, const int weights_size);

extern "C" __global__ void qkv_mat_vec_kernel(half* q_output, half* k_output, half* v_output, const half* input,
    const uint32_t* q_weight, const uint32_t* q_zeros, const half*  q_scales,
    const uint32_t* k_weight, const uint32_t* k_zeros, const half*  k_scales,
    const uint32_t* v_weight, const uint32_t* v_zeros, const half*  v_scales,
    const int rows, const int cols, const int zeros_size, const int scales_size, const int weights_size);

extern "C" __global__ void ffn_matvec_silu_kernel(half* __restrict__ output, const half* __restrict__ input,
    const uint32_t* __restrict__ g_weight, const uint32_t* __restrict__ g_zeros, const half* __restrict__ g_scales,
    const uint32_t* __restrict__ u_weight, const uint32_t* __restrict__ u_zeros, const half* __restrict__ u_scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height);
