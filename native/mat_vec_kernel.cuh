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

extern "C" __global__ void mat_vec_swiglu_kernel(half* output, const half* input,
    const uint32_t* g_weight, const uint32_t* g_zeros, const half* g_scales,
    const uint32_t* u_weight, const uint32_t* u_zeros, const half* u_scales,
    const int rows, const int cols, const int zeros_size, const int scales_size, const int weights_size);
