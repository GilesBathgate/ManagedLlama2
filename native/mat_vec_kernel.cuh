#pragma once

#include <cstdint>
#include <cuda_fp16.h>

extern "C" __global__ void mat_vec_kernel(half* output, const half* vector, const half* matrix, const int rows, const int cols);

extern "C" __global__ void mat_vec_strided_kernel(half* output, const half* vector, const half* matrix, const int rows, const int cols,
    const int v_stride, const int m_col_stride, const int m_row_stride, const int o_stride, const float alpha);

extern "C" __global__ void mat_vec_residual_int4_kernel(half* output, const half* input, const uint32_t* weights, const uint32_t* zeros, const half* scales,
    const int rows, const int cols, const int zeros_size, const int scales_size, const int weights_size);

extern "C" __global__ void qkv_matvec_kernel(half* __restrict__ q, half* __restrict__ key_cache, half* __restrict__ value_cache, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ q_scales,
    const uint32_t* __restrict__ k_weight, const uint32_t* __restrict__ k_zeros, const half* __restrict__ k_scales,
    const uint32_t* __restrict__ v_weight, const uint32_t* __restrict__ v_zeros, const half* __restrict__ v_scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, int loff, int* pPos);

extern "C" __global__ void ffn_matvec_silu_kernel(half* __restrict__ output, const half* __restrict__ input,
    const uint32_t* __restrict__ g_weight, const uint32_t* __restrict__ g_zeros, const half* __restrict__ g_scales,
    const uint32_t* __restrict__ u_weight, const uint32_t* __restrict__ u_zeros, const half* __restrict__ u_scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height);
