#pragma once

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include "common.cuh"

// Only used for the final linear layer to get logits (for most other layers we use the INT4 version below)
__global__ void mat_vec_kernel(half* op, const half* ip, const half* wt, int n, int d, int numSerialLoads,
    int ip_stride, int w_stride, int op_stride, int w_row_stride, float alpha);

// Simpler version of the above - handles non multiple of 8 dimensions too (used only by MHA block)
__global__ void mat_vec_kernel_simple(half* op, half* ip, half* wt, int n, int numSerialElements,
    int ip_stride, int w_stride, int w_row_stride, float alpha, int* pPos, int kv_mul);

// hardcoded for group-count = 128
__forceinline__ __device__ float get_mat_vec_int4(int index, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height);

__device__ void mat_vec_int4(half* __restrict__ output, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, bool accum, int loff, int* pPos);

__global__ void mat_vec_kernel_int4(half* __restrict__ output, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, bool accum, int loff, int* pPos);

__global__ void qkv_matvec_kernel(half* __restrict__ q, half* __restrict__ key_cache, half* __restrict__ value_cache, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ q_scales,
    const uint32_t* __restrict__ k_weight, const uint32_t* __restrict__ k_zeros, const half* __restrict__ k_scales,
    const uint32_t* __restrict__ v_weight, const uint32_t* __restrict__ v_zeros, const half* __restrict__ v_scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height, int loff, int* pPos);

__global__ void ffn_matvec_silu_kernel(half* __restrict__ output, const half* __restrict__ input,
    const uint32_t* __restrict__ g_weight, const uint32_t* __restrict__ g_zeros, const half* __restrict__ g_scales,
    const uint32_t* __restrict__ u_weight, const uint32_t* __restrict__ u_zeros, const half* __restrict__ u_scales,
    int inputElements, int opElements, int packed_zeros_height, int scales_height, int packed_weights_height);
