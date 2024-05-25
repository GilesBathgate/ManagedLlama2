#pragma once

#include <cuda_fp16.h>

extern "C" __global__ void vec_mat_kernel(half* output, const half* vector, const half* matrix, const int rows, const int cols,
    const int m_col_stride, const int o_stride, const int m_row_stride);