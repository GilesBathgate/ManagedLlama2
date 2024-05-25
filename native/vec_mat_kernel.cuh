#pragma once

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

// Here we make use of shared memory to achieve better memory access pattern, and transpose a 32x32 chunk of the matrix on the fly
// Again used only by the MHA block
__global__ void vec_mat_kernel(half* op, const half* __restrict__ ip, const half* __restrict__ wt, int N, int* pPos, int w_stride, int op_stride, int w_row_stride, int kv_mul);
