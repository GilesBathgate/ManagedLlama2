#pragma once

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

extern "C" __global__ void copy_embedding_kernel(half* x, const half* __restrict__ table, int size, int* tokens, int* pPos);
