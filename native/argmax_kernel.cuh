#pragma once

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

__global__ void argmax_kernel(half* __restrict__ x, int size, int* result, volatile int* pPos, int* pPosGpu, bool write_token);
