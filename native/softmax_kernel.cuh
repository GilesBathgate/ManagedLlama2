#pragma once

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include "common.cuh"

extern "C" __global__ void softmax_kernel(half* __restrict__ arr, int num_heads, int* pPos);
