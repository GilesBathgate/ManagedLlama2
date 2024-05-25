#pragma once

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

// find the index in the array that crosses top-p threshold
extern "C" __global__ void sample_top_p_kernel(half* sorted_logits_prefix_sum, int* indices, int n, float top_p_threshold, int* result, volatile int* pPos, int* pPosGpu);
