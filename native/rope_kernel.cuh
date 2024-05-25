#pragma once

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

// Each block processes a single head
extern "C" __global__ void RoPERotation_kernel(half* sq, half* sk_base, int num_kv_heads, int head_size, int* pPos, int loff, float rope_theta);
