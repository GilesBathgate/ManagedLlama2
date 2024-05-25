#pragma once

#include <cuda_fp16.h>

// Each block processes a single head
extern "C" __global__ void rope_kernel(half* sq, half* sk_base, int num_kv_heads, int head_size, int pos, float rope_theta);
