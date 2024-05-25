#pragma once

#include <cuda_fp16.h>

extern "C" __global__ void sample_top_p_kernel(half* logits, int* indices, const int size, const float threshold, int* result, const int pos);
