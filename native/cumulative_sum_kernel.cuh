#pragma once

#include <cuda_fp16.h>

extern "C" __global__ void cumulative_sum_kernel(half* values, const int size);
