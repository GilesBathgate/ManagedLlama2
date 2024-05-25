#pragma once

#include <cuda_fp16.h>

extern "C" __global__ void rmsnorm_kernel(half* output, const half* input, const half* gamma, const int size, const float eps);
