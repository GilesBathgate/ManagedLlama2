#pragma once

#include <cuda_fp16.h>

extern "C" __global__ void convert_kernel(float* output, const half* input, const int size);
