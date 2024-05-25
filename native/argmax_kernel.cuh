#pragma once

#include <cuda_fp16.h>

extern "C" __global__ void argmax_kernel(const half* input, const int size, int* result, const int pos, const bool write_token);
