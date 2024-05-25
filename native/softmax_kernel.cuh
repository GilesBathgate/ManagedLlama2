#pragma once

#include <cuda_fp16.h>

extern "C" __global__ void softmax_kernel(half* vector, const int size);
