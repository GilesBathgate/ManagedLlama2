#pragma once

#include <cuda_fp16.h>

extern "C" __global__ void sort_kernel(half* keys, int* values, const int size);