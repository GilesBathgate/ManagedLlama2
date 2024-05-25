#pragma once

#include <cuda_fp16.h>

extern "C" __global__ void embedding_kernel(half* output, const half* table, const int size, const int* tokens, const int pos);
