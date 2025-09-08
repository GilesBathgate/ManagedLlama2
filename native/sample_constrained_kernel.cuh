#pragma once

#include <cuda_fp16.h>

extern "C" __global__ void sample_constrained_kernel(half* logits, const int size, const int* constraint, const int constraint_size);
