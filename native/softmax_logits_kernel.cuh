#pragma once

#include <cuda_fp16.h>

extern "C" __global__ void softmax_logits_kernel(half* logits, const int size, const float temperature, int *indices);
