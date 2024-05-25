#pragma once

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

// This is used for Top-P sampling.
__global__ void softmax_logits_kernel(half* __restrict__ logits, int size, float temperature, int *indices);
