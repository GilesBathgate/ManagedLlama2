#pragma once

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

__global__ void convert_fp16_to_fp32(float* out, half* in, int elements);
