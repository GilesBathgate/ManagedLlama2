#pragma once

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
extern "C" {
__global__ void rmsnorm_kernel(half* o, half* x, half* weight, int size, int elementsPerThread);
}
