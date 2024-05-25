#include "convert_kernel.cuh"

__global__ void convert_fp16_to_fp32(float* out, half* in, int elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        out[index] = (float)in[index];
}
