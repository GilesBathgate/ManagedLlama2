#include "common.cuh"
#include "rmsnorm_kernel.cuh"

// Single block - not enough parallelism for the GPU, but it's just 1% of total time
__global__ void rmsnorm_kernel(half* o, half* x, half* weight, int size, int elementsPerThread) {
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size) {
            float val = (float)x[index];
            ss += val * val;
        }
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss);

    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        shared_ss = rsqrtf(ss / size + 1e-5f) ;
    }
    __syncthreads();
    ss = shared_ss;

    // normalize
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size) {
            float val = (float)x[index];
            val *= ss * (float)weight[index];
            o[index] = (half)val;
        }
    }
}

void rmsnorm(half* o, half* x, half* weight, int size, cudaStream_t stream) {
    int elementsPerThread = div_ceil(size, 1024);
    rmsnorm_kernel <<< 1, 1024, 0, stream>>> (o, x, weight, size, elementsPerThread);
}
