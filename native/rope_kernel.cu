#include "rope_kernel.cuh"

template <typename T>
inline __device__ void rope(T* q, T* k, int n_kv_heads, int head_size, int pos, float theta)
{
    const int i = blockIdx.x * head_size + threadIdx.x;
    const int j = blockDim.x + i;

    const int head_dim = (threadIdx.x * 2) % head_size;
    const float freq = 1.0f / powf(theta, head_dim / (float)head_size);
    const float val = pos * freq;
    const float fcr = cosf(val);
    const float fci = sinf(val);

    const float q0 = q[i];
    const float q1 = q[j];
    q[i] = q0 * fcr - q1 * fci;
    q[j] = q0 * fci + q1 * fcr;

    if (blockIdx.x >= n_kv_heads) return;

    const float k0 = k[i];
    const float k1 = k[j];
    k[i] = k0 * fcr - k1 * fci;
    k[j] = k0 * fci + k1 * fcr;
}

__global__ void rope_kernel(half* q, half* k, int n_kv_heads, int head_size, int pos, float theta)
{
    return rope<half>(q, k, n_kv_heads, head_size, pos, theta);
}