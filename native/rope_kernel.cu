#include "rope_kernel.cuh"

// Each block processes a single head
__global__ void RoPERotation_kernel(half* sq, half* sk_base, int num_kv_heads, int head_size, int* pPos, int loff, float rope_theta) {
    int pos = *pPos;

    int h = blockIdx.x;
    half* q = sq + h * head_size;
    int i = threadIdx.x;
    int head_dim = (i * 2) % head_size;
    float freq = 1.0f / powf(rope_theta, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    float q0 = q[i];
    float q1 = q[i + head_size / 2];
    q[i] = q0 * fcr - q1 * fci;
    q[i + head_size / 2] = q0 * fci + q1 * fcr;
    if (h < num_kv_heads) {
        half* sk = sk_base + loff + pos * num_kv_heads * head_size;
        half* k = sk + h * head_size;
        float k0 = k[i];
        float k1 = k[i + head_size / 2];
        k[i] = k0 * fcr - k1 * fci;
        k[i + head_size / 2] = k0 * fci + k1 * fcr;
    }
}
