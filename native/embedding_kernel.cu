#include "embedding_kernel.cuh"

template <typename T>
inline __device__ void embedding(T* output, const T* table, const int size, const int* tokens, const int pos)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;

    int token = tokens[pos];
    int table_index = index + token * size;
    output[index] = table[table_index];
}

__global__ void embedding_kernel(half* output, const half* table, const int size, const int* tokens, const int pos)
{
    return embedding<half>(output, table, size, tokens, pos);
}
