#include "embedding_kernel.cuh"

__global__ void copy_embedding_kernel(half* x, const half* __restrict__ table, int size, int* tokens, int* pPos)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    int pos = *pPos;
    int token = tokens[pos];
    int table_index = index + token * size;
    x[index] = table[table_index];
}
