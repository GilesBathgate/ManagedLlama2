#include "sample_constrained_kernel.cuh"

template <typename T>
inline __device__ void sample_constrained_kernel(T* logits, const int size, const int* constraint, const int constraint_index, const int constraint_size, const bool allow)
{
    for (int t = threadIdx.x; t < size; t += blockDim.x)
    {
        bool in_constraint = false;
        for (int i = 0; i < constraint_size; ++i) {
            if (t == constraint[constraint_index + i]) {
                in_constraint = true;
                break;
            }
        }

        if (in_constraint != allow)
            logits[t] = -INFINITY;
    }
}

__global__ void sample_constrained_kernel(half* logits, const int size, const int* constraint, const int constraint_index, const int constraint_size, const bool allow)
{
    return sample_constrained_kernel<half>(logits, size, constraint, constraint_index, constraint_size, allow);
}