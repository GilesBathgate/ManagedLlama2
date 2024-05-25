#include "vec_mat_kernel.cuh"

#include <cub/warp/warp_reduce.cuh>

using WarpReduce = cub::WarpReduce<float>;
using WarpStorage = WarpReduce::TempStorage;

template <typename T>
inline __device__ void vec_mat(T* output, const T* vector, const T* matrix, const int rows, const int cols,
    const int m_col_stride, const int o_stride, const int m_row_stride)
{
    const int start_row = blockIdx.x * 32;
    const int row_index = start_row + threadIdx.y;

    if (row_index >= rows) return;

    __shared__ T buffer[2][32][32 + 2];

    const int index = start_row + threadIdx.x + blockIdx.y * m_col_stride;

    float sum = 0;
    for (int i = 0, start_col = 0; start_col < cols; start_col += 32, i = !i) {

        const int row = start_row + threadIdx.x;
        const int col = start_col + threadIdx.y;
        const int mat_index = index + col * m_row_stride;
        buffer[i][threadIdx.y][threadIdx.x] = ((row < rows) && (col < cols)) ? matrix[mat_index] : (T)0.0f;

        __syncthreads();

        const int vec_index = start_col + threadIdx.x;
        if (vec_index >= cols) continue;

        sum += (float)buffer[i][threadIdx.x][threadIdx.y] * (float)vector[vec_index + blockIdx.y * cols];
    }

    __shared__ WarpStorage storage;
    sum = WarpReduce(storage).Sum(sum);

    if (threadIdx.x == 0)
        output[row_index + blockIdx.y * o_stride] = (T)sum;
}

__global__ void vec_mat_kernel(half* output, const half* vector, const half* matrix, const int rows, const int cols,
    const int m_col_stride, const int o_stride, const int m_row_stride)
{
    return vec_mat<half>(output, vector, matrix, rows, cols, m_col_stride, o_stride, m_row_stride);
}