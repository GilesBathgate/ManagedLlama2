#pragma once

#include <cstdint>
#include <cuda_fp16.h>

/**
 * @brief Performs matrix-vector multiplication using half-precision floating-point format.
 *
 * This function multiplies a matrix by a vector and stores the result in an output vector.
 * All data types are in half-precision floating-point format (16 bits).
 *
 * @param output Pointer to the output vector where the result will be stored.
 * @param vector Pointer to the input vector.
 * @param matrix Pointer to the input matrix.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 */
extern "C" __global__ void mat_vec_kernel(half* output, const half* vector, const half* matrix, const int rows, const int cols);

/**
 * @brief Performs matrix-vector multiplication with strides using half-precision floating-point format.
 *
 * This function multiplies a matrix by a vector and stores the result in an output vector.
 * All data types are in half-precision floating-point format (16 bits). Strides are used
 * to specify the memory access pattern for each array.
 *
 * @param output Pointer to the output vector where the result will be stored.
 * @param vector Pointer to the input vector.
 * @param matrix Pointer to the input matrix.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param v_stride Stride (step size) between elements in the vector.
 * @param m_col_stride Stride (step size) between elements in each column of the matrix.
 * @param m_row_stride Stride (step size) between rows of the matrix.
 * @param o_stride Stride (step size) between elements in the output vector.
 * @param alpha Scaling factor applied during the matrix-vector multiplication.
 */
extern "C" __global__ void mat_vec_strided_kernel(half* output, const half* vector, const half* matrix, const int rows, const int cols,
    const int v_stride, const int m_col_stride, const int m_row_stride, const int o_stride, const float alpha);

/**
 * @brief Performs matrix-vector multiplication using quantized weights, with a fused residual connection
 *
 * This function performs matrix-vector multiplication with a fused residual connection from the input back
 * into the output using quantized weights (QWeight) stored in `weights` with zero-point values in `zeros`
 * and scaling factors in `scales`. All data types are assumed to be in their respective formats.
 *
 * @param output Pointer to the output vector where the result will be stored.
 * @param input Pointer to the input vector.
 * @param weights Pointer to the quantized weight values.
 * @param zeros Pointer to the zero-point values.
 * @param scales Pointer to the scaling factors.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param zeros_size Size of the zero-point vector.
 * @param scales_size Size of the scaling factor vector.
 * @param weights_size Size of the quantized weight vector.
 */
extern "C" __global__ void mat_vec_residual_int4_kernel(half* output, const half* input, const uint32_t* weights, const uint32_t* zeros, const half* scales,
    const int rows, const int cols, const int zeros_size, const int scales_size, const int weights_size);

/**
 * @brief Performs q,k,v matrix-vector multiplication using quantized weights.
 *
 * This function performs matrix-vector multiplication on the input vector using seperate
 * quantized weights (QWeight) for query key and value. The results are stored in respective
 * outputs. The weights are stored in `weights` with zero-point values in `zeros` and scaling
 * factors in `scales`. All data types are assumed to be in their respective formats.
 *
 * @param q_output Pointer to the output vector where the query result will be stored.
 * @param k_output Pointer to the output vector where the key result will be stored.
 * @param v_output Pointer to the output vector where the value result will be stored.
 * @param input Pointer to the input vector.
 * @param q_weights Pointer to the quantized query weight values.
 * @param q_zeros Pointer to the query zero-point values.
 * @param q_scales Pointer to the query scaling factors.
 * @param k_weights Pointer to the quantized key weight values.
 * @param k_zeros Pointer to the key zero-point values.
 * @param k_scales Pointer to the key scaling factors.
 * @param v_weights Pointer to the quantized value weights
 * @param v_zeros Pointer to the value zero-point values.
 * @param v_scales Pointer to the value scaling factors.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param zeros_size Size of the zero-point vector.
 * @param scales_size Size of the scaling factor vector.
 * @param weights_size Size of the quantized weight vector.
 */
extern "C" __global__ void qkv_mat_vec_kernel(half* q_output, half* k_output, half* v_output, const half* input,
    const uint32_t* q_weight, const uint32_t* q_zeros, const half*  q_scales,
    const uint32_t* k_weight, const uint32_t* k_zeros, const half*  k_scales,
    const uint32_t* v_weight, const uint32_t* v_zeros, const half*  v_scales,
    const int rows, const int cols, const int zeros_size, const int scales_size, const int weights_size);

/**
 * @brief Performs matrix-vector multiplication and SwiGLU activation using quantized weights.
 *
 * This function performs a fused matrix-vector multiplication and SwiGLU activation
 * on the input vector. The weights are quantized weights (QWeight) stored in `weights`
 * with zero-point values in `zeros` and scaling factors in `scales`. All other data types
 * are assumed to be in their respective formats.
 *
 * @param output Pointer to the output vector where the result will be stored.
 * @param input Pointer to the input vector.
 * @param g_weights Pointer to the quantized weight values.
 * @param g_zeros Pointer to the zero-point values.
 * @param g_scales Pointer to the scaling factors.
 * @param u_weights Pointer to the quantized weight values.
 * @param u_zeros Pointer to the zero-point values.
 * @param u_scales Pointer to the scaling factors.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param zeros_size Size of the zero-point vector.
 * @param scales_size Size of the scaling factor vector.
 * @param weights_size Size of the quantized weight vector.
 */
extern "C" __global__ void mat_vec_swiglu_kernel(half* output, const half* input,
    const uint32_t* g_weight, const uint32_t* g_zeros, const half* g_scales,
    const uint32_t* u_weight, const uint32_t* u_zeros, const half* u_scales,
    const int rows, const int cols, const int zeros_size, const int scales_size, const int weights_size);
