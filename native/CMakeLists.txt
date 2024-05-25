
cmake_minimum_required(VERSION 3.8)
project(managedLlama2)
# Find CUDA package
find_package(CUDA REQUIRED)
# Enable CUDA language
enable_language(CUDA)
# Set CUDA Standard
set(CMAKE_CUDA_STANDARD 23)
# Enable separable compilation
set(CUDA_SEPARABLE_COMPILATION ON)
# Create the shared library
add_library(Llama2 OBJECT
    argmax_kernel.cu
    convert_kernel.cu
    cumulative_sum_kernel.cu
    embedding_kernel.cu
    mat_vec_kernel.cu
    rmsnorm_kernel.cu
    rope_kernel.cu
    sample_top_p_kernel.cu
    softmax_kernel.cu
    softmax_logits_kernel.cu
    sort_kernel.cu
    vec_mat_kernel.cu
    vecadd_kernel.cu)
# Set CUDA architectures the library supports
set_property(TARGET Llama2 PROPERTY CUDA_ARCHITECTURES 86-real 86-virtual)
set_property(TARGET Llama2 PROPERTY CUDA_PTX_COMPILATION ON)
# Install the ptx files in the source directory
install(
    FILES $<TARGET_OBJECTS:Llama2>
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
)
