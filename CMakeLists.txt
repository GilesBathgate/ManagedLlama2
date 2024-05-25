
cmake_minimum_required(VERSION 3.8)
project(nanoLlama2)
# Find CUDA package
find_package(CUDA REQUIRED)
# Enable CUDA language
enable_language(CUDA)
# Set CUDA Standard
set(CMAKE_CUDA_STANDARD 23)
# Enable separable compilation
set(CUDA_SEPARABLE_COMPILATION ON)
# Create the shared library
add_library(Llama2 SHARED libLlama2.cu)
# Set CUDA architectures the library supports
set_property(TARGET Llama2 PROPERTY CUDA_ARCHITECTURES 86-real 86-virtual)
