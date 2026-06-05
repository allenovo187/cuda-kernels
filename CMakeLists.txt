cmake_minimum_required(VERSION 3.18)
project(GemmSoftmax LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90)

find_package(CUDAToolkit REQUIRED)

add_library(cuda_config INTERFACE)
target_include_directories(cuda_config INTERFACE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(cuda_config INTERFACE CUDA::cudart CUDA::cublas)

add_library(gemm_softmax_lib
    # GEMM kernels (progressive optimization)
    src/gemm/naive_gemm.cu
    src/gemm/tiled_gemm.cu
    src/gemm/tiled_gemm_padded.cu
    src/gemm/tiled_gemm_vec4.cu
    src/gemm/gemm_regtile_vec4.cu
    src/gemm/optimized_gemm.cu
    src/gemm/wmma_gemm.cu
    # Softmax kernels
    src/softmax/softmax_blockwise.cu
    src/softmax/online_softmax.cu
    # Fused operators
    src/fused/gemm_softmax_attention.cu
    # Infrastructure
    src/benchmark.cu
)
target_link_libraries(gemm_softmax_lib PUBLIC cuda_config)

add_executable(gemm_softmax src/main.cu)
target_link_libraries(gemm_softmax PRIVATE gemm_softmax_lib)
