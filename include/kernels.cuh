#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

void launch_naive_gemm(const float* A, const float* B, float* C,
                        int M, int N, int K, cudaStream_t stream = 0);

void launch_tiled_gemm(const float* A, const float* B, float* C,
                        int M, int N, int K, cudaStream_t stream = 0);

void launch_tiled_gemm_padded(const float* A, const float* B, float* C,
                                int M, int N, int K, cudaStream_t stream = 0);

void launch_tiled_gemm_vec4(const float* A, const float* B, float* C,
                              int M, int N, int K, cudaStream_t stream = 0);

void launch_optimized_gemm(const float* A, const float* B, float* C,
                             int M, int N, int K, cudaStream_t stream = 0);

void launch_wmma_tf32_gemm(const float* A, const float* B, float* C,
                              int M, int N, int K, cudaStream_t stream = 0);

void launch_softmax_blockwise(const float* input, float* output,
                                int M, int N, cudaStream_t stream = 0);

void launch_online_softmax(const float* input, float* output,
                              int M, int N, cudaStream_t stream = 0);

void launch_attention_fp32_reference(const float* Q, const float* K, const float* V,
                                        float* O, float* S, float* P,
                                        int N, int D, cudaStream_t stream = 0);

void launch_attention_naive_fp16(const __half* Q, const __half* K, const __half* V,
                                   __half* O, __half* S, __half* P,
                                   int N, int D, cudaStream_t stream = 0);

void launch_attention_online_pure_fp16(const __half* Q, const __half* K, const __half* V,
                                          __half* O, int N, int D, cudaStream_t stream = 0);

void launch_attention_online_mixed_fp16(const __half* Q, const __half* K, const __half* V,
                                           __half* O, int N, int D, cudaStream_t stream = 0);

void print_occupancy_analysis();
