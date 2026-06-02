#include "cuda_utils.h"
#include "kernels.cuh"

template<int TILE>
__global__ void tiled_gemm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    __shared__ float s_A[TILE][TILE];
    __shared__ float s_B[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int k = 0; k < K; k += TILE) {
        int row_a = blockIdx.y * TILE + threadIdx.y;
        int col_a = k + threadIdx.x;
        s_A[threadIdx.y][threadIdx.x] =
            (row_a < M && col_a < K) ? A[row_a * K + col_a] : 0.0f;

        int row_b = k + threadIdx.y;
        int col_b = blockIdx.x * TILE + threadIdx.x;
        s_B[threadIdx.y][threadIdx.x] =
            (row_b < K && col_b < N) ? B[row_b * N + col_b] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; i++) {
            sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_tiled_gemm(const float* A, const float* B, float* C,
                        int M, int N, int K, cudaStream_t stream) {
    constexpr int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    tiled_gemm_kernel<TILE><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}
