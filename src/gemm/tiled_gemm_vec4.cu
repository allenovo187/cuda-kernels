#include "cuda_utils.h"
#include "kernels.cuh"

template<int TILE, int PAD>
__global__ void tiled_gemm_vec4_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K) {
    __shared__ float s_A[TILE][TILE + PAD];
    __shared__ float s_B[TILE][TILE + PAD];

    int tx = threadIdx.x; // 0..7
    int ty = threadIdx.y; // 0..31

    int row = blockIdx.y * TILE + ty;
    int col_base = blockIdx.x * TILE + tx * 4;

    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = 0; k < K; k += TILE) {
        int row_a = blockIdx.y * TILE + ty;
        int col_a_base = k + tx * 4;

        if (row_a < M && col_a_base + 3 < K) {
            const float4* A4 = reinterpret_cast<const float4*>(A + row_a * K + k);
            float4 val = A4[tx];
            s_A[ty][tx * 4 + 0] = val.x;
            s_A[ty][tx * 4 + 1] = val.y;
            s_A[ty][tx * 4 + 2] = val.z;
            s_A[ty][tx * 4 + 3] = val.w;
        } else {
            for (int i = 0; i < 4; i++) {
                int ca = k + tx * 4 + i;
                s_A[ty][tx * 4 + i] =
                    (row_a < M && ca < K) ? A[row_a * K + ca] : 0.0f;
            }
        }

        for (int i = 0; i < 4; i++) {
            int row_b = k + ty;
            int col_b = blockIdx.x * TILE + tx * 4 + i;
            s_B[ty][tx * 4 + i] =
                (row_b < K && col_b < N) ? B[row_b * N + col_b] : 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE; i++) {
            float a_val = s_A[ty][i];
            for (int j = 0; j < 4; j++) {
                sum[j] += a_val * s_B[i][tx * 4 + j];
            }
        }

        __syncthreads();
    }

    for (int j = 0; j < 4; j++) {
        int col = col_base + j;
        if (row < M && col < N) {
            C[row * N + col] = sum[j];
        }
    }
}

void launch_tiled_gemm_vec4(const float* A, const float* B, float* C,
                              int M, int N, int K, cudaStream_t stream) {
    constexpr int TILE = 32;
    constexpr int PAD = 1;
    dim3 block(8, 32);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    tiled_gemm_vec4_kernel<TILE, PAD><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}
