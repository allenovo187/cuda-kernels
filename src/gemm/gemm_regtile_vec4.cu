#include "cuda_utils.h"
#include "kernels.cuh"

template<int BM, int BN, int BK, int TM, int TN, int PAD>
__global__ void __launch_bounds__(256)
gemm_regtile_vec4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float s_A[BM][BK + PAD];
    __shared__ float s_B[BK][BN + PAD];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int row_base = blockIdx.y * BM + ty * TM;
    int col_base = blockIdx.x * BN + tx * TN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    int num_k_tiles = (K + BK - 1) / BK;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int k = k_tile * BK;

        {
            constexpr int float4_per_row = BK / 4;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int idx = tid * 2 + i;
                int r  = idx / float4_per_row;
                int c4 = idx % float4_per_row;
                int global_r = blockIdx.y * BM + r;
                int global_c = k + c4 * 4;
                if (global_r < M && global_c + 3 < K) {
                    float4 val = *reinterpret_cast<const float4*>(
                        A + global_r * K + global_c);
                    s_A[r][c4*4+0] = val.x;
                    s_A[r][c4*4+1] = val.y;
                    s_A[r][c4*4+2] = val.z;
                    s_A[r][c4*4+3] = val.w;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int gc = k + c4*4 + j;
                        s_A[r][c4*4+j] =
                            (global_r < M && gc < K) ? A[global_r * K + gc] : 0.0f;
                    }
                }
            }
        }

        {
            constexpr int float4_per_row = BN / 4;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int idx = tid * 2 + i;
                int r  = idx / float4_per_row;
                int c4 = idx % float4_per_row;
                int global_r = k + r;
                int global_c = blockIdx.x * BN + c4 * 4;
                if (global_r < K && global_c + 3 < N) {
                    float4 val = *reinterpret_cast<const float4*>(
                        B + global_r * N + global_c);
                    s_B[r][c4*4+0] = val.x;
                    s_B[r][c4*4+1] = val.y;
                    s_B[r][c4*4+2] = val.z;
                    s_B[r][c4*4+3] = val.w;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int gc = blockIdx.x * BN + c4*4 + j;
                        s_B[r][c4*4+j] =
                            (global_r < K && gc < N) ? B[global_r * N + gc] : 0.0f;
                    }
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BK; i++) {
            float a_vals[TM];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                a_vals[m] = s_A[ty * TM + m][i];

            float b_vals[TN];
            #pragma unroll
            for (int n = 0; n < TN; n++)
                b_vals[n] = s_B[i][tx * TN + n];

            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    acc[m][n] += a_vals[m] * b_vals[n];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int row = row_base + m;
        if (row < M && col_base + 3 < N) {
            float4 val = {acc[m][0], acc[m][1], acc[m][2], acc[m][3]};
            *reinterpret_cast<float4*>(C + row * N + col_base) = val;
        } else {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int col = col_base + n;
                if (row < M && col < N)
                    C[row * N + col] = acc[m][n];
            }
        }
    }
}

void launch_gemm_regtile_vec4(const float* A, const float* B, float* C,
                                int M, int N, int K, cudaStream_t stream) {
    constexpr int BM  = 64;
    constexpr int BN  = 64;
    constexpr int BK  = 32;
    constexpr int TM  = 4;
    constexpr int TN  = 4;
    constexpr int PAD = 4;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    gemm_regtile_vec4_kernel<BM, BN, BK, TM, TN, PAD>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}
