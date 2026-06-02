#include "cuda_utils.h"
#include "kernels.cuh"
#include <mma.h>

template<int BM, int BN, int BK, int PAD>
__global__ void __launch_bounds__(256)
wmma_tf32_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float s_A[2][BM][BK + PAD];
    __shared__ float s_B[2][BK][BN + PAD];
    __shared__ float s_out[16][16 + 1];

    using namespace nvcuda;

    constexpr int WMMAM = 16, WMMAN = 16, WMMAK = 8;
    constexpr int WARP_M = 16, WARP_N = 32;
    constexpr int TM = WARP_M / WMMAM;       // 1
    constexpr int TN = WARP_N / WMMAN;        // 2

    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m  = warp_id / 2;
    int warp_n  = warp_id % 2;

    wmma::fragment<wmma::accumulator, WMMAM, WMMAN, WMMAK, float> frag_c[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            for (int k = 0; k < frag_c[i][j].num_elements; k++)
                frag_c[i][j].x[k] = 0.0f;

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
                    s_A[0][r][c4*4+0] = val.x;
                    s_A[0][r][c4*4+1] = val.y;
                    s_A[0][r][c4*4+2] = val.z;
                    s_A[0][r][c4*4+3] = val.w;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int gc = k + c4*4 + j;
                        s_A[0][r][c4*4+j] =
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
                    s_B[0][r][c4*4+0] = val.x;
                    s_B[0][r][c4*4+1] = val.y;
                    s_B[0][r][c4*4+2] = val.z;
                    s_B[0][r][c4*4+3] = val.w;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int gc = blockIdx.x * BN + c4*4 + j;
                        s_B[0][r][c4*4+j] =
                            (global_r < K && gc < N) ? B[global_r * N + gc] : 0.0f;
                    }
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k_wmma = 0; k_wmma < BK / WMMAK; k_wmma++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                wmma::fragment<wmma::matrix_a, WMMAM, WMMAN, WMMAK,
                               wmma::precision::tf32, wmma::row_major> frag_a;
                wmma::load_matrix_sync(frag_a,
                    &s_A[0][warp_m * WARP_M + i * WMMAM][k_wmma * WMMAK], BK + PAD);

                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    wmma::fragment<wmma::matrix_b, WMMAM, WMMAN, WMMAK,
                                   wmma::precision::tf32, wmma::row_major> frag_b;
                    wmma::load_matrix_sync(frag_b,
                        &s_B[0][k_wmma * WMMAK][warp_n * WARP_N + j * WMMAN], BN + PAD);

                    wmma::mma_sync(frag_c[i][j], frag_a, frag_b, frag_c[i][j]);
                }
            }
        }

        __syncthreads();
    }

    int row_base = blockIdx.y * BM + warp_m * WARP_M;
    int col_base = blockIdx.x * BN + warp_n * WARP_N;

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int row = row_base + i * WMMAM;
            int col = col_base + j * WMMAN;
            bool fully_in = (row + WMMAM <= M) && (col + WMMAN <= N);

            if (fully_in) {
                wmma::store_matrix_sync(C + row * N + col,
                    frag_c[i][j], N, wmma::mem_row_major);
            } else {
                if (warp_id == (warp_m * 2 + warp_n)) {
                    wmma::store_matrix_sync(&s_out[0][0],
                        frag_c[i][j], 17, wmma::mem_row_major);
                }
                __syncthreads();

                if (row < M && col < N) {
                    for (int idx = tid; idx < WMMAM * WMMAN; idx += 256) {
                        int r = idx / WMMAN;
                        int c = idx % WMMAN;
                        int gr = row + r;
                        int gc = col + c;
                        if (gr < M && gc < N)
                            C[gr * N + gc] = s_out[r][c];
                    }
                }
                __syncthreads();
            }
        }
    }
}

void launch_wmma_tf32_gemm(const float* A, const float* B, float* C,
                             int M, int N, int K, cudaStream_t stream) {
    constexpr int BM  = 64;
    constexpr int BN  = 64;
    constexpr int BK  = 32;
    constexpr int PAD = 4;

    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    wmma_tf32_gemm_kernel<BM, BN, BK, PAD>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}
