#include "cuda_utils.h"
#include "kernels.cuh"

#define SWIZZLE_A(x, y) ((y) ^ ((x >> 2) << 3))
#define FLOAT4(p) (reinterpret_cast<float4&>(p))

template<int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__(256, 2)
optimized_gemm_kernel(
    float* a, float* b, float* c,
    int M, int N, int K)
{
    __shared__ float As_T[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;
    int load_b_row = tid / 32;
    int load_b_col = (tid % 32) * 4;

    int t_row_in_warp = (lane_id / 16) * 8;
    int c_row = warp_id * 16 + t_row_in_warp;
    int c_col_base = (lane_id % 16) * 4;
    int c_col_0 = c_col_base;

    float* a_ptr = a + (by * BM + load_a_row) * K + load_a_col;
    float* b_ptr = b + load_b_row * N + bx * BN + load_b_col;

    float sum[TM][TN] = {0.f};

    float4 tmp_a0 = FLOAT4(a_ptr[0]);
    float4 tmp_a1 = FLOAT4(a_ptr[64 * K]);
    float4 tmp_b0 = FLOAT4(b_ptr[0]);
    float4 tmp_b1 = FLOAT4(b_ptr[8 * N]);

    As_T[0][load_a_col+0][SWIZZLE_A(load_a_col+0, load_a_row)] = tmp_a0.x;
    As_T[0][load_a_col+1][SWIZZLE_A(load_a_col+1, load_a_row)] = tmp_a0.y;
    As_T[0][load_a_col+2][SWIZZLE_A(load_a_col+2, load_a_row)] = tmp_a0.z;
    As_T[0][load_a_col+3][SWIZZLE_A(load_a_col+3, load_a_row)] = tmp_a0.w;

    As_T[0][load_a_col+0][SWIZZLE_A(load_a_col+0, load_a_row+64)] = tmp_a1.x;
    As_T[0][load_a_col+1][SWIZZLE_A(load_a_col+1, load_a_row+64)] = tmp_a1.y;
    As_T[0][load_a_col+2][SWIZZLE_A(load_a_col+2, load_a_row+64)] = tmp_a1.z;
    As_T[0][load_a_col+3][SWIZZLE_A(load_a_col+3, load_a_row+64)] = tmp_a1.w;

    FLOAT4(Bs[0][load_b_row][load_b_col]) = tmp_b0;
    FLOAT4(Bs[0][load_b_row+8][load_b_col]) = tmp_b1;

    __syncthreads();

    int write_idx = 1;
    int read_idx = 0;

    for (int bk = BK; bk < K; bk += BK) {
        a_ptr += BK;
        b_ptr += BK * N;

        tmp_a0 = FLOAT4(a_ptr[0]);
        tmp_a1 = FLOAT4(a_ptr[64 * K]);
        tmp_b0 = FLOAT4(b_ptr[0]);
        tmp_b1 = FLOAT4(b_ptr[8 * N]);

        #pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row + 4)]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[read_idx][i][c_col_0]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[read_idx][i][c_col_0 + 64]);

            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    sum[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }

        As_T[write_idx][load_a_col+0][SWIZZLE_A(load_a_col+0, load_a_row)] = tmp_a0.x;
        As_T[write_idx][load_a_col+1][SWIZZLE_A(load_a_col+1, load_a_row)] = tmp_a0.y;
        As_T[write_idx][load_a_col+2][SWIZZLE_A(load_a_col+2, load_a_row)] = tmp_a0.z;
        As_T[write_idx][load_a_col+3][SWIZZLE_A(load_a_col+3, load_a_row)] = tmp_a0.w;

        As_T[write_idx][load_a_col+0][SWIZZLE_A(load_a_col+0, load_a_row+64)] = tmp_a1.x;
        As_T[write_idx][load_a_col+1][SWIZZLE_A(load_a_col+1, load_a_row+64)] = tmp_a1.y;
        As_T[write_idx][load_a_col+2][SWIZZLE_A(load_a_col+2, load_a_row+64)] = tmp_a1.z;
        As_T[write_idx][load_a_col+3][SWIZZLE_A(load_a_col+3, load_a_row+64)] = tmp_a1.w;

        FLOAT4(Bs[write_idx][load_b_row][load_b_col]) = tmp_b0;
        FLOAT4(Bs[write_idx][load_b_row+8][load_b_col]) = tmp_b1;

        __syncthreads();
        write_idx ^= 1;
        read_idx ^= 1;
    }

    #pragma unroll
    for (int i = 0; i < BK; i++) {
        float reg_a[TM], reg_b[TN];

        FLOAT4(reg_a[0]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row)]);
        FLOAT4(reg_a[4]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row + 4)]);

        FLOAT4(reg_b[0]) = FLOAT4(Bs[read_idx][i][c_col_0]);
        FLOAT4(reg_b[4]) = FLOAT4(Bs[read_idx][i][c_col_0 + 64]);

        #pragma unroll
        for (int m = 0; m < TM; m++) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                sum[m][n] += reg_a[m] * reg_b[n];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * N + bx * BN + c_col_0]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * N + bx * BN + c_col_0 + 64]) = FLOAT4(sum[i][4]);
    }
}

void launch_optimized_gemm(const float* A, const float* B, float* C,
                            int M, int N, int K, cudaStream_t stream) {
    constexpr int BM  = 128;
    constexpr int BN  = 128;
    constexpr int BK  = 16;
    constexpr int TM  = 8;
    constexpr int TN  = 8;

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    optimized_gemm_kernel<BM, BN, BK, TM, TN>
        <<<grid, 256, 0, stream>>>(
            const_cast<float*>(A), const_cast<float*>(B), C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}
