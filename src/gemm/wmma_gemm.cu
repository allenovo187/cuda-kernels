#include "cuda_utils.h"
#include "kernels.cuh"

#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 2))
#define SWIZZLE_B(row, col) ((col) ^ (((row >> 1) & 0x7) << 3))
#define FLOAT2(p) (reinterpret_cast<float2&>(p))

#define CP_ASYNC_CG(dst, src) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst), "l"(src))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP_0() asm volatile("cp.async.wait_group 0;\n" ::)

#define LDMATRIX_X4(R0, R1, R2, R3, PTR) \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(PTR))

#define M16N8K8(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1) \
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 " \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n" \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3) \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), "f"(C0), "f"(C1), "f"(C2), "f"(C3))

template<const int BM, const int BN, const int BK>
__global__ void wmma_tf32_gemm_kernel(
    float* a, float* b, float* c,
    int M, int N, int K)
{
    int bx = blockIdx.x, by = blockIdx.y;

    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BM];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;
    int load_b_row = tid / 32;
    int load_b_col = (tid % 32) * 4;

    int warp_id_m = warp_id / 4;
    int warp_id_n = warp_id % 4;

    float sum[4][4][4] = {0.f};

    int a_swz_col0 = SWIZZLE_A(load_a_row, load_a_col);
    int a_swz_col1 = SWIZZLE_A(load_a_row + 64, load_a_col);
    int b_swz_col0 = SWIZZLE_B(load_b_row, load_b_col);
    int b_swz_col1 = SWIZZLE_B(load_b_row + 8, load_b_col);

    uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row][a_swz_col0]));
    uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row + 64][a_swz_col1]));
    uint32_t smem_b0 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[0][load_b_row][b_swz_col0]));
    uint32_t smem_b1 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[0][load_b_row + 8][b_swz_col1]));

    CP_ASYNC_CG(smem_a0, &a[(by * BM + load_a_row) * K + 0 + load_a_col]);
    CP_ASYNC_CG(smem_a1, &a[(by * BM + load_a_row + 64) * K + 0 + load_a_col]);
    CP_ASYNC_CG(smem_b0, &b[(0 + load_b_row) * N + bx * BN + load_b_col]);
    CP_ASYNC_CG(smem_b1, &b[(0 + load_b_row + 8) * N + bx * BN + load_b_col]);
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP_0();
    __syncthreads();

    int read_idx = 0, write_idx = 1;

    for (int bk = BK; bk < K; bk += BK) {
        smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[write_idx][load_a_row][a_swz_col0]));
        smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[write_idx][load_a_row + 64][a_swz_col1]));
        smem_b0 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[write_idx][load_b_row][b_swz_col0]));
        smem_b1 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[write_idx][load_b_row + 8][b_swz_col1]));

        CP_ASYNC_CG(smem_a0, &a[(by * BM + load_a_row) * K + bk + load_a_col]);
        CP_ASYNC_CG(smem_a1, &a[(by * BM + load_a_row + 64) * K + bk + load_a_col]);
        CP_ASYNC_CG(smem_b0, &b[(bk + load_b_row) * N + bx * BN + load_b_col]);
        CP_ASYNC_CG(smem_b1, &b[(bk + load_b_row + 8) * N + bx * BN + load_b_col]);
        CP_ASYNC_COMMIT_GROUP();

        #pragma unroll
        for (int k_step = 0; k_step < 2; k_step++) {
            int k_offset = k_step * 8;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            #pragma unroll
            for (int m = 0; m < 4; m++) {
                int a_row = warp_id_m * 64 + m * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m][0], reg_a[m][1], reg_a[m][2], reg_a[m][3], addr);
            }

            #pragma unroll
            for (int n = 0; n < 4; n++) {
                int n_base = warp_id_n * 32 + n * 8;
                int b_col = n_base + (lane_id / 4);
                int b_row_0 = k_offset + (lane_id % 4);
                int b_row_1 = k_offset + (lane_id % 4) + 4;
                reg_b[n][0] = __float_as_uint(Bs[read_idx][b_row_0][SWIZZLE_B(b_row_0, b_col)]);
                reg_b[n][1] = __float_as_uint(Bs[read_idx][b_row_1][SWIZZLE_B(b_row_1, b_col)]);
            }

            #pragma unroll
            for (int m = 0; m < 4; m++) {
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    M16N8K8(sum[m][n][0], sum[m][n][1], sum[m][n][2], sum[m][n][3],
                            reg_a[m][0], reg_a[m][1], reg_a[m][2], reg_a[m][3],
                            reg_b[n][0], reg_b[n][1]);
                }
            }
        }

        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();
        read_idx ^= 1;
        write_idx ^= 1;
    }

    #pragma unroll
    for (int k_step = 0; k_step < 2; k_step++) {
        int k_offset = k_step * 8;
        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

        #pragma unroll
        for (int m = 0; m < 4; m++) {
            int a_row = warp_id_m * 64 + m * 16 + (lane_id % 16);
            int a_col = k_offset + (lane_id / 16) * 4;
            uint32_t addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m][0], reg_a[m][1], reg_a[m][2], reg_a[m][3], addr);
        }

        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int n_base = warp_id_n * 32 + n * 8;
            int b_col = n_base + (lane_id / 4);
            int b_row_0 = k_offset + (lane_id % 4);
            int b_row_1 = k_offset + (lane_id % 4) + 4;
            reg_b[n][0] = __float_as_uint(Bs[read_idx][b_row_0][SWIZZLE_B(b_row_0, b_col)]);
            reg_b[n][1] = __float_as_uint(Bs[read_idx][b_row_1][SWIZZLE_B(b_row_1, b_col)]);
        }

        #pragma unroll
        for (int m = 0; m < 4; m++) {
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                M16N8K8(sum[m][n][0], sum[m][n][1], sum[m][n][2], sum[m][n][3],
                        reg_a[m][0], reg_a[m][1], reg_a[m][2], reg_a[m][3],
                        reg_b[n][0], reg_b[n][1]);
            }
        }
    }

    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

    #pragma unroll
    for (int m = 0; m < 4; m++) {
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int c_row = by * BM + warp_id_m * 64 + m * 16;
            int c_col = bx * BN + warp_id_n * 32 + n * 8;

            FLOAT2(c[(c_row + t_row) * N + c_col + t_col]) = FLOAT2(sum[m][n][0]);
            FLOAT2(c[(c_row + t_row + 8) * N + c_col + t_col]) = FLOAT2(sum[m][n][2]);
        }
    }
}

void launch_wmma_tf32_gemm(const float* A, const float* B, float* C,
                             int M, int N, int K, cudaStream_t stream) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    wmma_tf32_gemm_kernel<BM, BN, BK>
        <<<grid, 256, 0, stream>>>(
            const_cast<float*>(A), const_cast<float*>(B), C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}
