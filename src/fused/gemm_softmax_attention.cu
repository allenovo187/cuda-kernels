#include "cuda_utils.h"
#include "kernels.cuh"
#include "fp16_utils.h"

#define FA_Br 64
#define FA_D  64

__global__ void gemm_qk_fp32_kernel(const float* Q, const float* K, float* S, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < FA_D; k++) sum += Q[i*FA_D+k] * K[j*FA_D+k];
    S[i*N+j] = sum;
}

__global__ void softmax_fp32_kernel(const float* S, float* P, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float m = -FLT_MAX;
    for (int j = 0; j < N; j++) m = fmaxf(m, S[i*N+j]);
    float l = 0.0f;
    for (int j = 0; j < N; j++) { float v = expf(S[i*N+j]-m); P[i*N+j] = v; l += v; }
    float il = 1.0f / l;
    for (int j = 0; j < N; j++) P[i*N+j] *= il;
}

__global__ void gemm_pv_fp32_kernel(const float* P, const float* V, float* O, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || k >= FA_D) return;
    float sum = 0.0f;
    for (int j = 0; j < N; j++) sum += P[i*N+j] * V[j*FA_D+k];
    O[i*FA_D+k] = sum;
}

__global__ void gemm_qk_fp16_kernel(const __half* Q, const __half* K, __half* S, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < FA_D; k++) sum += h2f(Q[i*FA_D+k]) * h2f(K[j*FA_D+k]);
    S[i*N+j] = f2h(sum);
}

__global__ void softmax_fp16_kernel(const __half* S, __half* P, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float m = -FLT_MAX;
    for (int j = 0; j < N; j++) m = fmaxf(m, h2f(S[i*N+j]));
    float l = 0.0f;
    for (int j = 0; j < N; j++) {
        float v = expf(h2f(S[i*N+j]) - m);
        P[i*N+j] = f2h(v); l += v;
    }
    float il = 1.0f / l;
    for (int j = 0; j < N; j++) P[i*N+j] = f2h(h2f(P[i*N+j]) * il);
}

__global__ void gemm_pv_fp16_kernel(const __half* P, const __half* V, __half* O, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || k >= FA_D) return;
    float sum = 0.0f;
    for (int j = 0; j < N; j++) sum += h2f(P[i*N+j]) * h2f(V[j*FA_D+k]);
    O[i*FA_D+k] = f2h(sum);
}

__global__ void online_fused_fp16_pure_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int N)
{
    int tid = threadIdx.x;
    int q_block = blockIdx.x;
    int q_start = q_block * FA_Br;

    __shared__ __half Qs[FA_Br * FA_D];
    __shared__ __half Ks[FA_Br * FA_D];
    __shared__ __half Vs[FA_Br * FA_D];

    if (q_start + tid < N) {
        #pragma unroll
        for (int k = 0; k < FA_D; k++) Qs[tid * FA_D + k] = Q[(q_start + tid) * FA_D + k];
    } else {
        #pragma unroll
        for (int k = 0; k < FA_D; k++) Qs[tid * FA_D + k] = f2h(0.0f);
    }
    __syncthreads();

    __half row_m = f2h(-65504.0f);
    __half row_l = f2h(0.0f);
    __half o_acc[FA_D];
    #pragma unroll
    for (int k = 0; k < FA_D; k++) o_acc[k] = f2h(0.0f);

    int num_kv = (N + FA_Br - 1) / FA_Br;
    for (int t = 0; t < num_kv; t++) {
        int kv_start = t * FA_Br;

        int ept = (FA_Br * FA_D) / FA_Br;
        #pragma unroll
        for (int e = 0; e < ept; e++) {
            int offset = e * FA_Br + tid;
            int r = offset / FA_D;
            int c = offset % FA_D;
            int idx = (kv_start + r) * FA_D + c;
            if (kv_start + r < N) {
                Ks[offset] = K[idx];
                Vs[offset] = V[idx];
            } else {
                Ks[offset] = f2h(0.0f);
                Vs[offset] = f2h(0.0f);
            }
        }
        __syncthreads();

        __half new_max = row_m;
        __half s_row[FA_Br];
        #pragma unroll
        for (int j = 0; j < FA_Br; j++) {
            float dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < FA_D; k++) dot += h2f(Qs[tid * FA_D + k]) * h2f(Ks[j * FA_D + k]);
            s_row[j] = f2h(dot);
            if (kv_start + j < N) new_max = __hgt(s_row[j], new_max) ? s_row[j] : new_max;
        }

        float sum = 0.0f;
        float f_new_max = h2f(new_max);
        #pragma unroll
        for (int j = 0; j < FA_Br; j++) {
            if (kv_start + j < N) {
                float v = expf(h2f(s_row[j]) - f_new_max);
                s_row[j] = f2h(v);
                sum += v;
            } else {
                s_row[j] = f2h(0.0f);
            }
        }

        float f_old_m = h2f(row_m);
        row_m = new_max;
        row_l = f2h(h2f(row_l) * expf(f_old_m - f_new_max) + sum);

        float scale = expf(f_old_m - f_new_max);
        __half h_scale = f2h(scale);
        #pragma unroll
        for (int k = 0; k < FA_D; k++) {
            float pv = 0.0f;
            #pragma unroll
            for (int j = 0; j < FA_Br; j++) pv += h2f(s_row[j]) * h2f(Vs[j * FA_D + k]);
            o_acc[k] = f2h(h2f(o_acc[k]) * scale + pv);
        }
        __syncthreads();
    }

    int global_row = q_start + tid;
    if (global_row < N) {
        float il = 1.0f / h2f(row_l);
        #pragma unroll
        for (int k = 0; k < FA_D; k++) O[global_row * FA_D + k] = f2h(h2f(o_acc[k]) * il);
    }
}

__global__ void online_fused_fp16_mixed_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int N)
{
    int tid = threadIdx.x;
    int q_block = blockIdx.x;
    int q_start = q_block * FA_Br;

    __shared__ __half Qs[FA_Br * FA_D];
    __shared__ __half Ks[FA_Br * FA_D];
    __shared__ __half Vs[FA_Br * FA_D];

    if (q_start + tid < N) {
        #pragma unroll
        for (int k = 0; k < FA_D; k++) Qs[tid * FA_D + k] = Q[(q_start + tid) * FA_D + k];
    } else {
        #pragma unroll
        for (int k = 0; k < FA_D; k++) Qs[tid * FA_D + k] = f2h(0.0f);
    }
    __syncthreads();

    float row_m = -FLT_MAX;
    float row_l = 0.0f;
    float o_acc[FA_D];
    #pragma unroll
    for (int k = 0; k < FA_D; k++) o_acc[k] = 0.0f;

    int num_kv = (N + FA_Br - 1) / FA_Br;
    for (int t = 0; t < num_kv; t++) {
        int kv_start = t * FA_Br;

        int ept = (FA_Br * FA_D) / FA_Br;
        #pragma unroll
        for (int e = 0; e < ept; e++) {
            int offset = e * FA_Br + tid;
            int r = offset / FA_D;
            int c = offset % FA_D;
            int idx = (kv_start + r) * FA_D + c;
            if (kv_start + r < N) {
                Ks[offset] = K[idx];
                Vs[offset] = V[idx];
            } else {
                Ks[offset] = f2h(0.0f);
                Vs[offset] = f2h(0.0f);
            }
        }
        __syncthreads();

        float new_max = row_m;
        float s_row[FA_Br];
        #pragma unroll
        for (int j = 0; j < FA_Br; j++) {
            float dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < FA_D; k++) dot += h2f(Qs[tid * FA_D + k]) * h2f(Ks[j * FA_D + k]);
            s_row[j] = dot;
            if (kv_start + j < N) new_max = fmaxf(new_max, dot);
        }

        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < FA_Br; j++) {
            if (kv_start + j < N) {
                s_row[j] = expf(s_row[j] - new_max);
                sum += s_row[j];
            } else {
                s_row[j] = 0.0f;
            }
        }

        float old_m = row_m;
        row_m = new_max;
        row_l = row_l * expf(old_m - row_m) + sum;

        float scale = expf(old_m - row_m);
        #pragma unroll
        for (int k = 0; k < FA_D; k++) {
            float pv = 0.0f;
            #pragma unroll
            for (int j = 0; j < FA_Br; j++) pv += s_row[j] * h2f(Vs[j * FA_D + k]);
            o_acc[k] = o_acc[k] * scale + pv;
        }
        __syncthreads();
    }

    int global_row = q_start + tid;
    if (global_row < N) {
        float il = 1.0f / row_l;
        #pragma unroll
        for (int k = 0; k < FA_D; k++) O[global_row * FA_D + k] = f2h(o_acc[k] * il);
    }
}

void launch_attention_fp32_reference(const float* Q, const float* K, const float* V,
                                        float* O, float* S, float* P,
                                        int N, int D, cudaStream_t stream) {
    dim3 block_gemm(16, 16);
    dim3 grid_qk((N+15)/16, (N+15)/16);
    dim3 grid_pv((N+15)/16, (D+15)/16);
    dim3 block_softmax(256);
    dim3 grid_softmax((N+255)/256);

    gemm_qk_fp32_kernel<<<grid_qk, block_gemm, 0, stream>>>(Q, K, S, N);
    softmax_fp32_kernel<<<grid_softmax, block_softmax, 0, stream>>>(S, P, N);
    gemm_pv_fp32_kernel<<<grid_pv, block_gemm, 0, stream>>>(P, V, O, N);
    CUDA_CHECK(cudaGetLastError());
}

void launch_attention_naive_fp16(const __half* Q, const __half* K, const __half* V,
                                   __half* O, __half* S, __half* P,
                                   int N, int D, cudaStream_t stream) {
    dim3 block_gemm(16, 16);
    dim3 grid_qk((N+15)/16, (N+15)/16);
    dim3 grid_pv((N+15)/16, (D+15)/16);
    dim3 block_softmax(256);
    dim3 grid_softmax((N+255)/256);

    gemm_qk_fp16_kernel<<<grid_qk, block_gemm, 0, stream>>>(Q, K, S, N);
    softmax_fp16_kernel<<<grid_softmax, block_softmax, 0, stream>>>(S, P, N);
    gemm_pv_fp16_kernel<<<grid_pv, block_gemm, 0, stream>>>(P, V, O, N);
    CUDA_CHECK(cudaGetLastError());
}

void launch_attention_online_pure_fp16(const __half* Q, const __half* K, const __half* V,
                                          __half* O, int N, int D, cudaStream_t stream) {
    dim3 block(FA_Br);
    dim3 grid((N + FA_Br - 1) / FA_Br);
    online_fused_fp16_pure_kernel<<<grid, block, 0, stream>>>(Q, K, V, O, N);
    CUDA_CHECK(cudaGetLastError());
}

void launch_attention_online_mixed_fp16(const __half* Q, const __half* K, const __half* V,
                                           __half* O, int N, int D, cudaStream_t stream) {
    dim3 block(FA_Br);
    dim3 grid((N + FA_Br - 1) / FA_Br);
    online_fused_fp16_mixed_kernel<<<grid, block, 0, stream>>>(Q, K, V, O, N);
    CUDA_CHECK(cudaGetLastError());
}
