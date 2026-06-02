#include "cuda_utils.h"
#include "kernels.cuh"

__global__ void online_softmax_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int warps_per_block = blockDim.x / 32;

    float m = -1e30f;
    float d = 0.0f;

    for (int col = tid; col < N; col += blockDim.x) {
        float x = input[row * N + col];
        float m_new = fmaxf(m, x);
        d = d * expf(m - m_new) + expf(x - m_new);
        m = m_new;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        float other_m = __shfl_down_sync(0xffffffff, m, offset);
        float other_d = __shfl_down_sync(0xffffffff, d, offset);
        float m_new = fmaxf(m, other_m);
        d = d * expf(m - m_new) + other_d * expf(other_m - m_new);
        m = m_new;
    }
    float warp_m = __shfl_sync(0xffffffff, m, 0);
    float warp_d = __shfl_sync(0xffffffff, d, 0);

    __shared__ float s_m[32];
    __shared__ float s_d[32];
    if (lane == 0) {
        s_m[warp_id] = warp_m;
        s_d[warp_id] = warp_d;
    }
    __syncthreads();

    float row_m = -1e30f;
    float row_d = 0.0f;
    if (warp_id == 0) {
        m = (tid < warps_per_block) ? s_m[tid] : -1e30f;
        d = (tid < warps_per_block) ? s_d[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_m = __shfl_down_sync(0xffffffff, m, offset);
            float other_d = __shfl_down_sync(0xffffffff, d, offset);
            float m_new = fmaxf(m, other_m);
            d = d * expf(m - m_new) + other_d * expf(other_m - m_new);
            m = m_new;
        }
        if (tid == 0) {
            s_m[0] = m;
            s_d[0] = d;
        }
    }
    __syncthreads();
    row_m = s_m[0];
    row_d = s_d[0];

    for (int col = tid; col < N; col += blockDim.x) {
        float x = input[row * N + col];
        output[row * N + col] = expf(x - row_m) / row_d;
    }
}

void launch_online_softmax(const float* input, float* output,
                             int M, int N, cudaStream_t stream) {
    const int block_size = 256;
    online_softmax_kernel<<<M, block_size, 0, stream>>>(input, output, N);
}
