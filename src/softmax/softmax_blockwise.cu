#include "cuda_utils.h"
#include "kernels.cuh"

__global__ void softmax_blockwise_kernel(const float* input, float* output, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int warps_per_block = blockDim.x / 32;

    float local_max = -1e30f;
    for (int col = tid; col < N; col += blockDim.x) {
        local_max = fmaxf(local_max, input[row * N + col]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    float warp_max = __shfl_sync(0xffffffff, local_max, 0);

    __shared__ float s_warp_max[32];
    if (lane == 0) s_warp_max[warp_id] = warp_max;
    __syncthreads();

    float row_max = -1e30f;
    if (warp_id == 0) {
        row_max = (tid < warps_per_block) ? s_warp_max[tid] : -1e30f;
        for (int offset = 16; offset > 0; offset /= 2) {
            row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
        }
        if (tid == 0) s_warp_max[0] = row_max;
    }
    __syncthreads();
    row_max = s_warp_max[0];

    float local_sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        local_sum += expf(input[row * N + col] - row_max);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    float warp_sum = __shfl_sync(0xffffffff, local_sum, 0);

    __shared__ float s_warp_sum[32];
    if (lane == 0) s_warp_sum[warp_id] = warp_sum;
    __syncthreads();

    float row_sum = 0.0f;
    if (warp_id == 0) {
        row_sum = (tid < warps_per_block) ? s_warp_sum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
        }
        if (tid == 0) s_warp_sum[0] = row_sum;
    }
    __syncthreads();
    row_sum = s_warp_sum[0];

    for (int col = tid; col < N; col += blockDim.x) {
        output[row * N + col] = expf(input[row * N + col] - row_max) / row_sum;
    }
}

void launch_softmax_blockwise(const float* input, float* output, int M, int N, cudaStream_t stream) {
    const int block_size = 256;
    softmax_blockwise_kernel<<<M, block_size, 0, stream>>>(input, output, N);
}
