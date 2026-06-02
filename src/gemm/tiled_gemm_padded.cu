#include "cuda_utils.h"
#include "kernels.cuh"

template<int TILE, int PAD>
__global__ void tiled_gemm_padded_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int N, int K) {
    __shared__ float s_A[TILE][TILE + PAD];
    __shared__ float s_B[TILE][TILE + PAD];

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

void launch_tiled_gemm_padded(const float* A, const float* B, float* C,
                                int M, int N, int K, cudaStream_t stream) {
    constexpr int TILE = 32;
    constexpr int PAD = 1;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    tiled_gemm_padded_kernel<TILE, PAD><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void print_occupancy_analysis() {
    constexpr int TILE = 32;
    constexpr int PAD = 1;
    const void* kernel = (const void*)tiled_gemm_padded_kernel<TILE, PAD>;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;

    int minGridSize = 0, opt_blockSize = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &opt_blockSize, kernel, 0, 0));
    std::cout << "cudaOccupancyMaxPotentialBlockSize: blockSize=" << opt_blockSize << std::endl;

    std::cout << "\nBlock decomposition comparison (256 threads):" << std::endl;
    struct { int tx, ty; } configs[] = {{16, 16}, {32, 8}, {8, 32}};
    for (auto& c : configs) {
        int bs = c.tx * c.ty;
        int max_blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks, kernel, bs, (size_t)0));
        int occupancy = max_blocks * bs * 100 / prop.maxThreadsPerMultiProcessor;
        std::cout << "  (" << c.tx << ", " << c.ty << "): "
                  << max_blocks << " blocks/SM, "
                  << occupancy << "% occupancy" << std::endl;
    }

    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    std::cout << "Registers per thread: " << attr.numRegs << std::endl;
    std::cout << "Shared memory per block: " << attr.sharedSizeBytes << " bytes" << std::endl;
}
