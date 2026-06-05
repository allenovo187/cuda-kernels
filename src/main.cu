#include "cuda_utils.h"
#include "kernels.cuh"
#include "benchmark.h"
#include "fp16_utils.h"

#include <cublas_v2.h>

#include <iostream>
#include <random>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iomanip>
#include <algorithm>

// // ============================================================
// // CPU Reference Implementations
// // ============================================================

// void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < N; ++j) {
//             double sum = 0.0;
//             for (int k = 0; k < K; ++k) {
//                 sum += (double)A[i * K + k] * (double)B[k * N + j];
//             }
//             C[i * N + j] = (float)sum;
//         }
//     }
// }

// void cpu_softmax(const float* in, float* out, int M, int N) {
//     for (int i = 0; i < M; ++i) {
//         double max_val = (double)in[i * N];
//         for (int j = 1; j < N; ++j) {
//             max_val = fmax(max_val, (double)in[i * N + j]);
//         }
//         double sum = 0.0;
//         for (int j = 0; j < N; ++j) {
//             sum += exp((double)in[i * N + j] - max_val);
//         }
//         for (int j = 0; j < N; ++j) {
//             out[i * N + j] = (float)(exp((double)in[i * N + j] - max_val) / sum);
//         }
//     }
// }

// void cpu_gemm_softmax(const float* A, const float* B, float* Out, int M, int N, int K) {
//     std::vector<float> C(M * N);
//     cpu_gemm(A, B, C.data(), M, N, K);
//     cpu_softmax(C.data(), Out, M, N);
// }

// // ============================================================
// // Validation Helper
// // ============================================================

// bool validate(const float* gpu, const float* cpu_ref, int count, const char* label) {
//     float max_abs_err = 0.0f;
//     float max_rel_err = 0.0f;
//     for (int i = 0; i < count; ++i) {
//         float diff = std::fabs(gpu[i] - cpu_ref[i]);
//         float rel = diff / std::fmax(std::fabs(cpu_ref[i]), 1.0f);
//         if (diff > max_abs_err) max_abs_err = diff;
//         if (rel > max_rel_err) max_rel_err = rel;
//     }
//     std::cout << "[Validation] " << label
//               << ": Max abs error: " << std::scientific << max_abs_err
//               << ", Max rel error: " << max_rel_err << std::endl;
//     return max_rel_err < 1e-4f;
// }

// ============================================================
// Part 1: GEMM + Softmax (FP32) Benchmark
// ============================================================

void run_gemm_softmax_benchmark() {
    std::cout << "\n================================================================\n";
    std::cout << "  Part 1: GEMM + Softmax (FP32) — Progressive Optimization\n";
    std::cout << "================================================================\n";

    const int M = 4096, N = 4096, K = 4096;
    const int total_C = M * N;

    std::vector<float> h_A(M * K), h_B(K * N);
    std::vector<float> h_C_cpu(total_C), h_C_gpu(total_C);
    std::vector<float> h_Out_gpu(total_C), h_Out_cpu(total_C);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_A) v = dist(rng);
    for (auto& v : h_B) v = dist(rng);

    DeviceBuffer<float> d_A(M * K), d_B(K * N), d_C(total_C), d_Out(total_C);
    d_A.copy_from_host(h_A.data());
    d_B.copy_from_host(h_B.data());

    // std::cout << "Computing CPU reference..." << std::endl;
    // cpu_gemm_softmax(h_A.data(), h_B.data(), h_Out_cpu.data(), M, N, K);
    // cpu_gemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);

    // // ---- GEMM Validation ----
    // std::cout << "\n--- GEMM Validation ---" << std::endl;
    // auto validate_gemm = [&](const char* name, std::function<void()> fn) {
    //     fn();
    //     CUDA_CHECK(cudaDeviceSynchronize());
    //     d_C.copy_to_host(h_C_gpu.data());
    //     validate(h_C_gpu.data(), h_C_cpu.data(), total_C, name);
    // };

    // validate_gemm("Naive GEMM",
    //     [&]() { launch_naive_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K); });
    // validate_gemm("Tiled (SMEM, bank conflict)",
    //     [&]() { launch_tiled_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K); });
    // validate_gemm("Tiled + Padding (no bank conflict)",
    //     [&]() { launch_tiled_gemm_padded(d_A.get(), d_B.get(), d_C.get(), M, N, K); });
    // validate_gemm("Tiled + float4 vec",
    //     [&]() { launch_tiled_gemm_vec4(d_A.get(), d_B.get(), d_C.get(), M, N, K); });
    // validate_gemm("Optimized (RegTile 4×4)",
    //     [&]() { launch_optimized_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K); });
    // validate_gemm("WMMA TF32 (Tensor Core)",
    //     [&]() { launch_wmma_tf32_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K); });

    // // cuBLAS reference
    // {
    //     cublasHandle_t handle;
    //     cublasCreate(&handle);
    //     float alpha = 1.0f, beta = 0.0f;
    //     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //                 N, M, K, &alpha,
    //                 d_B.get(), N, d_A.get(), K, &beta, d_C.get(), N);
    //     CUDA_CHECK(cudaDeviceSynchronize());
    //     d_C.copy_to_host(h_C_gpu.data());
    //     validate(h_C_gpu.data(), h_C_cpu.data(), total_C, "cuBLAS GEMM");
    //     cublasDestroy(handle);
    // }

    // // ---- Softmax Validation ----
    // std::cout << "\n--- Softmax Validation ---" << std::endl;
    // launch_tiled_gemm_vec4(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    // CUDA_CHECK(cudaDeviceSynchronize());

    // launch_softmax_blockwise(d_C.get(), d_Out.get(), M, N);
    // CUDA_CHECK(cudaDeviceSynchronize());
    // d_Out.copy_to_host(h_Out_gpu.data());
    // validate(h_Out_gpu.data(), h_Out_cpu.data(), total_C, "Softmax Blockwise");

    // launch_online_softmax(d_C.get(), d_Out.get(), M, N);
    // CUDA_CHECK(cudaDeviceSynchronize());
    // d_Out.copy_to_host(h_Out_gpu.data());
    // validate(h_Out_gpu.data(), h_Out_cpu.data(), total_C, "Online Softmax");

    // ---- Benchmark ----
    std::cout << "\n--- Benchmark (20 iters) ---" << std::endl;

    auto r_naive = Benchmark::run("Naive GEMM",
        [&]() { launch_naive_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K); }, 5, 20);
    auto r_tiled = Benchmark::run("Tiled+Pad+vec4",
        [&]() { launch_tiled_gemm_vec4(d_A.get(), d_B.get(), d_C.get(), M, N, K); }, 5, 20);
    auto r_regtile = Benchmark::run("RegTile 4×4 (Pad+Vec4)",
        [&]() { launch_gemm_regtile_vec4(d_A.get(), d_B.get(), d_C.get(), M, N, K); }, 5, 20);
    auto r_optimized = Benchmark::run("Optimized (Swizzle+DBF)",
        [&]() { launch_optimized_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K); }, 5, 20);
    auto r_wmma = Benchmark::run("WMMA TF32",
        [&]() { launch_wmma_tf32_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K); }, 5, 20);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float alpha = 1.0f, beta = 0.0f;
    auto r_cublas = Benchmark::run("cuBLAS FP32",
        [&]() {
            cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, d_B.get(), N, d_A.get(), K, &beta, d_C.get(), N);
        }, 5, 20);

    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    auto r_cublas_tf32 = Benchmark::run("cuBLAS TF32",
        [&]() {
            cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, d_B.get(), N, d_A.get(), K, &beta, d_C.get(), N);
        }, 5, 20);
    cublasDestroy(cublas_handle);
    

    auto r_softmax_block = Benchmark::run("Blockwise Softmax",
        [&]() { launch_softmax_blockwise(d_C.get(), d_Out.get(), M, N); }, 5, 20);
    auto r_softmax_online = Benchmark::run("Online Softmax",
        [&]() { launch_online_softmax(d_C.get(), d_Out.get(), M, N); }, 5, 20);

    auto r_separated = Benchmark::run("GEMM + Softmax (separated)",
        [&]() {
            launch_naive_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K);
            launch_softmax_blockwise(d_C.get(), d_Out.get(), M, N);
        }, 5, 20);

    // ---- Results Table ----
    std::cout << "\n========================================" << std::endl;
    std::cout << "Matrix Size: (" << M << ", " << K << ") x (" << K << ", " << N << ")" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    float baseline = r_naive.avg_ms;
    auto print_row = [&](const BenchmarkResult& r) {
        std::cout << r.name << ":"
                  << std::setw(std::max(1, 36 - (int)r.name.size())) << ""
                  << "avg=" << r.avg_ms << " ms"
                  << "  (speedup: " << baseline / r.avg_ms << "x vs Naive)"
                  << std::endl;
    };

    print_row(r_naive);
    print_row(r_tiled);
    print_row(r_regtile);
    print_row(r_optimized);
    print_row(r_cublas);
    print_row(r_cublas_tf32);
    print_row(r_wmma);
    print_row(r_softmax_block);
    print_row(r_softmax_online);

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "GEMM + Softmax (separated):   "
              << "avg=" << r_separated.avg_ms << " ms" << std::endl;
    std::cout << "========================================" << std::endl;
}

// ============================================================
// Part 2: GEMM+Softmax Attention (FP16) Benchmark
// ============================================================

void run_gemm_softmax_attention_benchmark(int N) {
    std::cout << "\n================================================================\n";
    std::cout << "  Part 2: GEMM+Softmax Attention (FP16) — Precision & Fusion Analysis\n";
    std::cout << "================================================================\n";
    std::cout << "Sequence Length N=" << N << ", Head Dim D=64, Block Br=64\n\n";

    const int D = 64;

    size_t half_size = (size_t)N * D * sizeof(__half);
    size_t float_size = (size_t)N * D * sizeof(float);
    size_t s_half = (size_t)N * N * sizeof(__half);
    size_t s_float = (size_t)N * N * sizeof(float);

    // Host allocation
    __half *h_Qh, *h_Kh, *h_Vh;
    float *h_Qf, *h_Kf, *h_Vf, *h_O_gt;
    float *h_O_naive, *h_O_pure, *h_O_mixed;
    cudaMallocHost(&h_Qh, half_size); cudaMallocHost(&h_Kh, half_size); cudaMallocHost(&h_Vh, half_size);
    cudaMallocHost(&h_Qf, float_size); cudaMallocHost(&h_Kf, float_size); cudaMallocHost(&h_Vf, float_size);
    cudaMallocHost(&h_O_gt, float_size);
    cudaMallocHost(&h_O_naive, float_size);
    cudaMallocHost(&h_O_pure, float_size);
    cudaMallocHost(&h_O_mixed, float_size);

    srand(42);
    float scale = 0.02f;
    init_random_fp16(h_Qh, N*D, scale); init_random_fp16(h_Kh, N*D, scale); init_random_fp16(h_Vh, N*D, scale);
    for (int i = 0; i < N*D; i++) {
        h_Qf[i] = h2f(h_Qh[i]); h_Kf[i] = h2f(h_Kh[i]); h_Vf[i] = h2f(h_Vh[i]);
    }

    // Device allocation
    __half *d_Qh, *d_Kh, *d_Vh, *d_Oh;
    float *d_Qf, *d_Kf, *d_Vf, *d_Of;
    __half *d_Sh, *d_Ph;
    float *d_Sf, *d_Pf;
    cudaMalloc(&d_Qh, half_size); cudaMalloc(&d_Kh, half_size); cudaMalloc(&d_Vh, half_size); cudaMalloc(&d_Oh, half_size);
    cudaMalloc(&d_Qf, float_size); cudaMalloc(&d_Kf, float_size); cudaMalloc(&d_Vf, float_size); cudaMalloc(&d_Of, float_size);
    cudaMalloc(&d_Sh, s_half); cudaMalloc(&d_Ph, s_half);
    cudaMalloc(&d_Sf, s_float); cudaMalloc(&d_Pf, s_float);

    cudaMemcpy(d_Qh, h_Qh, half_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kh, h_Kh, half_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vh, h_Vh, half_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Qf, h_Qf, float_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kf, h_Kf, float_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vf, h_Vf, float_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int warmup = 3, repeats = 10;

    // Ground Truth: FP32
    float ms_gt = 0;
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        launch_attention_fp32_reference(d_Qf, d_Kf, d_Vf, d_Of, d_Sf, d_Pf, N, D);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_gt, start, stop); ms_gt /= repeats;
    cudaMemcpy(h_O_gt, d_Of, float_size, cudaMemcpyDeviceToHost);

    // Variant 1: Naive FP16 separated
    float ms_naive = 0;
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        launch_attention_naive_fp16(d_Qh, d_Kh, d_Vh, d_Oh, d_Sh, d_Ph, N, D);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_naive, start, stop); ms_naive /= repeats;
    __half* h_tmp = (__half*)malloc(half_size);
    cudaMemcpy(h_tmp, d_Oh, half_size, cudaMemcpyDeviceToHost);
    half_to_float(h_tmp, h_O_naive, N*D);

    // Variant 2: Online Pure FP16
    float ms_pure = 0;
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        launch_attention_online_pure_fp16(d_Qh, d_Kh, d_Vh, d_Oh, N, D);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_pure, start, stop); ms_pure /= repeats;
    cudaMemcpy(h_tmp, d_Oh, half_size, cudaMemcpyDeviceToHost);
    half_to_float(h_tmp, h_O_pure, N*D);

    // Variant 3: Online Mixed FP16
    float ms_mixed = 0;
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        launch_attention_online_mixed_fp16(d_Qh, d_Kh, d_Vh, d_Oh, N, D);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_mixed, start, stop); ms_mixed /= repeats;
    cudaMemcpy(h_tmp, d_Oh, half_size, cudaMemcpyDeviceToHost);
    half_to_float(h_tmp, h_O_mixed, N*D);

    // ---- Accuracy ----
    std::cout << "[Accuracy vs FP32 Ground Truth]" << std::endl;
    float pure_max = max_diff_fp32(h_O_gt, h_O_pure, N*D);
    printf("  %-32s  MaxDiff=%.6e  %s\n",
           "1. Naive FP16 (separated)",
           max_diff_fp32(h_O_gt, h_O_naive, N*D),
           max_diff_fp32(h_O_gt, h_O_naive, N*D) < 1e-3 ? "[PASS]" : "[WARN]");
    printf("  %-32s  MaxDiff=%.6e  %s\n",
           "2. Online Pure FP16",
           pure_max,
           pure_max < 1e-3 ? "[PASS]" : (pure_max < 1e-1 ? "[WARN]" : "[FAIL]"));
    printf("  %-32s  MaxDiff=%.6e  %s\n",
           "3. Online Mixed FP16",
           max_diff_fp32(h_O_gt, h_O_mixed, N*D),
           max_diff_fp32(h_O_gt, h_O_mixed, N*D) < 1e-3 ? "[PASS]" : "[WARN]");
    printf("  Pure/Mixed accuracy ratio: %.1fx\n",
           max_diff_fp32(h_O_gt, h_O_pure, N*D) / std::max(max_diff_fp32(h_O_gt, h_O_mixed, N*D), 1e-10f));

    // ---- Performance ----


    std::cout << "\n[Performance]" << std::endl;
    printf("  %-28s %12s %12s %12s\n", "Metric", "Naive FP16", "Online Pure", "Online Mixed");
    printf("  %-28s %12.3f %12.3f %12.3f\n", "Time (ms)", ms_naive, ms_pure, ms_mixed);
    printf("  %-28s %12.2fx %12.2fx %12.2fx\n", "Speedup vs Naive", 1.0f, ms_naive/ms_pure, ms_naive/ms_mixed);

    if (pure_max > 1e-1) {
        printf("\n[WARNING] Online Pure FP16 numerical deviation too large!\n");
        printf("  Cause: running max/sum/accumulator in FP16 leads to\n");
        printf("  large-swallow-small, exponent underflow, accumulation precision loss.\n");
        printf("  Production must use Mixed Precision (Variant 3).\n");
    }

    // Cleanup
    free(h_tmp);
    cudaFreeHost(h_Qh); cudaFreeHost(h_Kh); cudaFreeHost(h_Vh);
    cudaFreeHost(h_Qf); cudaFreeHost(h_Kf); cudaFreeHost(h_Vf);
    cudaFreeHost(h_O_gt); cudaFreeHost(h_O_naive); cudaFreeHost(h_O_pure); cudaFreeHost(h_O_mixed);
    cudaFree(d_Qh); cudaFree(d_Kh); cudaFree(d_Vh); cudaFree(d_Oh);
    cudaFree(d_Qf); cudaFree(d_Kf); cudaFree(d_Vf); cudaFree(d_Of);
    cudaFree(d_Sh); cudaFree(d_Ph); cudaFree(d_Sf); cudaFree(d_Pf);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;
    std::cout << "SM count: " << prop.multiProcessorCount << std::endl;
    std::cout << "HBM: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    // Part 1: GEMM + Softmax FP32 progressive optimization
    run_gemm_softmax_benchmark();

    // Part 2: GEMM+Softmax Attention FP16 benchmark (default N=1024, or from argv)
    int attn_N = (argc > 1) ? atoi(argv[1]) : 4096;
    run_gemm_softmax_attention_benchmark(attn_N);

    // Occupancy analysis
    std::cout << "\n========== Occupancy Analysis ==========" << std::endl;
    print_occupancy_analysis();

    return 0;
}
