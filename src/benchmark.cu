#include "benchmark.h"
#include "cuda_utils.h"
#include <cmath>
#include <algorithm>

BenchmarkResult Benchmark::run(
    const std::string& name,
    std::function<void()> kernel_launch,
    int warmup_iters,
    int bench_iters
) {
    for (int i = 0; i < warmup_iters; ++i) {
        kernel_launch();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<float> times;
    times.reserve(bench_iters);

    for (int i = 0; i < bench_iters; ++i) {
        CudaEvent start_event;
        CudaEvent end_event;

        start_event.record();
        kernel_launch();
        end_event.record();

        CUDA_CHECK(cudaGetLastError());
        end_event.synchronize();

        float ms = end_event.elapsed_since(start_event);
        times.push_back(ms);
    }

    float sum = 0.0f;
    float min_ms = times[0];
    float max_ms = times[0];

    for (float t : times) {
        sum += t;
        if (t < min_ms) min_ms = t;
        if (t > max_ms) max_ms = t;
    }

    float avg_ms = sum / bench_iters;

    float sq_sum = 0.0f;
    for (float t : times) {
        float diff = t - avg_ms;
        sq_sum += diff * diff;
    }
    float stddev_ms = std::sqrt(sq_sum / bench_iters);

    return BenchmarkResult{name, avg_ms, min_ms, max_ms, stddev_ms};
}
