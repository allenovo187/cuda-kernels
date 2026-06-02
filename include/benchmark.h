#pragma once
#include <string>
#include <vector>
#include <functional>

struct BenchmarkResult {
    std::string name;
    float avg_ms;
    float min_ms;
    float max_ms;
    float stddev_ms;
};

class Benchmark {
public:
    static BenchmarkResult run(
        const std::string& name,
        std::function<void()> kernel_launch,
        int warmup_iters = 5,
        int bench_iters = 10
    );
};
