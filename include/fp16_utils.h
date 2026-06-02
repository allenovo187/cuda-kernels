#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

inline __host__ __device__ float h2f(__half h) { return __half2float(h); }
inline __host__ __device__ __half f2h(float f) { return __float2half(f); }

inline void init_random_fp16(__half* ptr, int n, float scale) {
    for (int i = 0; i < n; i++)
        ptr[i] = f2h(((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale);
}

inline void init_random_fp32(float* ptr, int n, float scale) {
    for (int i = 0; i < n; i++)
        ptr[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
}

inline float max_diff_fp32(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

inline float avg_diff_fp32(const float* a, const float* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += fabsf(a[i] - b[i]);
    return (float)(s / n);
}

inline void half_to_float(const __half* src, float* dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = h2f(src[i]);
}
