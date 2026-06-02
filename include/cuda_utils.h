#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl; \
            std::abort(); \
        }\
    } while(0)

class CudaStream {
    cudaStream_t stream_;
public:
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_)); }
    ~CudaStream() {
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }
};

class CudaEvent {
    cudaEvent_t event_;
public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_)); 
    }
    ~CudaEvent() {
        CUDA_CHECK(cudaEventDestroy(event_)); 
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    cudaEvent_t get() const { return event_; }
    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }
    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }
    float elapsed_since(const CudaEvent& start) const {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }
};

template<typename T>
class DeviceBuffer {
    T* ptr_ = nullptr;
    size_t size_ = 0;
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) : size_(n) {
        CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (ptr_) CUDA_CHECK(cudaFree(ptr_));
    }

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) CUDA_CHECK(cudaFree(ptr_));
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }

    void copy_from_host(const T* host_ptr, cudaStream_t stream = 0) {
        CUDA_CHECK(cudaMemcpyAsync(ptr_, host_ptr, bytes(),
                    cudaMemcpyHostToDevice, stream));
    }
    void copy_to_host(T* host_ptr, cudaStream_t stream = 0) {
        CUDA_CHECK(cudaMemcpyAsync(host_ptr, ptr_, bytes(),
                    cudaMemcpyDeviceToHost, stream));
    }
};
