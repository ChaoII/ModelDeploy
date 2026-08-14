//
// 设备输出缓冲池：CUDA 预处理输出复用设备内存，避免每帧 cudaMalloc/cudaFree。
// 池由 CudaProcessorBackend 持有（与持久 stream 同生命周期），生命周期内按需扩容。
// 输出的 GPU Tensor 通过 Tensor::from_external_memory(..., Device::GPU) 零拷贝包装，
// Tensor 不拥有设备内存（空 deleter），设备内存由本池负责释放。
//
#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace modeldeploy::vision {

    class CudaOutputBufferPool {
    public:
        CudaOutputBufferPool() = default;
        ~CudaOutputBufferPool() { release(); }

        CudaOutputBufferPool(const CudaOutputBufferPool&) = delete;
        CudaOutputBufferPool& operator=(const CudaOutputBufferPool&) = delete;

        // 获取至少 bytes 字节的设备缓冲（复用或按需 cudaMalloc 扩容）
        float* acquire(const size_t bytes) {
            if (bytes > capacity_) {
                release();
                if (cudaMalloc(&d_ptr_, bytes) != cudaSuccess) {
                    d_ptr_ = nullptr;
                    capacity_ = 0;
                    return nullptr;
                }
                capacity_ = bytes;
            }
            return static_cast<float*>(d_ptr_);
        }

        void release() {
            if (d_ptr_) {
                cudaFree(d_ptr_);
                d_ptr_ = nullptr;
                capacity_ = 0;
            }
        }

        [[nodiscard]] size_t capacity() const { return capacity_; }

    private:
        void* d_ptr_ = nullptr;
        size_t capacity_ = 0;
    };

} // namespace modeldeploy::vision
