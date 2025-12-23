//
// Created by aichao on 2025/6/27.
//
#pragma once
#include <core/md_log.h>

class CudaBuffer {
public:
    CudaBuffer() = default;

    explicit CudaBuffer(const size_t size) : byte_size_(size) {
        const cudaError_t err = cudaMalloc(&buffer_, byte_size_);
        if (err != cudaSuccess) {
            MD_LOG_ERROR << "CUDA malloc failed: " << cudaGetErrorString(err);
            throw std::runtime_error("CUDA memory allocation failed");
        }
    }

    explicit CudaBuffer(void* external_buffer, const size_t byte_size) {
        byte_size_ = byte_size;
        external_buffer_ = external_buffer;
    }

    ~CudaBuffer() {
        if (buffer_) {
            cudaFree(buffer_);
        }
        external_buffer_ = nullptr;
    }

    [[nodiscard]] void* data() const {
        if (external_buffer_)
            return external_buffer_;
        return buffer_;
    }

    [[nodiscard]] size_t byte_size() const { return byte_size_; }

    // 复制数据到设备
    void copy_from_host(const void* host_data) const {
        void* device_buffer = nullptr;
        if (external_buffer_)
            device_buffer = external_buffer_;
        else
            device_buffer = buffer_;
        const cudaError_t err = cudaMemcpy(device_buffer, host_data, byte_size_, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            MD_LOG_ERROR << "CUDA memory copy failed for buffer: " << cudaGetErrorString(err);
            throw std::runtime_error("CUDA memory copy to device failed");
        }
    }

    void copy_from_device(const void* device_data) const {
        const cudaError_t err = cudaMemcpy(buffer_, device_data, byte_size_, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            MD_LOG_ERROR << "CUDA memory copy failed for buffer: " << cudaGetErrorString(err);
            throw std::runtime_error("CUDA memory copy to device failed");
        }
    }

    // 从设备复制数据到主机
    void copy_to_host(void* host_data) const {
        const cudaError_t err = cudaMemcpy(host_data, buffer_, byte_size_, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            MD_LOG_ERROR << "CUDA memory copy failed for buffer: " << cudaGetErrorString(err);
            throw std::runtime_error("CUDA memory copy to host failed");
        }
    }

    void shared_from_external(const size_t byte_size, void* external_buffer) {
        byte_size_ = byte_size;
        external_buffer_ = external_buffer;
    }

private:
    size_t byte_size_ = 0;
    void* buffer_ = nullptr;
    void* external_buffer_ = nullptr;
};

using CudaBufferPrt = std::unique_ptr<CudaBuffer>;

inline CudaBufferPrt allocate_cuda_buffer(size_t size) {
    return std::make_unique<CudaBuffer>(size);
}
