//
// Created by aichao on 2025/6/27.
//
#pragma once

class CudaBuffer {
public:
    explicit CudaBuffer(size_t size) : size_(size), data_(nullptr) {
        const cudaError_t err = cudaMalloc(&data_, size_);
        if (err != cudaSuccess) {
            MD_LOG_ERROR << "CUDA malloc failed: " << cudaGetErrorString(err);
            throw std::runtime_error("CUDA memory allocation failed");
        }
    }

    ~CudaBuffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    [[nodiscard]] void* data() const { return data_; }
    [[nodiscard]] size_t size() const { return size_; }

    // 复制数据到设备
    void copy_to_device(const void* host_data) const {
        cudaError_t err = cudaMemcpy(data_, host_data, size_, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            MD_LOG_ERROR << "CUDA memory copy failed for buffer: " << cudaGetErrorString(err);
            throw std::runtime_error("CUDA memory copy to device failed");
        }
    }

    // 从设备复制数据到主机
    void copy_to_host(void* host_data) const {
        cudaError_t err = cudaMemcpy(host_data, data_, size_, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            MD_LOG_ERROR << "CUDA memory copy failed for buffer: " << cudaGetErrorString(err);
            throw std::runtime_error("CUDA memory copy to host failed");
        }
    }

private:
    size_t size_;
    void* data_;
};

using CudaBufferPrt = std::unique_ptr<CudaBuffer>;

inline CudaBufferPrt allocate_cuda_buffer(size_t size) {
    return std::make_unique<CudaBuffer>(size);
}
