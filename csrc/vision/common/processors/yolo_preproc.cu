//
// Created by aichao on 2025/7/22.
//

#include "vision/common/processors/yolo_preproc.cuh"
#include <cuda_fp16.h>
#include <core/md_log.h>

__global__ void kernel_yolo_preproc(
    const uint8_t* __restrict__ src,
    const int src_h, const int src_w, const int src_step,
    float* __restrict__ dst,
    const int dst_h, const int dst_w,
    const float* mean,
    const float* std,
    const float pad_value,
    float* scale_out,
    float* pad_w_out,
    float* pad_h_out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    // 使用共享内存缓存计算结果，避免重复计算
    __shared__ float shared_scale, shared_pad_h, shared_pad_w;

    // 第一个线程计算共享参数
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        shared_scale = fminf(float(dst_h) / src_h, float(dst_w) / src_w);
        const int new_h = int(src_h * shared_scale + 0.5f);
        const int new_w = int(src_w * shared_scale + 0.5f);
        shared_pad_h = (dst_h - new_h) * 0.5f;
        shared_pad_w = (dst_w - new_w) * 0.5f;

        *scale_out = shared_scale;
        *pad_h_out = shared_pad_h;
        *pad_w_out = shared_pad_w;
    }
    __syncthreads();

    // 目标坐标 → 源坐标，使用共享变量
    const int src_y = int((y - shared_pad_h) * src_h / (src_h * shared_scale));
    const int src_x = int((x - shared_pad_w) * src_w / (src_w * shared_scale));

    const int dst_idx = y * dst_w + x;

    if (src_y < 0 || src_y >= src_h || src_x < 0 || src_x >= src_w) {
        // padding: 一次性计算所有通道，减少循环开销
        dst[dst_idx] = (pad_value - mean[0]) * std[0];
        dst[dst_h * dst_w + dst_idx] = (pad_value - mean[1]) * std[1];
        dst[2 * dst_h * dst_w + dst_idx] = (pad_value - mean[2]) * std[2];
    }
    else {
        const int src_idx = src_y * src_step + src_x * 3;
        const float b = src[src_idx + 0];
        const float g = src[src_idx + 1];
        const float r = src[src_idx + 2];
        // 使用预计算的索引，减少重复计算
        dst[dst_idx] = (r - mean[0]) * std[0];
        dst[dst_h * dst_w + dst_idx] = (g - mean[1]) * std[1];
        dst[2 * dst_h * dst_w + dst_idx] = (b - mean[2]) * std[2];
    }
}

cudaError_t yolo_preproc(
    const uint8_t* src, int src_h, int src_w,
    float* dst, int dst_h, int dst_w, cudaStream_t stream,
    const float mean[3], const float std[3],
    const float pad_value,
    modeldeploy::vision::LetterBoxRecord* info_out) {
    // 使用更优的线程块配置
    dim3 block(16, 16); // 减少warp分歧，提高占用率
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);

    const size_t src_size = src_h * src_w * 3;
    uint8_t* d_src = nullptr;
    bool need_copy = true;

    // 优化指针属性检查
    cudaPointerAttributes attr{};
    if (cudaSuccess == cudaPointerGetAttributes(&attr, src) &&
        attr.type == cudaMemoryTypeDevice) {
        need_copy = false;
    }

    if (need_copy) {
        cudaMalloc(&d_src, src_size);
        cudaMemcpyAsync(d_src, src, src_size, cudaMemcpyHostToDevice, stream);
    }
    else {
        d_src = const_cast<uint8_t*>(src);
    }

    // 使用栈分配小内存，减少malloc调用
    float mean_std[6];
    for (int i = 0; i < 3; ++i) {
        mean_std[i] = mean[i];
        mean_std[i + 3] = std[i];
    }

    float* d_mean_std;
    cudaMalloc(&d_mean_std, 6 * sizeof(float));
    cudaMemcpyAsync(d_mean_std, mean_std, sizeof(float) * 6, cudaMemcpyHostToDevice, stream);

    float *d_scale, *d_pad_w, *d_pad_h;
    cudaMalloc(&d_scale, sizeof(float));
    cudaMalloc(&d_pad_w, sizeof(float));
    cudaMalloc(&d_pad_h, sizeof(float));

    kernel_yolo_preproc<<<grid, block, 0, stream>>>(
        d_src, src_h, src_w, src_w * 3,
        dst, dst_h, dst_w, d_mean_std, d_mean_std + 3, pad_value,
        d_scale, d_pad_w, d_pad_h);

    // 异步拷贝 letterbox 信息，减少同步等待
    float scale, pad_w, pad_h;
    cudaMemcpyAsync(&scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&pad_w, d_pad_w, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&pad_h, d_pad_h, sizeof(float), cudaMemcpyDeviceToHost, stream);

    // 仅在需要info_out时同步
    if (info_out) {
        cudaStreamSynchronize(stream);
    }

    if (info_out) {
        info_out->ipt_w = static_cast<float>(src_w);
        info_out->ipt_h = static_cast<float>(src_h);
        info_out->out_w = static_cast<float>(dst_w);
        info_out->out_h = static_cast<float>(dst_h);
        info_out->pad_w = pad_w;
        info_out->pad_h = pad_h;
        info_out->scale = scale;
    }

    // 清理 - 按分配逆序释放
    if (need_copy) cudaFree(d_src);
    cudaFree(d_mean_std);
    cudaFree(d_scale);
    cudaFree(d_pad_w);
    cudaFree(d_pad_h);
    return cudaGetLastError();
}

namespace modeldeploy::vision {
    bool yolo_preprocess_cuda(ImageData* image, Tensor* output, const std::vector<int>& dst_size,
                              const std::vector<float>& pad_val,
                              LetterBoxRecord* letter_box_record) {
        // 1. 获取原图信息
        const auto* input_ptr = static_cast<const uint8_t*>(image->data());
        const int src_h = image->height();
        const int src_w = image->width();
        int dst_h = dst_size[1];
        int dst_w = dst_size[0];
        // 2. 在 CPU 内存分配 output Tensor (float32, CHW)
        output->allocate({3, dst_h, dst_w}, DataType::FP32);
        auto* output_ptr_cpu = static_cast<float*>(output->data());
        // 3. 在 GPU 内存分配临时缓冲区
        float* d_output = nullptr;
        const size_t output_size = 3UL * dst_h * dst_w * sizeof(float);
        cudaError_t err = cudaMalloc(&d_output, output_size);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        // 4. 创建和管理CUDA流
        cudaStream_t stream;
        cudaError_t stream_err = cudaStreamCreate(&stream);
        if (stream_err != cudaSuccess) {
            std::cerr << "cudaStreamCreate failed: " << cudaGetErrorString(stream_err) << std::endl;
            cudaFree(d_output);
            return false;
        }
        // 5. 归一化参数和 pad_value
        constexpr float mean[3] = {0.0f, 0.0f, 0.0f};
        constexpr float std[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        const auto pad_value = pad_val[0];

        // 6. 调用 CUDA kernel
        err = yolo_preproc(
            input_ptr, src_h, src_w,
            d_output, dst_h, dst_w,
            stream,
            mean, std, pad_value,
            letter_box_record);
        if (err != cudaSuccess) {
            std::cerr << "yolo11n_preproc failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_output);
            return false;
        }
        // 7. cudaMemcpy 从 GPU 拷贝回 CPU Tensor
        err = cudaMemcpy(output_ptr_cpu, d_output, output_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_output);
            return false;
        }
        // 8. 释放 GPU 内存和流
        cudaFree(d_output);
        cudaStreamDestroy(stream);
        // 9. 添加 batch 维度
        output->expand_dim(0);
        return true;
    }
}
