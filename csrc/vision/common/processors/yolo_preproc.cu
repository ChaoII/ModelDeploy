//
// Created by aichao on 2025/7/22.
//

#include "vision/common/processors/yolo_preproc.cuh"
#include <cuda_fp16.h>

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

    // 计算缩放比例和padding
    const float scale = fminf(float(dst_h) / src_h, float(dst_w) / src_w);
    const int new_h = int(src_h * scale + 0.5f);
    const int new_w = int(src_w * scale + 0.5f);
    const int pad_h = (dst_h - new_h) / 2;
    const int pad_w = (dst_w - new_w) / 2;

    // 写缩放信息（一个线程即可）
    if (x == 0 && y == 0) {
        *scale_out = scale;
        *pad_h_out = float(pad_h);
        *pad_w_out = float(pad_w);
    }

    // 目标坐标 → 源坐标
    const int src_y = (y - pad_h) * src_h / new_h;
    const int src_x = (x - pad_w) * src_w / new_w;

    if (src_y < 0 || src_y >= src_h || src_x < 0 || src_x >= src_w) {
        for (int c = 0; c < 3; ++c) {
            dst[c * dst_h * dst_w + y * dst_w + x] = (pad_value - mean[c]) * std[c];
        }
    }
    else {
        const int src_idx = src_y * src_step + src_x * 3;
        const float b = src[src_idx + 0];
        const float g = src[src_idx + 1];
        const float r = src[src_idx + 2];
        dst[0 * dst_h * dst_w + y * dst_w + x] = (r - mean[0]) * std[0];
        dst[1 * dst_h * dst_w + y * dst_w + x] = (g - mean[1]) * std[1];
        dst[2 * dst_h * dst_w + y * dst_w + x] = (b - mean[2]) * std[2];
    }
}

cudaError_t yolo_preproc(
    const uint8_t* src, int src_h, int src_w,
    float* dst, int dst_h, int dst_w, cudaStream_t stream,
    const float mean[3], const float std[3],
    const float pad_value,
    modeldeploy::vision::LetterBoxRecord* info_out) {
    dim3 block(32, 32);
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);

    const size_t src_size = src_h * src_w * 3;
    uint8_t* d_src = nullptr;
    bool need_copy = true;

    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, src) == cudaSuccess) {
        if (attr.type == cudaMemoryTypeDevice) need_copy = false;
    }

    if (need_copy) {
        cudaMalloc(&d_src, src_size);
        cudaMemcpyAsync(d_src, src, src_size, cudaMemcpyHostToDevice, stream);
    }
    else {
        d_src = const_cast<uint8_t*>(src);
    }

    // 分配辅助参数内存
    float *d_mean, *d_std, *d_scale, *d_pad_w, *d_pad_h;
    cudaMalloc(&d_mean, 3 * sizeof(float));
    cudaMalloc(&d_std, 3 * sizeof(float));
    cudaMalloc(&d_scale, sizeof(float));
    cudaMalloc(&d_pad_w, sizeof(float));
    cudaMalloc(&d_pad_h, sizeof(float));

    cudaMemcpyAsync(d_mean, mean, sizeof(float) * 3, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_std, std, sizeof(float) * 3, cudaMemcpyHostToDevice, stream);

    kernel_yolo_preproc<<<grid, block, 0, stream>>>(
        d_src, src_h, src_w, src_w * 3,
        dst, dst_h, dst_w, d_mean, d_std, pad_value,
        d_scale, d_pad_w, d_pad_h);

    // 拷贝 letterbox 信息
    float scale, pad_w, pad_h;
    cudaMemcpyAsync(&scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&pad_w, d_pad_w, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&pad_h, d_pad_h, sizeof(float), cudaMemcpyDeviceToHost, stream);

    // 等待 stream 完成后填充 info_out
    cudaStreamSynchronize(stream);

    if (info_out) {
        info_out->ipt_w = static_cast<float>(src_w);
        info_out->ipt_h = static_cast<float>(src_h);
        info_out->out_w = static_cast<float>(dst_w);
        info_out->out_h = static_cast<float>(dst_h);
        info_out->pad_w = pad_w;
        info_out->pad_h = pad_h;
        info_out->scale = scale;
    }

    // 清理
    if (need_copy) cudaFree(d_src);
    cudaFree(d_mean);
    cudaFree(d_std);
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
        size_t output_size = 3UL * dst_h * dst_w * sizeof(float);
        cudaError_t err = cudaMalloc(&d_output, output_size);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        // 4. CUDA 流
        cudaStream_t stream = nullptr;
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
        // 8. 释放 GPU 内存
        cudaFree(d_output);
        // 9. 添加 batch 维度
        output->expand_dim(0);
        return true;
    }
}
