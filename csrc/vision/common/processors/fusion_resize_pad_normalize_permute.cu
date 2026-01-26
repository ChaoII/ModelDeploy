//
// Created by aichao on 2026/1/26.
//

#include <opencv2/opencv.hpp>
#include "vision/utils.h"
#include "vision/common/processors/fusion_resize_pad_normalize_permute.cuh"

namespace modeldeploy::vision {
    __global__ void kernel_fusion_resize_pad_normalize_permute(
        const uint8_t* src, // HWC
        const int src_w,
        const int src_h,
        float* dst, // CHW
        const int dst_w, // max_w
        const int dst_h, // max_h
        const int resize_w,
        const int resize_h,
        const float alpha0, const float alpha1, const float alpha2,
        const float beta0, const float beta1, const float beta2,
        const float pad_value
    ) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= dst_w || y >= dst_h) return;



        float* dst_b = dst + 0 * dst_w * dst_h;
        float* dst_g = dst + 1 * dst_w * dst_h;
        float* dst_r = dst + 2 * dst_w * dst_h;

        const int out_idx = y * dst_w + x;
        // pad（右 & 下）
        if (x >= resize_w || y >= resize_h) {
            dst_b[out_idx] = pad_value * alpha0 + beta0;
            dst_g[out_idx] = pad_value * alpha1 + beta1;
            dst_r[out_idx] = pad_value * alpha2 + beta2;
            return;
        }
        const float scale_x = static_cast<float>(src_w) / resize_w;
        const float scale_y = static_cast<float>(src_h) / resize_h;

        const int sx = min(static_cast<int>(x * scale_x), src_w - 1);
        const int sy = min(static_cast<int>(y * scale_y), src_h - 1);

        const int src_idx = (sy * src_w + sx) * 3;
        const float b = src[src_idx + 0];
        const float g = src[src_idx + 1];
        const float r = src[src_idx + 2];
        dst_b[out_idx] = r * alpha0 + beta0;
        dst_g[out_idx] = g * alpha1 + beta1;
        dst_r[out_idx] = b * alpha2 + beta2;
    }

    struct PreprocWorkspace {
        uint8_t* d_src = nullptr;
        size_t capacity = 0;

        ~PreprocWorkspace() {
            if (d_src) {
                cudaFree(d_src);
            }
        }
    };

    static thread_local PreprocWorkspace ws;

    bool fusion_resize_pad_normalize_permute_cuda(
        const ImageData& image, Tensor* output,
        const std::vector<int>& resize_size,
        const std::vector<int>& dst_size,
        const std::vector<float>& mean,
        const std::vector<float>& std,
        const float pad_value) {
        const int src_w = image.width();
        const int src_h = image.height();
        const int resize_w = resize_size[0];
        const int resize_h = resize_size[1];
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];

        const float alpha[3] = {
            1.0f / 255.0f / std[0],
            1.0f / 255.0f / std[1],
            1.0f / 255.0f / std[2]
        };
        const float beta[3] = {
            -mean[0] / std[0],
            -mean[1] / std[1],
            -mean[2] / std[2]
        };

        // 1 allocate output
        output->allocate({3, dst_h, dst_w}, DataType::FP32, Device::GPU);

        // 2 CUDA stream
        cudaStream_t stream;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            return false;
        }

        // 4 src 指针处理（workspace，避免反复 malloc）
        const uint8_t* src = image.data();
        const size_t src_bytes = static_cast<size_t>(src_h) * src_w * image.channels();
        const uint8_t* d_src = nullptr;
        cudaPointerAttributes attr{};
        const bool is_device =
            cudaPointerGetAttributes(&attr, src) == cudaSuccess && attr.type == cudaMemoryTypeDevice;
        if (is_device) {
            d_src = src;
        }
        else {
            if (ws.capacity < src_bytes) {
                if (ws.d_src) cudaFree(ws.d_src);
                cudaMalloc(&ws.d_src, src_bytes);
                ws.capacity = src_bytes;
            }
            cudaMemcpyAsync(ws.d_src, src, src_bytes, cudaMemcpyHostToDevice, stream);
            d_src = ws.d_src;
        }
        // 5 launch kernel
        dim3 block(16, 16);
        dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

        kernel_fusion_resize_pad_normalize_permute<<<grid, block, 0, stream>>>(
            d_src,
            src_w,
            src_h,
            output->data_ptr<float>(),
            dst_w,
            dst_h,
            resize_w,
            resize_h,
            alpha[0], alpha[1], alpha[2],
            beta[0], beta[1], beta[2],
            pad_value);
        const cudaError_t err = cudaGetLastError();
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            return false;
        }
        // 6 增加batch维
        output->expand_dim(0);
        return true;
    }
}
