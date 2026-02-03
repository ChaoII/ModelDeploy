//
// Created by aichao on 2026/1/26.
//

#include <opencv2/opencv.hpp>
#include "vision/utils.h"
#include "vision/common/processors/fusion_resize_pad_normalize_permute.cuh"

namespace modeldeploy::vision {
    __constant__ float c_alpha[3];
    __constant__ float c_beta[3];


    __global__ void kernel_fusion_resize_pad_normalize_permute(
        const uint8_t* src, // HWC
        const int src_b,
        const int* src_w_ptr,
        const int* src_h_ptr,
        float* dst, // CHW
        const int dst_w, // max_w
        const int dst_h, // max_h
        const int* resize_w_ptr,
        const int* resize_h_ptr,
        const float pad_value
    ) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        const int b = blockIdx.z; //batch
        if (x >= dst_w || y >= dst_h || b >= src_b) return;


        const int src_w_b = src_w_ptr[b];
        const int src_h_b = src_h_ptr[b];
        const int resize_w_b = resize_w_ptr[b];
        const int resize_h_b = resize_h_ptr[b];

        const int src_image_size = src_h_b * src_w_b * 3;
        const int dst_image_size = 3 * dst_w * dst_h;

        const uint8_t* src_ptr = src + b * src_image_size;
        float* dst_ptr = dst + b * dst_image_size;

        float* dst_c0 = dst_ptr + 0 * dst_w * dst_h;
        float* dst_c1 = dst_ptr + 1 * dst_w * dst_h;
        float* dst_c2 = dst_ptr + 2 * dst_w * dst_h;

        const int out_idx = y * dst_w + x;
        // pad（右 & 下）
        if (x >= resize_w_b || y >= resize_h_b) {
            dst_c0[out_idx] = pad_value * c_alpha[0] + c_beta[0];
            dst_c1[out_idx] = pad_value * c_alpha[1] + c_beta[1];
            dst_c2[out_idx] = pad_value * c_alpha[2] + c_beta[2];
            return;
        }
        const float scale_x = static_cast<float>(src_w_b) / resize_w_b;
        const float scale_y = static_cast<float>(src_h_b) / resize_h_b;

        const int sx = min(static_cast<int>(x * scale_x), src_w_b - 1);
        const int sy = min(static_cast<int>(y * scale_y), src_h_b - 1);

        const int src_idx = (sy * src_w_b + sx) * 3;

        const float channel_b = src_ptr[src_idx + 0];
        const float channel_g = src_ptr[src_idx + 1];
        const float channel_r = src_ptr[src_idx + 2];

        dst_c0[out_idx] = channel_r * c_alpha[0] + c_beta[0];
        dst_c1[out_idx] = channel_g * c_alpha[1] + c_beta[1];
        dst_c2[out_idx] = channel_b * c_alpha[2] + c_beta[2];
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
        const std::vector<ImageData>& images, Tensor* output,
        const std::vector<std::array<int, 2>>& resize_sizes,
        const std::vector<int>& dst_size,
        const std::vector<float>& mean,
        const std::vector<float>& std,
        const float pad_value) {
        const int batch_size = images.size();
        if (batch_size == 0) return false;

        // 1. 准备host数据
        std::vector<int> src_w(batch_size);
        std::vector<int> src_h(batch_size);
        std::vector<int> resize_w(batch_size);
        std::vector<int> resize_h(batch_size);

        for (int i = 0; i < batch_size; i++) {
            resize_w[i] = resize_sizes[i][0];
            resize_h[i] = resize_sizes[i][1];
            src_w[i] = images[i].width();
            src_h[i] = images[i].height();
        }

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

        cudaMemcpyToSymbol(c_alpha, alpha, sizeof(alpha));
        cudaMemcpyToSymbol(c_beta, beta, sizeof(beta));

        // 2. 创建CUDA stream
        cudaStream_t stream = nullptr;
        cudaStreamCreate(&stream);

        // 3. 分配GPU内存
        int* d_src_w = nullptr;
        int* d_src_h = nullptr;
        int* d_resize_w = nullptr;
        int* d_resize_h = nullptr;


        // 分配内存
        cudaMallocAsync(&d_src_w, batch_size * sizeof(int), stream);
        cudaMallocAsync(&d_src_h, batch_size * sizeof(int), stream);
        cudaMallocAsync(&d_resize_w, batch_size * sizeof(int), stream);
        cudaMallocAsync(&d_resize_h, batch_size * sizeof(int), stream);

        // 4. 拷贝数据到GPU
        cudaMemcpy(d_src_w, src_w.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_src_h, src_h.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_resize_w, resize_w.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_resize_h, resize_h.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);


        // 5. 分配输出tensor
        output->allocate({batch_size, 3, dst_h, dst_w}, DataType::FP32, Device::GPU);

        // 6. 处理输入数据
        Tensor input_tensor;
        ImageData::images_to_tensor(images, &input_tensor);
        const uint8_t* src = static_cast<const uint8_t*>(input_tensor.data());
        const size_t src_bytes = input_tensor.byte_size();

        // 检查是否已经在GPU上
        cudaPointerAttributes attr{};
        const bool is_device =
            cudaPointerGetAttributes(&attr, src) == cudaSuccess &&
            attr.type == cudaMemoryTypeDevice;

        const uint8_t* d_src = nullptr;
        if (is_device) {
            d_src = const_cast<uint8_t*>(src); // 已经在GPU
        }
        else {
            // 使用workspace
            if (ws.capacity < src_bytes) {
                if (ws.d_src) cudaFree(ws.d_src);
                cudaMalloc(&ws.d_src, src_bytes);
                ws.capacity = src_bytes;
            }
            cudaMemcpyAsync(ws.d_src, src, src_bytes, cudaMemcpyHostToDevice, stream);
            d_src = ws.d_src;
        }

        // 7. 启动kernel
        dim3 block(16, 16);
        dim3 grid(
            (dst_w + block.x - 1) / block.x,
            (dst_h + block.y - 1) / block.y,
            batch_size
        );

        kernel_fusion_resize_pad_normalize_permute<<<grid, block, 0, stream>>>(
            d_src,
            batch_size,
            d_src_w,
            d_src_h,
            static_cast<float*>(output->data()),
            dst_w,
            dst_h,
            d_resize_w,
            d_resize_h,
            pad_value
        );

        // 8. 同步和错误检查
        cudaStreamSynchronize(stream);

        // 9. 清理
        cudaFree(d_src_w);
        cudaFree(d_src_h);
        cudaFree(d_resize_w);
        cudaFree(d_resize_h);
        cudaStreamDestroy(stream);
        return cudaGetLastError() == cudaSuccess;
    }
}
