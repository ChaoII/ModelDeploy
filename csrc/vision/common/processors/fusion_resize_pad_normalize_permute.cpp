//
// Created by aichao on 2025/7/22.
//

#include <opencv2/opencv.hpp>
#include "vision/utils.h"
#include "vision/common/struct.h"
#include "vision/common/processors/fusion_resize_pad_normalize_permute.h"
#include "vision/common/processors/convert_and_permute.h"


namespace modeldeploy::vision {
    void fusion_resize_pad_normalize_permute_native(
        const uint8_t* src, // HWC
        const int src_b,
        const int* src_w_ptr,
        const int* src_h_ptr,
        float* dst, // CHW
        const int dst_w, // max_w
        const int dst_h, // max_h
        const int* resize_w_ptr,
        const int* resize_h_ptr,
        const float alpha[3],
        const float beta[3],
        const float pad_value
    ) {
        const int dst_image_size = 3 * dst_h * dst_w;
        for (int b = 0; b < src_b; b++) {
            const int src_w_b = src_w_ptr[b];
            const int src_h_b = src_h_ptr[b];
            const int resize_w_b = resize_w_ptr[b];
            const int resize_h_b = resize_h_ptr[b];
            const int src_image_size = src_h_b * src_w_b * 3;

            const uint8_t* src_ptr = src + b * src_image_size;
            float* dst_ptr = dst + b * dst_image_size;

            float* dst_b = dst_ptr + 0 * dst_w * dst_h;
            float* dst_g = dst_ptr + 1 * dst_w * dst_h;
            float* dst_r = dst_ptr + 2 * dst_w * dst_h;

            const float scale_x = static_cast<float>(src_w_b) / resize_w_b;
            const float scale_y = static_cast<float>(src_h_b) / resize_h_b;

            for (int dy = 0; dy < dst_h; ++dy) {
                for (int dx = 0; dx < dst_w; ++dx) {
                    const int out_idx = dy * dst_w + dx;
                    // pad（右 & 下）
                    if (dx >= resize_w_b || dy >= resize_h_b) {
                        dst_b[out_idx] = pad_value * alpha[2] + beta[1];
                        dst_g[out_idx] = pad_value * alpha[1] + beta[1];
                        dst_r[out_idx] = pad_value * alpha[0] + beta[0];
                        continue;
                    }
                    const int sx = std::min(static_cast<int>(dx * scale_x), src_w_b - 1);
                    const int sy = std::min(static_cast<int>(dy * scale_y), src_h_b - 1);
                    const int src_idx = (sy * src_w_b + sx) * 3;
                    const float channel_b = src_ptr[src_idx + 0];
                    const float channel_g = src_ptr[src_idx + 1];
                    const float channel_r = src_ptr[src_idx + 2];
                    // 做了swap
                    dst_b[out_idx] = channel_r * alpha[0] + beta[0];
                    dst_g[out_idx] = channel_g * alpha[1] + beta[1];
                    dst_r[out_idx] = channel_b * alpha[2] + beta[2];
                }
            }
        }
    }


    bool fusion_resize_pad_normalize_permute_ocv(const ImageData& image, Tensor* output,
                                                 const std::vector<int>& resize_size,
                                                 int pad_top,
                                                 int pad_bottom,
                                                 int pad_left,
                                                 int pad_right,
                                                 std::vector<float> mean,
                                                 std::vector<float> std,
                                                 float pad_value) {
        // yolo's preprocess steps
        // 1. letterbox
        // 2. convert_and_permute(swap_rb=true)
        image.fuse_resize_and_pad(resize_size[0],
                                  resize_size[1],
                                  pad_right,
                                  pad_bottom,
                                  pad_value)
             .fuse_normalize_and_permute(mean, std).to_tensor(output);
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }

    bool fusion_resize_pad_normalize_permute_cpu(
        const std::vector<ImageData>& images, Tensor* output,
        const std::vector<std::array<int, 2>>& resize_sizes,
        const std::vector<int>& dst_size,
        const std::vector<float>& mean,
        const std::vector<float>& std,
        const float pad_value) {
        const int batch_size = images.size();
        if (batch_size == 0) return false;

        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];

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

        const int* src_w_ptr = src_w.data();
        const int* src_h_ptr = src_h.data();
        const int* resize_w_ptr = resize_w.data();
        const int* resize_h_ptr = resize_h.data();


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

        Tensor input_tensor;
        ImageData::images_to_tensor(images, &input_tensor);
        output->allocate({batch_size, 3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        const uint8_t* src = static_cast<uint8_t*>(input_tensor.data());
        float* dst = output->data_ptr<float>();
        fusion_resize_pad_normalize_permute_native(
            src, // HWC
            batch_size,
            src_w_ptr,
            src_h_ptr,
            dst, // CHW
            dst_w, // max_w
            dst_h, // max_h
            resize_w_ptr,
            resize_h_ptr,
            alpha,
            beta,
            pad_value
        );
        return true;
    }
}
