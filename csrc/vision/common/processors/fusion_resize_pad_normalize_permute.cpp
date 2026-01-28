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
        const int src_w,
        const int src_h,
        float* dst, // CHW
        const int dst_w, // max_w
        const int dst_h, // max_h
        const int resize_w,
        const int resize_h,
        const float alpha[3],
        const float beta[3],
        const float pad_value
    ) {
        float* dst_b = dst + 0 * dst_w * dst_h;
        float* dst_g = dst + 1 * dst_w * dst_h;
        float* dst_r = dst + 2 * dst_w * dst_h;

        const float scale_x = static_cast<float>(src_w) / resize_w;
        const float scale_y = static_cast<float>(src_h) / resize_h;

        for (int dy = 0; dy < dst_h; ++dy) {
            for (int dx = 0; dx < dst_w; ++dx) {
                const int out_idx = dy * dst_w + dx;
                // pad（右 & 下）
                if (dx >= resize_w || dy >= resize_h) {
                    dst_b[out_idx] = pad_value * alpha[2] + beta[1];
                    dst_g[out_idx] = pad_value * alpha[1] + beta[1];
                    dst_r[out_idx] = pad_value * alpha[0] + beta[0];
                    continue;
                }
                const int sx = std::min(static_cast<int>(dx * scale_x), src_w - 1);
                const int sy = std::min(static_cast<int>(dy * scale_y), src_h - 1);
                const int src_idx = (sy * src_w + sx) * 3;
                const float b = src[src_idx + 0];
                const float g = src[src_idx + 1];
                const float r = src[src_idx + 2];
                // 做了swap
                dst_b[out_idx] = r * alpha[0] + beta[0];
                dst_g[out_idx] = g * alpha[1] + beta[1];
                dst_r[out_idx] = b * alpha[2] + beta[2];
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

        output->allocate({3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        const uint8_t* src = image.data();
        float* dst = output->data_ptr<float>();
        fusion_resize_pad_normalize_permute_native(
            src, // HWC
            src_w,
            src_h,
            dst, // CHW
            dst_w, // max_w
            dst_h, // max_h
            resize_w,
            resize_h,
            alpha,
            beta,
            pad_value
        );
        output->expand_dim(0);
        return true;
    }
}
