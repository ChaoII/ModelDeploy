//
// Created by aichao on 2025/7/22.
//

#include <opencv2/opencv.hpp>
#include "vision/utils.h"
#include "vision/common/struct.h"
#include "vision/common/processors/yolo_preproc.h"
#include "vision/common/processors/convert_and_permute.h"


namespace modeldeploy::vision {
    constexpr float inv_255 = 1.0f / 255.0f;

    void yolo_preprocess_bgr_cpu_inner(
        const uint8_t* src,
        const int src_w,
        const int src_h,
        float* dst,
        const int dst_w,
        const int dst_h,
        const float scale,
        const float pad_w,
        const float pad_h,
        const float pad_value
    ) {
        const int plane_size = dst_w * dst_h;


        const float pad_x_end = pad_w + static_cast<float>(src_w) * scale;
        const float pad_y_end = pad_h + static_cast<float>(src_h) * scale;
        const float pad_f = pad_value * inv_255;

        for (int dy = 0; dy < dst_h; ++dy) {
            for (int dx = 0; dx < dst_w; ++dx) {
                const int out_idx = dy * dst_w + dx;
                // padding
                if (dx < pad_w || dx >= pad_x_end ||
                    dy < pad_h || dy >= pad_y_end) {
                    dst[0 * plane_size + out_idx] = pad_f;
                    dst[1 * plane_size + out_idx] = pad_f;
                    dst[2 * plane_size + out_idx] = pad_f;
                    continue;
                }

                const int sx = static_cast<int>((dx - pad_w) / scale);
                const int sy = static_cast<int>((dy - pad_h) / scale);

                const int src_idx = (sy * src_w + sx) * 3;
                float c0 = src[src_idx + 0] * inv_255;
                float c1 = src[src_idx + 1] * inv_255;
                float c2 = src[src_idx + 2] * inv_255;

                std::swap(c0, c2);

                dst[0 * plane_size + out_idx] = c0;
                dst[1 * plane_size + out_idx] = c1;
                dst[2 * plane_size + out_idx] = c2;
            }
        }
    }


    void yolo_preprocess_nv12_cpu_inner(
        const uint8_t* src_y,
        const uint8_t* src_uv,
        const int src_w,
        const int src_h,
        const int step_y,
        const int step_uv,
        float* dst,
        const int dst_w,
        const int dst_h,
        const float scale,
        const float pad_w,
        const float pad_h,
        const float pad_value) {
        const int plane_size = dst_h * dst_w;

        for (int dy = 0; dy < dst_h; dy++) {
            for (int dx = 0; dx < dst_w; dx++) {
                // 反推源图像坐标 (最近邻插值)
                // dst_coord = pad + src_coord * scale  =>  src_coord = (dst_coord - pad) / scale
                const float src_xf = (dx - pad_w) / scale;
                const float src_yf = (dy - pad_h) / scale;
                const int src_cori_x = static_cast<int>(src_xf);
                const int src_cori_y = static_cast<int>(src_yf);

                const int dst_idx = dy * dst_w + dx;
                // 判断是否在有效图像区域内 (Letterbox 逻辑)
                if (src_cori_x < 0 || src_cori_x >= src_w || src_cori_y < 0 || src_cori_y >= src_h) {
                    // --- 填充区域 ---
                    const float v0 = pad_value * inv_255;
                    const float v1 = pad_value * inv_255;
                    const float v2 = pad_value * inv_255;
                    // 写入 CHW: C0(R), C1(G), C2(B)
                    dst[0 * plane_size + dst_idx] = v0;
                    dst[1 * plane_size + dst_idx] = v1;
                    dst[2 * plane_size + dst_idx] = v2;
                }
                else {
                    // --- 有效图像区域 ---

                    // 1. 采样 Y 分量
                    const float y_val = src_y[src_cori_y * step_y + src_cori_x];

                    // 2. 采样 U, V 分量
                    // NV12: UV 平面分辨率是 Y 的一半，且交错存储 (U0, V0, U1, V1...)
                    const int uv_x = src_cori_x >> 1; // src_x / 2
                    const int uv_y = src_cori_y >> 1; // src_y / 2

                    // 边界保护 (防止 src_w/src_h 为奇数时越界)
                    const int safe_uv_x = std::min(uv_x, (src_w >> 1) - 1);
                    const int safe_uv_y = std::min(uv_y, (src_h >> 1) - 1);

                    const uint8_t* uv_row = src_uv + safe_uv_y * step_uv;
                    float u_val = uv_row[safe_uv_x * 2 + 0]; // U
                    float v_val = uv_row[safe_uv_x * 2 + 1]; // V

                    // 3. YUV -> RGB 转换 (BT.601 标准)
                    // Y 范围 0-255, UV 范围 0-255 (中心 128)
                    u_val = u_val - 128.0f;
                    v_val = v_val - 128.0f;

                    // 计算 RGB (结果范围 0-255)
                    float r = y_val + 1.402f * v_val;
                    float g = y_val - 0.344136f * u_val - 0.714136f * v_val;
                    float b = y_val + 1.772f * u_val;

                    // clip到 [0, 255] 防止溢出
                    r = fminf(fmaxf(r, 0.0f), 255.0f);
                    g = fminf(fmaxf(g, 0.0f), 255.0f);
                    b = fminf(fmaxf(b, 0.0f), 255.0f);

                    // 4. 归一化/标准化 并写入 CHW
                    dst[0 * plane_size + dst_idx] = r * inv_255; // R -> C0
                    dst[1 * plane_size + dst_idx] = g * inv_255; // G -> C1
                    dst[2 * plane_size + dst_idx] = b * inv_255; // B -> C2
                }
            }
        }
    }


    void yolo_preprocess_cpu_batch(
        const uint8_t* src,
        const int src_b,
        const int* src_ws_ptr,
        const int* src_hs_ptr,
        float* dst,
        const int dst_w,
        const int dst_h,
        const float* scales_ptr,
        const float* pad_ws_ptr,
        const float* pad_hs_ptr,
        const bool swap_rb,
        const float pad_val
    ) {
        const int dst_image_size = 3 * dst_h * dst_w;
        const int dst_hw = dst_w * dst_h;
        constexpr float inv_255 = 1.0f / 255.0f;
        for (int b = 0; b < src_b; b++) {
            const int src_w = src_ws_ptr[b];
            const int src_h = src_hs_ptr[b];
            const float scale = scales_ptr[b];
            const float pad_w = pad_ws_ptr[b];
            const float pad_h = pad_hs_ptr[b];
            const int src_image_size = src_h * src_w * 3;
            const uint8_t* src_ptr = src + b * src_image_size;
            float* dst_ptr = dst + b * dst_image_size;

            float* dst_c0 = dst_ptr + 0 * dst_hw;
            float* dst_c1 = dst_ptr + 1 * dst_hw;
            float* dst_c2 = dst_ptr + 2 * dst_hw;

            const float pad_x_end = pad_w + static_cast<float>(src_w) * scale;
            const float pad_y_end = pad_h + static_cast<float>(src_h) * scale;
            const float pad_f = pad_val * inv_255;

            for (int dy = 0; dy < dst_h; ++dy) {
                for (int dx = 0; dx < dst_w; ++dx) {
                    const int out_idx = dy * dst_w + dx;
                    // padding
                    if (dx < pad_w || dx >= pad_x_end ||
                        dy < pad_h || dy >= pad_y_end) {
                        dst_c0[out_idx] = pad_f;
                        dst_c1[out_idx] = pad_f;
                        dst_c2[out_idx] = pad_f;
                        continue;
                    }

                    const int sx = static_cast<int>((dx - pad_w) / scale);
                    const int sy = static_cast<int>((dy - pad_h) / scale);

                    const int src_idx = (sy * src_w + sx) * 3;
                    float c0 = src_ptr[src_idx + 0] * inv_255;
                    float c1 = src_ptr[src_idx + 1] * inv_255;
                    float c2 = src_ptr[src_idx + 2] * inv_255;

                    if (swap_rb) std::swap(c0, c2);
                    dst_c0[out_idx] = c0;
                    dst_c1[out_idx] = c1;
                    dst_c2[out_idx] = c2;
                }
            }
        }
    }


    bool yolo_preprocess_cpu(const ImageData& image,
                             Tensor* output,
                             const std::vector<int>& dst_size,
                             const float pad_val,
                             LetterBoxRecord* letter_box_record) {
        return yolo_preprocess_bgr_cpu(image.data(),
                                       {image.width(), image.height()},
                                       output,
                                       dst_size,
                                       pad_val,
                                       letter_box_record);
    }


    bool yolo_preprocess_bgr_cpu(const uint8_t* src,
                                 const std::vector<int>& src_size,
                                 Tensor* output,
                                 const std::vector<int>& dst_size,
                                 float pad_val,
                                 LetterBoxRecord* letter_box_record) {
        if (src == nullptr) return false;

        const int src_w = src_size[0];
        const int src_h = src_size[1];
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];

        output->allocate({3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        *letter_box_record = utils::cal_letter_box_param({src_w, src_h}, {dst_w, dst_h});
        const float scale = letter_box_record->scale;
        const float pad_w = letter_box_record->pad_w;
        const float pad_h = letter_box_record->pad_h;
        auto* dst = output->data_ptr<float>();
        yolo_preprocess_bgr_cpu_inner(src, src_w, src_h, dst, dst_w, dst_h, scale, pad_w, pad_h, pad_val);
        output->expand_dim(0);
        return true;
    }

    bool yolo_preprocess_nv12_cpu(const uint8_t* src_y,
                                  const uint8_t* src_uv,
                                  const std::vector<int>& src_size,
                                  int step_y,
                                  int step_uv,
                                  Tensor* output,
                                  const std::vector<int>& dst_size,
                                  float pad_value,
                                  LetterBoxRecord* letter_box_record) {
        if (src_y == nullptr || src_uv == nullptr) return false;

        const int src_w = src_size[0];
        const int src_h = src_size[1];
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];

        output->allocate({3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        *letter_box_record = utils::cal_letter_box_param({src_w, src_h}, {dst_w, dst_h});
        const float scale = letter_box_record->scale;
        const float pad_w = letter_box_record->pad_w;
        const float pad_h = letter_box_record->pad_h;
        auto* dst = output->data_ptr<float>();
        yolo_preprocess_nv12_cpu_inner(src_y,
                                       src_uv,
                                       src_w,
                                       src_h,
                                       step_y,
                                       step_uv,
                                       dst,
                                       dst_w,
                                       dst_h,
                                       scale,
                                       pad_w,
                                       pad_h,
                                       pad_value);
        output->expand_dim(0);
        return true;
    }


    bool yolo_preprocess_cpu_batch(const std::vector<ImageData>& images, Tensor* output,
                                   const std::vector<int>& dst_size,
                                   const float pad_val,
                                   std::vector<LetterBoxRecord>* letter_box_record) {
        const int batch_size = images.size();
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];

        letter_box_record->resize(batch_size);

        std::vector<float> scales(batch_size);
        std::vector<float> pad_ws(batch_size);
        std::vector<float> pad_hs(batch_size);
        std::vector<int> src_ws(batch_size);
        std::vector<int> src_hs(batch_size);

        size_t batch_image_bytes = 0;

        for (const auto& image : images) {
            batch_image_bytes += image.bytes();
        }

        std::vector<uint8_t> batch_image_buffer(batch_image_bytes);

        size_t offset = 0;
        for (int i = 0; i < images.size(); i++) {
            const int src_h = images[i].height();
            const int src_w = images[i].width();
            letter_box_record->at(i) = utils::cal_letter_box_param({src_w, src_h}, {dst_w, dst_h});
            scales.push_back(letter_box_record->at(i).scale);
            pad_ws.push_back(letter_box_record->at(i).pad_w);
            pad_hs.push_back(letter_box_record->at(i).pad_h);
            src_ws.push_back(src_w);
            src_hs.push_back(src_h);
            std::memcpy(batch_image_buffer.data() + offset, images[i].data(), images[i].bytes());
            offset += images[i].bytes();
        }

        const float* scales_ptr = scales.data();
        const float* pad_ws_ptr = pad_ws.data();
        const float* pad_hs_ptr = pad_hs.data();
        const int* src_ws_ptr = src_ws.data();
        const int* src_hs_ptr = src_hs.data();


        output->allocate({batch_size, 3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        auto* dst = output->data_ptr<float>();

        yolo_preprocess_cpu_batch(
            batch_image_buffer.data(),
            batch_size,
            src_ws_ptr,
            src_hs_ptr, dst,
            dst_w, dst_h,
            scales_ptr,
            pad_ws_ptr,
            pad_hs_ptr,
            true,
            pad_val);
        return true;
    }


    bool yolo_preprocess_cpu_ocv(const ImageData& image, Tensor* output,
                                 const std::vector<int>& dst_size,
                                 const float pad_val,
                                 LetterBoxRecord* letter_box_record) {
        // yolo's preprocess steps
        // 1. letterbox
        // 2. convert_and_permute(swap_rb=true)
        *letter_box_record = utils::cal_letter_box_param({image.width(), image.height()}, dst_size);
        auto s = image.letter_box(dst_size, pad_val).fuse_convert_and_permute();
        // s = s.normalize({1/255.0f, 1/255.0f, 1/255.0f}, {0.0f, 0.0f, 0.0f});
        // s.imshow("letter_box");
        // s = s.permute();
        s.to_tensor(output);
        output->expand_dim(0); // reshape to n, c, h, w
        return true;
    }
}
