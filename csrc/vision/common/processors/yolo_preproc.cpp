//
// Created by aichao on 2025/7/22.
//

#include <opencv2/opencv.hpp>
#include "vision/utils.h"
#include "vision/common/struct.h"
#include "vision/common/processors/yolo_preproc.h"
#include "vision/common/processors/convert_and_permute.h"


namespace modeldeploy::vision {
    void yolo11_preprocess_cpu(
        const uint8_t* src,
        const int src_w,
        const int src_h,
        float* dst,
        const int dst_w,
        const int dst_h,
        const float scale,
        const float pad_w,
        const float pad_h,
        const bool swap_rb,
        const float pad_value
    ) {
        const int hw = dst_w * dst_h;
        constexpr float inv_255 = 1.0f / 255.0f;

        float* dst_c0 = dst + 0 * hw;
        float* dst_c1 = dst + 1 * hw;
        float* dst_c2 = dst + 2 * hw;

        const float pad_x_end = pad_w + src_w * scale;
        const float pad_y_end = pad_h + src_h * scale;
        const float pad_f = pad_value * inv_255;

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
                float c0 = src[src_idx + 0] * inv_255;
                float c1 = src[src_idx + 1] * inv_255;
                float c2 = src[src_idx + 2] * inv_255;

                if (swap_rb) std::swap(c0, c2);

                dst_c0[out_idx] = c0;
                dst_c1[out_idx] = c1;
                dst_c2[out_idx] = c2;
            }
        }
    }


    void yolo11_preprocess_cpu_batch(
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

            const float pad_x_end = pad_w + src_w * scale;
            const float pad_y_end = pad_h + src_h * scale;
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


    bool yolo_preprocess_cpu(const ImageData& image, Tensor* output,
                             const std::vector<int>& dst_size,
                             const float pad_val,
                             LetterBoxRecord* letter_box_record) {
        const int src_h = image.height();
        const int src_w = image.width();
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];

        output->allocate({3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        *letter_box_record = utils::cal_letter_box_param({src_w, src_h}, {dst_w, dst_h});
        const float scale = letter_box_record->scale;
        const float pad_w = letter_box_record->pad_w;
        const float pad_h = letter_box_record->pad_h;
        const uint8_t* src = image.data();
        auto* dst = output->data_ptr<float>();
        yolo11_preprocess_cpu(src, src_w, src_h, dst, dst_w, dst_h, scale, pad_w, pad_h, true, pad_val);
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

        yolo11_preprocess_cpu_batch(
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
