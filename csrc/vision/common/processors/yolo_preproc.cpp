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

                const uint8_t* p = src + (sy * src_w + sx) * 3;

                float c0 = p[0] * inv_255;
                float c1 = p[1] * inv_255;
                float c2 = p[2] * inv_255;

                if (swap_rb) std::swap(c0, c2);

                dst_c0[out_idx] = c0;
                dst_c1[out_idx] = c1;
                dst_c2[out_idx] = c2;
            }
        }
    }


    bool yolo_preprocess_cpu(const ImageData& image, Tensor* output,
                             const std::vector<int>& dst_size,
                             const std::vector<float>& pad_val,
                             LetterBoxRecord* letter_box_record) {
        const int src_h = image.height();
        const int src_w = image.width();
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];
        const float pad_value = pad_val[0];
        output->allocate({3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        *letter_box_record = utils::cal_letter_box_param({src_w, src_h}, {dst_w, dst_h});
        const float scale = letter_box_record->scale;
        const float pad_w = letter_box_record->pad_w;
        const float pad_h = letter_box_record->pad_h;
        const uint8_t* src = image.data();
        float* dst = output->data_ptr<float>();
        yolo11_preprocess_cpu(src, src_w, src_h, dst, dst_w, dst_h, scale, pad_w, pad_h, true, pad_value);
        output->expand_dim(0);
        return true;
    }


    bool yolo_preprocess_cpu1(const ImageData& image, Tensor* output,
                              const std::vector<int>& dst_size,
                              const std::vector<float>& pad_val,
                              LetterBoxRecord* letter_box_record) {
        // yolo's preprocess steps
        // 1. letterbox
        // 2. convert_and_permute(swap_rb=true)
        *letter_box_record = utils::cal_letter_box_param({image.width(), image.height()}, dst_size);
        auto s = image.letter_box(dst_size, 114.0f).fuse_convert_and_permute();
        // s = s.normalize({1/255.0f, 1/255.0f, 1/255.0f}, {0.0f, 0.0f, 0.0f});
        // s.imshow("letter_box");
        // s = s.permute();
        s.to_tensor(output);
        output->expand_dim(0); // reshape to n, c, h, w
        // 这里不能深拷贝，深拷贝cv::Mat mat持有的内存就会被释放那么output指向的内存也会被释放
        // image->update_from_mat(&mat);
        // *image = ImageData::from_mat(&mat);
        return true;
    }
}
