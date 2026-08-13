//
// Created by aichao on 2026/8/13.
//

#include "core/md_log.h"
#include "vision/sem/postprocessor.h"

#include <algorithm>
#include <cstring>

namespace modeldeploy::vision::detection {
    UltralyticsSemPostprocessor::UltralyticsSemPostprocessor() = default;

    bool UltralyticsSemPostprocessor::run(
        const std::vector<Tensor>& tensors, std::vector<SemSegResult>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        if (tensors.empty() || tensors[0].shape().size() != 4) {
            MD_LOG_ERROR << "Semantic segmentation requires 4D output [B,C,H,W]." << std::endl;
            return false;
        }
        const auto& shape = tensors[0].shape();
        const int64_t batch = shape[0];
        const int64_t channels = shape[1];
        const int64_t height = shape[2];
        const int64_t width = shape[3];
        if (tensors[0].dtype() != DataType::FP32) {
            MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
            return false;
        }
        results->resize(batch);
        const float* src = static_cast<const float*>(tensors[0].data());
        for (int64_t bs = 0; bs < batch; ++bs) {
            const float* plane = src + bs * channels * height * width;
            (*results)[bs].num_classes = static_cast<int32_t>(channels);
            // 先去除 letterbox padding：有效区对应原图 [0, ipt_h]x[0, ipt_w]
            const auto& rec = letter_box_records[bs];
            const float scale = rec.scale;
            const float pad_h = rec.pad_h;
            const float pad_w = rec.pad_w;
            const int64_t out_h = static_cast<int64_t>(rec.out_h);
            const int64_t out_w = static_cast<int64_t>(rec.out_w);
            // 特征图上的有效区（模型输出与输入同尺寸）
            int64_t x1 = static_cast<int64_t>(pad_w);
            int64_t y1 = static_cast<int64_t>(pad_h);
            int64_t x2 = out_w - static_cast<int64_t>(pad_w);
            int64_t y2 = out_h - static_cast<int64_t>(pad_h);
            x1 = std::max<int64_t>(0, x1);
            y1 = std::max<int64_t>(0, y1);
            x2 = std::min<int64_t>(width, x2);
            y2 = std::min<int64_t>(height, y2);
            const int64_t crop_w = x2 - x1;
            const int64_t crop_h = y2 - y1;
            if (crop_w <= 0 || crop_h <= 0) {
                MD_LOG_ERROR << "Invalid letterbox crop region." << std::endl;
                return false;
            }
            // 目标尺寸为原图尺寸（去 padding 后还原到原图）
            const int64_t orig_w = static_cast<int64_t>(rec.ipt_w);
            const int64_t orig_h = static_cast<int64_t>(rec.ipt_h);
            (*results)[bs].labels.assign(static_cast<size_t>(orig_h) * orig_w, 0);
            (*results)[bs].shape = {orig_h, orig_w};
            auto& labels = (*results)[bs].labels;
            // 逐像素 argmax（在裁剪区上）
            for (int64_t y = 0; y < crop_h; ++y) {
                const float* row_base = plane + (y1 + y) * width * channels + x1 * channels;
                uint8_t* dst_row = labels.data() + static_cast<size_t>(y) * orig_w;
                for (int64_t x = 0; x < crop_w; ++x) {
                    const float* px = row_base + x * channels;
                    int32_t best = 0;
                    float best_val = px[0];
                    for (int64_t c = 1; c < channels; ++c) {
                        if (px[c] > best_val) {
                            best_val = px[c];
                            best = static_cast<int32_t>(c);
                        }
                    }
                    dst_row[x] = static_cast<uint8_t>(best);
                }
            }
            // 裁剪区经 scale 还原到原图坐标；由于是 letterbox 缩放，有效区正好对应原图整幅
            // （有效区尺寸 crop_w/crop_h 与 orig_w/orig_h 可能因取整差 1，做 resize 对齐）
            if (crop_w != orig_w || crop_h != orig_h) {
                std::vector<uint8_t> resized(static_cast<size_t>(orig_h) * orig_w);
                // 用最近邻缩放（保持类别边界清晰）
                for (int64_t y = 0; y < orig_h; ++y) {
                    int64_t sy = std::min<int64_t>(crop_h - 1, static_cast<int64_t>(y * crop_h / orig_h));
                    const uint8_t* srow = labels.data() + static_cast<size_t>(sy) * orig_w;
                    for (int64_t x = 0; x < orig_w; ++x) {
                        int64_t sx = std::min<int64_t>(crop_w - 1, static_cast<int64_t>(x * crop_w / orig_w));
                        resized[static_cast<size_t>(y) * orig_w + x] = srow[x];
                    }
                }
                labels.swap(resized);
            }
        }
        return true;
    }
}
