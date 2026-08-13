//
// Created by aichao on 2026/8/13.
//

#include "core/md_log.h"
#include "vision/depth/postprocessor.h"

#include <algorithm>
#include <cstring>

namespace modeldeploy::vision::detection {
    UltralyticsDepthPostprocessor::UltralyticsDepthPostprocessor() = default;

    bool UltralyticsDepthPostprocessor::run(
        const std::vector<Tensor>& tensors, std::vector<DepthResult>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        if (tensors.empty() || tensors[0].shape().size() != 4) {
            MD_LOG_ERROR << "Depth estimation requires 4D output [B,C,H,W]." << std::endl;
            return false;
        }
        const auto& shape = tensors[0].shape();
        const int64_t batch = shape[0];
        const int64_t height = shape[2];
        const int64_t width = shape[3];
        if (tensors[0].dtype() != DataType::FP32) {
            MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
            return false;
        }
        results->resize(batch);
        const float* src = static_cast<const float*>(tensors[0].data());
        for (int64_t bs = 0; bs < batch; ++bs) {
            const float* plane = src + bs * height * width;
            const auto& rec = letter_box_records[bs];
            const float pad_h = rec.pad_h;
            const float pad_w = rec.pad_w;
            const int64_t out_h = static_cast<int64_t>(rec.out_h);
            const int64_t out_w = static_cast<int64_t>(rec.out_w);
            int64_t x1 = std::max<int64_t>(0, static_cast<int64_t>(pad_w));
            int64_t y1 = std::max<int64_t>(0, static_cast<int64_t>(pad_h));
            int64_t x2 = std::min<int64_t>(width, out_w - static_cast<int64_t>(pad_w));
            int64_t y2 = std::min<int64_t>(height, out_h - static_cast<int64_t>(pad_h));
            const int64_t crop_w = x2 - x1;
            const int64_t crop_h = y2 - y1;
            if (crop_w <= 0 || crop_h <= 0) {
                MD_LOG_ERROR << "Invalid letterbox crop region." << std::endl;
                return false;
            }
            const int64_t orig_w = static_cast<int64_t>(rec.ipt_w);
            const int64_t orig_h = static_cast<int64_t>(rec.ipt_h);
            (*results)[bs].depth.assign(static_cast<size_t>(orig_h) * orig_w, 0.0f);
            (*results)[bs].shape = {orig_h, orig_w};
            auto& depth = (*results)[bs].depth;
            // onnx 输出已是绝对深度（米）：模型内部已做 log->exp + 校准，直接拷贝到 crop 缓冲
            std::vector<float> crop_buf(static_cast<size_t>(crop_h) * crop_w);
            for (int64_t y = 0; y < crop_h; ++y) {
                const float* srow = plane + (y1 + y) * width + x1;
                float* drow = crop_buf.data() + static_cast<size_t>(y) * crop_w;
                std::memcpy(drow, srow, static_cast<size_t>(crop_w) * sizeof(float));
            }
            if (crop_w != orig_w || crop_h != orig_h) {
                std::vector<float> resized(static_cast<size_t>(orig_h) * orig_w, 0.0f);
                for (int64_t y = 0; y < orig_h; ++y) {
                    int64_t sy = std::min<int64_t>(crop_h - 1, static_cast<int64_t>(y * crop_h / orig_h));
                    const float* srow = crop_buf.data() + static_cast<size_t>(sy) * crop_w;
                    float* drow = resized.data() + static_cast<size_t>(y) * orig_w;
                    for (int64_t x = 0; x < orig_w; ++x) {
                        int64_t sx = std::min<int64_t>(crop_w - 1, static_cast<int64_t>(x * crop_w / orig_w));
                        drow[x] = srow[sx];
                    }
                }
                depth.swap(resized);
            } else {
                depth.swap(crop_buf);
            }
        }
        return true;
    }
}
