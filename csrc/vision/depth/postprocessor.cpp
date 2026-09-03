//
// Created by aichao on 2026/8/13.
//

#include "core/md_log.h"
#include "vision/depth/postprocessor.h"
#include "vision/utils/ncnn_output.h"

#include <algorithm>
#include <cstring>
#include <opencv2/opencv.hpp>

namespace modeldeploy::vision::detection {
    UltralyticsDepthPostprocessor::UltralyticsDepthPostprocessor() = default;

    bool UltralyticsDepthPostprocessor::run(
        const std::vector<Tensor>& tensors, std::vector<DepthResult>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        if (!tensors.empty() && tensors[0].shape().size() == 3) {
            // ncnn 在 batch==1 时压掉输出首维，depth 为全图 4D 输出 [1,1,H,W] → 补回 4D 后递归。
            const std::vector<Tensor> batched = {
                vision::ncnn_utils::restore_leading_batch1(tensors[0], 4)};
            return run(batched, results, letter_box_records);
        }
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
            // 整幅深度图（含 padding）包成 Mat 视图
            cv::Mat full(static_cast<int>(height), static_cast<int>(width), CV_32FC1,
                         const_cast<float*>(plane));
            // 裁剪去 letterbox padding
            const cv::Rect crop_roi(static_cast<int>(x1), static_cast<int>(y1),
                                    static_cast<int>(crop_w), static_cast<int>(crop_h));
            const cv::Mat crop_mat = full(crop_roi).clone();
            // 双线性缩放到原图尺寸（OpenCV SIMD，与 ultralytics scale_masks 一致）
            cv::Mat resized;
            cv::resize(crop_mat, resized, cv::Size(static_cast<int>(orig_w), static_cast<int>(orig_h)),
                       0, 0, cv::INTER_LINEAR);
            (*results)[bs].depth.assign(
                reinterpret_cast<const float*>(resized.data),
                reinterpret_cast<const float*>(resized.data) + resized.total());
            (*results)[bs].shape = {orig_h, orig_w};
        }
        return true;
    }
}
