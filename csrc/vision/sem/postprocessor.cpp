//
// Created by aichao on 2026/8/13.
//

#include "core/md_log.h"
#include "vision/sem/postprocessor.h"

#include <algorithm>
#include <cstring>
#include <opencv2/opencv.hpp>

namespace modeldeploy::vision::detection {
    UltralyticsSemPostprocessor::UltralyticsSemPostprocessor() = default;

    bool UltralyticsSemPostprocessor::run(
        std::vector<Tensor>& tensors, std::vector<SemSegResult>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        if (!tensors.empty() && tensors[0].shape().size() == 3) {
            // ncnn batch==1 压掉首维；sem 全图 4D [1,C,H,W] → 通用 Tensor::expand_dim(0) 补回后落体。
            tensors[0].expand_dim(0);
        }
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
        const int64_t hw = height * width;
        for (int64_t bs = 0; bs < batch; ++bs) {
            const float* plane = src + bs * channels * hw;
            (*results)[bs].num_classes = static_cast<int32_t>(channels);
            const auto& rec = letter_box_records[bs];
            const float pad_h = rec.pad_h;
            const float pad_w = rec.pad_w;
            const int64_t out_h = static_cast<int64_t>(rec.out_h);
            const int64_t out_w = static_cast<int64_t>(rec.out_w);
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
            const int64_t orig_w = static_cast<int64_t>(rec.ipt_w);
            const int64_t orig_h = static_cast<int64_t>(rec.ipt_h);

            // 每个通道建立 crop 区 Mat 视图（指向 plane 各平面，零拷贝，只读）
            const int cw = static_cast<int>(crop_w);
            const int chh = static_cast<int>(crop_h);
            cv::Mat ch0(chh, cw, CV_32FC1,
                        const_cast<float*>(plane + y1 * width + x1), width * sizeof(float));
            cv::Mat best = ch0.clone();  // 连续，作为累积最大值
            cv::Mat best_idx(chh, cw, CV_8UC1, cv::Scalar(0));
            cv::Mat cmp, cur;
            for (int64_t c = 1; c < channels; ++c) {
                cur = cv::Mat(chh, cw, CV_32FC1,
                              const_cast<float*>(plane + c * hw + y1 * width + x1), width * sizeof(float));
                cv::compare(cur, best, cmp, cv::CMP_GT);
                cv::max(cur, best, best);
                best_idx.setTo(static_cast<uchar>(c), cmp);
            }

            // 最近邻缩放到原图尺寸（OpenCV SIMD）
            cv::Mat resized;
            cv::resize(best_idx, resized, cv::Size(static_cast<int>(orig_w), static_cast<int>(orig_h)),
                       0, 0, cv::INTER_NEAREST);
            (*results)[bs].labels.assign(resized.data, resized.data + resized.total());
            (*results)[bs].shape = {orig_h, orig_w};
        }
        return true;
    }
}
