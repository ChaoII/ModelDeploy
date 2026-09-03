//
// Created by aichao on 2025/2/20.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/detection/postprocessor.h"
#include <algorithm>

namespace modeldeploy::vision::detection {
    UltralyticsPostprocessor::UltralyticsPostprocessor() {
        conf_threshold_ = 0.25;
        nms_threshold_ = 0.5;
    }

    bool UltralyticsPostprocessor::run_without_nms(
        const std::vector<Tensor>& tensors, std::vector<std::vector<DetectionResult>>* results,
        const std::vector<LetterBoxRecord>& letter_box_records) const {
        const size_t batch = tensors[0].shape()[0];
        if (tensors[0].dtype() != DataType::FP32) {
            MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
            return false;
        }
        // 原始布局 [B, C, N]：C=84 (4(xc,yc,w,h)+80 classes)，N=8400 (anchors)
        // 无需转置物化：按 anchor 分块顺序扫各 class 行求 max，只对过阈候选解码。
        const size_t num_classes_total = tensors[0].shape()[1]; // 84
        const size_t num_anchors = tensors[0].shape()[2];       // 8400
        const size_t num_classes = num_classes_total - 4;       // 80（通道 4..83）
        results->resize(batch);

        for (size_t bs = 0; bs < batch; ++bs) {
            const float* src = static_cast<const float*>(tensors[0].data()) + bs * num_classes_total * num_anchors;
            // 过阈值的结果通常很少（正常图 3-10 个），依赖 vector 自动增长即可
            std::vector<DetectionResult> _results;
            const size_t kBlock = 256;
            for (size_t blk = 0; blk < num_anchors; blk += kBlock) {
                const size_t cnt = std::min(kBlock, num_anchors - blk);
                float maxs[256];
                int argmax[256];
                const float* c0 = src + 4 * num_anchors + blk;  // class 0（通道4）
                for (size_t i = 0; i < cnt; ++i) { maxs[i] = c0[i]; argmax[i] = 0; }
                for (size_t c = 1; c < num_classes; ++c) {
                    const float* row = src + (4 + c) * num_anchors + blk;
                    for (size_t i = 0; i < cnt; ++i) {
                        if (row[i] > maxs[i]) { maxs[i] = row[i]; argmax[i] = static_cast<int>(c); }
                    }
                }
                for (size_t i = 0; i < cnt; ++i) {
                    const size_t a = blk + i;
                    const float x = src[0 * num_anchors + a], y = src[1 * num_anchors + a];
                    const float w = src[2 * num_anchors + a], h = src[3 * num_anchors + a];
                    // 过滤无效框（非正宽高）
                    if (w <= 0 || h <= 0) {
                        continue;
                    }
                    // maxs[i] 即最高类分数；argmax[i] 即类别。
                    // Ultralytics 官方导出 Detect 头已含 Sigmoid，输出即概率 [0,1]，
                    // 直接比阈值即可（二次 sigmoid 会把概率推向 1，导致全候选过阈、
                    // NMS 退化为 O(n^2)，实测 det post 463ms 的根因）。
                    const float confidence = maxs[i];
                    if (confidence <= conf_threshold_) {
                        continue;
                    }
                    // convert from [xc, yc, w, h] to [x, y, width, height]
                    Rect2f box = {x - w / 2.0f, y - h / 2.0f, w, h};
                    _results.push_back({box, argmax[i], confidence});
                }
            }
            if (_results.empty()) {
                continue;
            }
            utils::nms(&_results, nms_threshold_);
            // scale the boxes to the origin image shape

            const float ipt_h = letter_box_records[bs].ipt_h;
            const float ipt_w = letter_box_records[bs].ipt_w;
            const float scale = letter_box_records[bs].scale;
            const float pad_h = letter_box_records[bs].pad_h;
            const float pad_w = letter_box_records[bs].pad_w;

            for (auto& result : _results) {
                auto& box = result.box;
                // clip box()
                //1 先减去 padding;2除以缩放因子scale 3最后限制在原始图像范围内 [0, width], [0, height]。
                float x1 = (box.x - pad_w) / scale;
                float y1 = (box.y - pad_h) / scale;
                float x2 = (box.x + box.width - pad_w) / scale;
                float y2 = (box.y + box.height - pad_h) / scale;

                // 限制在图像边界内
                x1 = std::clamp(x1, 0.0f, ipt_w);
                y1 = std::clamp(y1, 0.0f, ipt_h);
                x2 = std::clamp(x2, 0.0f, ipt_w);
                y2 = std::clamp(y2, 0.0f, ipt_h);

                // 重新赋值到 box（与 iseg/pose 保持一致，不额外 -0.5）
                box.x = std::roundf(x1);
                box.y = std::roundf(y1);
                box.width = std::roundf(x2 - x1);
                box.height = std::roundf(y2 - y1);
            }
            // 缩放后窄框可能产生非正宽高，统一过滤
            _results.erase(std::remove_if(_results.begin(), _results.end(),
                [](const DetectionResult& r) { return r.box.width <= 0 || r.box.height <= 0; }),
                _results.end());
            (*results)[bs] = std::move(_results);
        }
        return true;
    }

    bool UltralyticsPostprocessor::run_with_nms(const std::vector<Tensor>& tensors,
                                                std::vector<std::vector<DetectionResult>>* results,
                                                const std::vector<LetterBoxRecord>& letter_box_records) const {
        const size_t batch = tensors[0].shape()[0];
        results->resize(batch);
        for (size_t bs = 0; bs < batch; ++bs) {
            if (tensors[0].dtype() != DataType::FP32) {
                MD_LOG_ERROR << "Only support post process with float32 data." << std::endl;
                return false;
            }
            const size_t dim1 = tensors[0].shape()[1]; //300
            const size_t dim2 = tensors[0].shape()[2]; //6
            const float* data = static_cast<const float*>(tensors[0].data()) + bs * dim1 * dim2;
            std::vector<DetectionResult> _results;
            _results.reserve(dim1);
            for (size_t i = 0; i < dim1; ++i) {
                const float* attr_ptr = data + i * dim2;
                const float score = attr_ptr[4];
                // filter boxes by conf_threshold
                if (score <= conf_threshold_) {
                    continue;
                }
                int32_t label_id = attr_ptr[5];
                // convert from [x1, y1, x2, y2] to [x, y, width, height]
                Rect2f box = {
                    attr_ptr[0],
                    attr_ptr[1],
                    attr_ptr[2] - attr_ptr[0],
                    attr_ptr[3] - attr_ptr[1]
                };
                _results.push_back({box, label_id, score});
            }
            if (_results.empty()) {
                continue;
            }
            // utils::nms(&_results, nms_threshold_);
            // // scale the boxes to the origin image shape

            const float ipt_h = letter_box_records[bs].ipt_h;
            const float ipt_w = letter_box_records[bs].ipt_w;
            const float scale = letter_box_records[bs].scale;
            const float pad_h = letter_box_records[bs].pad_h;
            const float pad_w = letter_box_records[bs].pad_w;

            for (auto& result : _results) {
                auto& box = result.box;
                // clip box()
                //1 先减去 padding;2除以缩放因子scale 3最后限制在原始图像范围内 [0, width], [0, height]。
                float x1 = (box.x - pad_w) / scale;
                float y1 = (box.y - pad_h) / scale;
                float x2 = (box.x + box.width - pad_w) / scale;
                float y2 = (box.y + box.height - pad_h) / scale;

                // 限制在图像边界内
                x1 = std::clamp(x1, 0.0f, ipt_w);
                y1 = std::clamp(y1, 0.0f, ipt_h);
                x2 = std::clamp(x2, 0.0f, ipt_w);
                y2 = std::clamp(y2, 0.0f, ipt_h);

                // 重新赋值到 box（与 iseg/pose 保持一致，不额外 -0.5）
                box.x = std::roundf(x1);
                box.y = std::roundf(y1);
                box.width = std::roundf(x2 - x1);
                box.height = std::roundf(y2 - y1);
            }
            (*results)[bs] = std::move(_results);
        }
        return true;
    }

    bool UltralyticsPostprocessor::run(const std::vector<Tensor>& tensors,
                                       std::vector<std::vector<DetectionResult>>* results,
                                       const std::vector<LetterBoxRecord>& letter_box_records) const {
        const auto& t = tensors[0];
        // ncnn 在 batch==1 时压掉首维，输出 2D [C,N]；检测层期望 3D [B,C,N]，
        // 故将 2D 视为 batch=1（共享内存视图），3D 输出行为保持不变。
        if (t.shape().size() == 2) {
            const std::vector<Tensor> batched = {t.reshape({1, t.shape()[0], t.shape()[1]})};
            return run(batched, results, letter_box_records);
        }
        if (t.shape().size() != 3) {
            MD_LOG_ERROR << "Only support post process with 3D tensor, got dims=" << t.shape().size() << std::endl;
            return false;
        }
        if (t.shape()[2] == 6) {
            return run_with_nms(tensors, results, letter_box_records);
        }
        return run_without_nms(tensors, results, letter_box_records);
    }
}
