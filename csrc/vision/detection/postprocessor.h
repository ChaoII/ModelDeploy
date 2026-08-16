//
// Created by aichao on 2025/2/20.
//

#pragma once

#include "core/md_decl.h"
#include "core/tensor.h"
#include "vision/common/result.h"
#include "vision/common/struct.h"


namespace modeldeploy::vision::detection {
    class MODELDEPLOY_CXX_EXPORT UltralyticsPostprocessor {
    public:
        UltralyticsPostprocessor();
        bool run_without_nms(const std::vector<Tensor>& tensors,
                             std::vector<std::vector<DetectionResult>>* results,
                             const std::vector<LetterBoxRecord>& letter_box_records) const;

        bool run_with_nms(const std::vector<Tensor>& tensors,
                          std::vector<std::vector<DetectionResult>>* results,
                          const std::vector<LetterBoxRecord>& letter_box_records) const;

        bool run(const std::vector<Tensor>& tensors,
                 std::vector<std::vector<DetectionResult>>* results,
                 const std::vector<LetterBoxRecord>& letter_box_records) const;

        /// Set conf_threshold, default 0.25
        void set_conf_threshold(const float& conf_threshold) {
            conf_threshold_ = conf_threshold;
        }

        /// Get conf_threshold, default 0.25
        [[nodiscard]] float get_conf_threshold() const { return conf_threshold_; }

        /// Set nms_threshold, default 0.5
        void set_nms_threshold(const float& nms_threshold) {
            nms_threshold_ = nms_threshold;
        }

        /// Get nms_threshold, default 0.5
        [[nodiscard]] float get_nms_threshold() const { return nms_threshold_; }

        /// 模型 class 通道是否已含 Sigmoid（输出即概率 [0,1]）。
        /// Ultralytics 官方导出 Detect 头已含 Sigmoid → 默认 true（不做二次 sigmoid）。
        /// 若模型输出为未激活 raw logits，设 false 后 postprocess 会先 sigmoid 再过滤。
        void set_apply_sigmoid(bool apply) { apply_sigmoid_ = apply; }
        [[nodiscard]] bool get_apply_sigmoid() const { return apply_sigmoid_; }

    protected:
        float conf_threshold_;
        float nms_threshold_;
        bool apply_sigmoid_ = true; // Ultralytics Detect 头已含 Sigmoid
    };
} // namespace modeldeploy
