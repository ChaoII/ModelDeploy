//
// Created by aichao on 2025/6/10.
//

#pragma once
#include "core/md_decl.h"
#include "core/tensor.h"
#include "vision/common/result.h"
#include "vision/common/struct.h"


namespace modeldeploy::vision::lpr {
    class MODELDEPLOY_CXX_EXPORT LprDetPostprocessor {
    public:
        LprDetPostprocessor();
        bool run(const std::vector<Tensor>& tensors,
                 std::vector<std::vector<DetectionLandmarkResult>>* results,
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


        /// Set nms_threshold, default 0.5
        void set_landmarks_per_card(const float& landmarks_per_card) {
            landmarks_per_card_ = landmarks_per_card;
        }

        /// Get nms_threshold, default 0.5
        [[nodiscard]] int get_landmarks_per_card() const { return landmarks_per_card_; }

    protected:
        float conf_threshold_;
        float nms_threshold_;
        int landmarks_per_card_;
    };
} // namespace modeldeploy
