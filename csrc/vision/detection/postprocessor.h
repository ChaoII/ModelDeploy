//
// Created by aichao on 2025/2/20.
//

#pragma once
#include "../common/result.h"
#include "../utils.h"

namespace modeldeploy::vision::detection {
    class YOLOv8Postprocessor {
    public:
        YOLOv8Postprocessor();
        bool run(const std::vector<MDTensor>& tensors,
                 std::vector<DetectionResult>* results,
                 const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info);

        /// Set conf_threshold, default 0.25
        void set_conf_threshold(const float& conf_threshold) {
            conf_threshold_ = conf_threshold;
        }

        /// Get conf_threshold, default 0.25
        float get_conf_threshold() const { return conf_threshold_; }

        /// Set nms_threshold, default 0.5
        void set_nms_threshold(const float& nms_threshold) {
            nms_threshold_ = nms_threshold;
        }

        /// Get nms_threshold, default 0.5
        float get_nms_threshold() const { return nms_threshold_; }

        /// Set multi_label, set true for eval, default true
        void set_multi_label(bool multi_label) {
            multi_label_ = multi_label;
        }

        /// Get multi_label, default true
        bool get_multi_label() const { return multi_label_; }

    protected:
        float conf_threshold_;
        float nms_threshold_;
        bool multi_label_;
        float max_wh_;
    };
} // namespace fastdeploy
