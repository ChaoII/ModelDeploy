//
// Created by aichao on 2025/06/2.
//

#pragma once
#include "csrc/core/md_decl.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/common/result.h"

namespace modeldeploy::vision::detection {
    class MODELDEPLOY_CXX_EXPORT UltralyticsPosePostprocessor {
    public:
        UltralyticsPosePostprocessor();
        bool run(const std::vector<Tensor>& tensors,
                 std::vector<std::vector<PoseResult>>* results,
                 const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info) const;

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

    protected:
        int keypoints_num_;
        float conf_threshold_;
        float nms_threshold_;
    };
} // namespace fastdeploy
