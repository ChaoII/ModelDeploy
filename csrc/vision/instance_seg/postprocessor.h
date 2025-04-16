//
// Created by aichao on 2025/4/14.
//
#pragma once
#include "csrc/core/tensor.h"
#include "csrc/vision/common/result.h"

namespace modeldeploy::vision::detection {
    /*! @brief Postprocessor object for YOLOv5Seg serials model.
   */
    class MODELDEPLOY_CXX_EXPORT YOLOv5SegPostprocessor {
    public:
        /** \brief Create a postprocessor instance for YOLOv5Seg serials model
       */
        YOLOv5SegPostprocessor();

        /** \brief Process the result of runtime and fill to DetectionResult structure
       *
       * \param[in] tensors The inference result from runtime
       * \param[in] results The output result of detection
       * \param[in] ims_info The shape info list, record input_shape and output_shape
       * \return true if the postprocess successed, otherwise false
       */
        bool run(const std::vector<Tensor>& tensors,
                 std::vector<DetectionResult>* results,
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

        /// Set multi_label, set true for eval, default true
        void set_multi_label(const bool multi_label) {
            multi_label_ = multi_label;
        }

        /// Get multi_label, default true
        [[nodiscard]] bool get_multi_label() const { return multi_label_; }

    protected:
        float conf_threshold_;
        float nms_threshold_;
        bool multi_label_;
        float max_wh_;
        // channel nums of masks
        int mask_nums_;
        // mask threshold
        float mask_threshold_;
    };
}
