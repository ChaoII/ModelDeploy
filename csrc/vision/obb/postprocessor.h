//
// Created by aichao on 2025/5/30.
//
#pragma once
#include "csrc/core/tensor.h"
#include "csrc/vision/common/result.h"

namespace modeldeploy::vision::detection {
    /*! @brief Postprocessor object for YOLOv8OBB serials model.
   */
    class MODELDEPLOY_CXX_EXPORT UltralyticsObbPostprocessor {
    public:
        /** \brief Create a postprocessor instance for YOLOv8OBB serials model
       */
        UltralyticsObbPostprocessor();

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

    protected:
        float conf_threshold_;
        float nms_threshold_;
        // mask threshold
        float mask_threshold_;
    };
}
