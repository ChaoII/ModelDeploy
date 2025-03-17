//
// Created by aichao on 2025/2/21.
//

#pragma once
#include "csrc/core/md_decl.h"
#include "csrc/core/md_tensor.h"

namespace modeldeploy::vision::ocr {
    class MODELDEPLOY_CXX_EXPORT ClassifierPostprocessor {
    public:
        /** \brief Process the result of runtime and fill to ClassifyResult structure
         *
         * \param[in] tensors The inference result from runtime
         * \param[in] cls_labels The output label results of classification model
         * \param[in] cls_scores The output score results of classification model
         * \return true if the postprocess successed, otherwise false
         */
        bool run(const std::vector<MDTensor>& tensors,
                 std::vector<int32_t>* cls_labels, std::vector<float>* cls_scores);

        bool run(const std::vector<MDTensor>& tensors,
                 std::vector<int32_t>* cls_labels, std::vector<float>* cls_scores,
                 size_t start_index, size_t total_size);

        /// Set threshold for the classification postprocess, default is 0.9
        void set_cls_thresh(float cls_thresh) { cls_thresh_ = cls_thresh; }

        /// Get threshold value of the classification postprocess.
        [[nodiscard]] float get_cls_thresh() const { return cls_thresh_; }

    private:
        float cls_thresh_ = 0.9;
    };
}
