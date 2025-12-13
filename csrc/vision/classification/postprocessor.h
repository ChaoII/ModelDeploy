//
// Created by aichao on 2025/2/24.
//

#pragma once
#include "core/md_decl.h"
#include "vision/common/result.h"
#include "core/tensor.h"

namespace modeldeploy::vision::classification {
    /*! @brief Postprocessor object for YOLOv5Cls serials model.
    */
    class MODELDEPLOY_CXX_EXPORT ClassificationPostprocessor {
    public:
        /** \brief Create a postprocessor instance for YOLOv5Cls serials model
        */
        ClassificationPostprocessor();
        /** \brief Process the result of runtime and fill to ClassifyResult structure
        *
        * \param[in] tensors The inference result from runtime
        * \param[in] results The output result of classification
        * \return true if the postprocess successful, otherwise false
        */
        bool run(const std::vector<Tensor>& tensors,
                 std::vector<ClassifyResult>* results) const;

        /// Set topk, default 1
        void set_top_k(const int& top_k) {top_k_ = top_k;}

        /// Get topk, default 1
        [[nodiscard]] int get_top_k() const { return top_k_; }

        // Set multi_label, default false
        void set_multi_label(const bool& multi_label) {multi_label_ = multi_label;}

        /// Get multi_label, default false
        [[nodiscard]] bool get_multi_label() const { return multi_label_; }

    protected:
        int top_k_;
        bool multi_label_ = false;
    };
}
