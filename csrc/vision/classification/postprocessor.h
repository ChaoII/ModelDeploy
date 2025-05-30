//
// Created by aichao on 2025/2/24.
//

#pragma once
#include <map>
#include "csrc/core/md_decl.h"
#include "csrc/vision/common/result.h"
#include "csrc/core/tensor.h"

namespace modeldeploy::vision::classification {
    /*! @brief Postprocessor object for YOLOv5Cls serials model.
    */
    class MODELDEPLOY_CXX_EXPORT UltralyticsClsPostprocessor {
    public:
        /** \brief Create a postprocessor instance for YOLOv5Cls serials model
        */
        UltralyticsClsPostprocessor();
        /** \brief Process the result of runtime and fill to ClassifyResult structure
        *
        * \param[in] tensors The inference result from runtime
        * \param[in] results The output result of classification
        * \return true if the postprocess successful, otherwise false
        */
        bool run(const std::vector<Tensor>& tensors,
                 std::vector<ClassifyResult>* results);

        /// Set topk, default 1
        void set_top_k(const int& top_k) {
            top_k_ = top_k;
        }

        /// Get topk, default 1
        [[nodiscard]] int get_top_k() const { return top_k_; }

    protected:
        int top_k_;
    };
}
