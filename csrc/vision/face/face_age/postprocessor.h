//
// Created by aichao on 2025/3/24.
//

#pragma once

#include "csrc/core/tensor.h"
#include "csrc/vision/common/result.h"


namespace modeldeploy::vision::face {
    /*! @brief Postprocessor object for AdaFace serials model.
     */
    class MODELDEPLOY_CXX_EXPORT SeetaFaceAgePostprocessor {
    public:
        /** \brief Create a postprocessor instance for AdaFace serials model
         */
        SeetaFaceAgePostprocessor() = default;

        /** \brief Process the result of runtime and fill to FaceRecognitionResult structure
         *
         * \param[in] infer_result The inference result from runtime
         * \param[in] ages The output result of FaceRecognitionResult
         * \return true if the postprocess successed, otherwise false
         */
        bool run(const std::vector<Tensor>& infer_result, std::vector<int>* ages) const;
    };
} // namespace modeldeploy::vision::faceid
