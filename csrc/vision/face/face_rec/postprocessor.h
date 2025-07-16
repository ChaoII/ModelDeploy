//
// Created by aichao on 2025/3/24.
//

#pragma once
#include "core/tensor.h"
#include "vision/common/result.h"


namespace modeldeploy::vision::face {
    /*! @brief Postprocessor object for AdaFace serials model.
     */
    class MODELDEPLOY_CXX_EXPORT SeetaFaceIDPostprocessor {
    public:
        /** \brief Create a postprocessor instance for AdaFace serials model
         */
        SeetaFaceIDPostprocessor() = default;

        /** \brief Process the result of runtime and fill to FaceRecognitionResult structure
         *
         * \param[in] tensors The inference result from runtime
         * \param[in] results The output result of FaceRecognitionResult
         * \return true if the postprocess successed, otherwise false
         */
        bool run(const std::vector<Tensor>& tensors, std::vector<FaceRecognitionResult>* results);
    };
} // namespace modeldeploy::vision::faceid
