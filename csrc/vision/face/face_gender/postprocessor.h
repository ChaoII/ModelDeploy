//
// Created by aichao on 2025/3/24.
//

#pragma once
#include "core/tensor.h"
#include "vision/common/result.h"


namespace modeldeploy::vision::face {
    /*! @brief Postprocessor object for AdaFace serials model.
     */
    class MODELDEPLOY_CXX_EXPORT SeetaFaceGenderPostprocessor {
    public:
        /** \brief Create a postprocessor instance for AdaFace serials model
         */
        SeetaFaceGenderPostprocessor() = default;

        /** \brief Process the result of runtime and fill to FaceRecognitionResult structure
         *
         * \param[in] infer_result The inference result from runtime
         * \param[in] genders The output result of FaceRecognitionResult
         * \return true if the postprocess successful, otherwise false
         */
        bool run(const std::vector<Tensor>& infer_result, std::vector<int>* genders) const;
    };
} // namespace modeldeploy::vision::faceid
