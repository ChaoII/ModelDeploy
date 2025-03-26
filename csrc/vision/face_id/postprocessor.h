//
// Created by aichao on 2025/3/24.
//

#pragma once
#include "csrc/core/md_tensor.h"
#include "csrc/vision/common/result.h"


namespace modeldeploy::vision::faceid {
    /*! @brief Postprocessor object for AdaFace serials model.
     */
    class MODELDEPLOY_CXX_EXPORT SeetaFacePostprocessor {
    public:
        /** \brief Create a postprocessor instance for AdaFace serials model
         */
        SeetaFacePostprocessor() = default;

        /** \brief Process the result of runtime and fill to FaceRecognitionResult structure
         *
         * \param[in] infer_result The inference result from runtime
         * \param[in] results The output result of FaceRecognitionResult
         * \return true if the postprocess successed, otherwise false
         */
        bool run(std::vector<MDTensor>& infer_result, std::vector<FaceRecognitionResult>* results);
    };
} // namespace modeldeploy::vision::faceid
