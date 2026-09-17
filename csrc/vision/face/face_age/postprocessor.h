//
// Created by aichao on 2025/3/24.
//

#pragma once

#include "core/tensor.h"
#include "vision/common/result.h"


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

        /** \brief 额外输出每张脸的年龄概率分布（新增，附属于既有接口）
         *
         * \param[in] infer_result 推理输出
         * \param[out] dists 每张脸一组分布，下标=年龄（0..N-1）；模型输出已 softmax，和为 1
         * \return true 成功
         */
        bool run_dist(const std::vector<Tensor>& infer_result,
                      std::vector<std::vector<float>>* dists) const;
    };
} // namespace modeldeploy::vision::faceid
