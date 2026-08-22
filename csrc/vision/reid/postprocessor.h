//
// Created for standalone pedestrian Re-ID (OSNet) postprocessing.
//

#pragma once

#include <vector>

#include "core/md_decl.h"
#include "core/tensor.h"
#include "vision/common/result.h"

namespace modeldeploy::vision::reid {
    /*! @brief Postprocessor for OSNet pedestrian Re-ID model.
     *  将模型输出 embedding 逐样本提取并作 L2 归一化。
     */
    class MODELDEPLOY_CXX_EXPORT ReIDPostprocessor {
    public:
        /** \brief Flatten the model output and L2-normalize each embedding.
         *  \param[in] inputs The runtime output tensors
         *  \param[in] results Per-image vector of ReIdResult
         *  \return true if the postprocess succeeded, otherwise false
         */
        bool run(const std::vector<Tensor>& inputs,
                 std::vector<std::vector<ReIdResult>>* results);
    };
} // namespace modeldeploy::vision::reid
