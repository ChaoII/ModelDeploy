//
// Created by aichao on 2026/8/13.
//

#pragma once

#include "core/tensor.h"
#include "vision/common/result.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision::detection {
    /*! @brief Postprocessor object for yolo26n-depth (depth-log model)
     * 输出 [1, 1, H, W] log 空间深度，后处理 exp 还原为米并去除 letterbox padding
     */
    class MODELDEPLOY_CXX_EXPORT UltralyticsDepthPostprocessor {
    public:
        UltralyticsDepthPostprocessor();

        bool run(std::vector<Tensor>& tensors,
                 std::vector<DepthResult>* results,
                 const std::vector<LetterBoxRecord>& letter_box_records) const;
    };
}
