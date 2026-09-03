//
// Created by aichao on 2026/8/13.
//

#pragma once

#include "core/tensor.h"
#include "vision/common/result.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision::detection {
    /*! @brief Postprocessor object for yolo26n-sem (semantic segmentation)
     * 输出 [1, C, H, W] raw logits，后处理 argmax 得每像素类别索引，并去除 letterbox padding
     */
    class MODELDEPLOY_CXX_EXPORT UltralyticsSemPostprocessor {
    public:
        UltralyticsSemPostprocessor();

        bool run(std::vector<Tensor>& tensors,
                 std::vector<SemSegResult>* results,
                 const std::vector<LetterBoxRecord>& letter_box_records) const;

    protected:
        int num_classes_ = 19;
    };
}
