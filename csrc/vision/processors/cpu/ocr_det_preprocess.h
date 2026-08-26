//
// Created by aichao on 2025/7/22.
//

#pragma once

#include <array>
#include <vector>
#include "core/tensor.h"
#include "core/md_decl.h"
#include "vision/common/image_data.h"


namespace modeldeploy::vision {
    MODELDEPLOY_CXX_EXPORT bool ocr_det_preprocess_cpu(
        const std::vector<ImageData>&, Tensor* output,
        const std::vector<std::array<int, 2>>& resize_sizes,
        const std::vector<int>& dst_size,
        const std::vector<float>& mean,
        const std::vector<float>& std,
        float pad_value);
}
