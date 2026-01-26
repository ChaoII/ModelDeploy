//
// Created by aichao on 2026/1/26.
//

#pragma once

#include "core/tensor.h"
#include "vision/common/image_data.h"


namespace modeldeploy::vision {
    bool fusion_resize_pad_normalize_permute_cuda(
        const ImageData& image, Tensor* output,
        const std::vector<int>& resize_size,
        const std::vector<int>& dst_size,
        const std::vector<float>& mean,
        const std::vector<float>& std,
        float pad_value);
}
