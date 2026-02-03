//
// Created by aichao on 2025/7/22.
//

#pragma once

#include "core/tensor.h"
#include "vision/common/image_data.h"


namespace modeldeploy::vision {
    bool yolo_preprocess_cpu(const ImageData& image, Tensor* output,
                             const std::vector<int>& dst_size,
                             float pad_val,
                             LetterBoxRecord* letter_box_record);
}
