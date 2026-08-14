//
// Created by aichao on 2025/7/22.
//

#pragma once

#include "core/tensor.h"
#include "vision/common/image_data.h"


namespace modeldeploy::vision {
    bool yolo_preprocess_nv12_cpu(const uint8_t* src_y,
                                  const uint8_t* src_uv,
                                  const std::vector<int>& src_size,
                                  int step_y,
                                  int step_uv,
                                  Tensor* output,
                                  const std::vector<int>& dst_size,
                                  float pad_value,
                                  LetterBoxRecord* letter_box_record);
}
