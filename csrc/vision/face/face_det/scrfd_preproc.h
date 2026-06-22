#pragma once
#include "core/tensor.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision {
    bool scrfd_preprocess_cpu(const ImageData& image,
                              Tensor* output,
                              const std::vector<int>& dst_size,
                              float pad_val,
                              LetterBoxRecord* letter_box_record);

    bool scrfd_preprocess_bgr_cpu(const uint8_t* src,
                                  const std::vector<int>& src_size,
                                  Tensor* output,
                                  const std::vector<int>& dst_size,
                                  float pad_val,
                                  LetterBoxRecord* letter_box_record);

    bool scrfd_preprocess_nv12_cpu(const uint8_t* src_y,
                                   const uint8_t* src_uv,
                                   const std::vector<int>& src_size,
                                   int step_y,
                                   int step_uv,
                                   Tensor* output,
                                   const std::vector<int>& dst_size,
                                   float pad_value,
                                   LetterBoxRecord* letter_box_record);
}