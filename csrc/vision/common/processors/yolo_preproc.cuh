//
// Created by aichao on 2025/7/22.
//

#pragma once
#include <cuda_runtime.h>
#include "core/tensor.h"
#include "vision/common/struct.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision {
    bool yolo_preprocess_cuda(const ImageData& image,
                              Tensor* output,
                              const std::vector<int>& dst_size,
                              float pad_val,
                              LetterBoxRecord* letter_box_record,
                              cudaStream_t stream = nullptr);

    bool yolo_preprocess_bgr_cuda(const uint8_t* src,
                                  const std::vector<int>& src_size,
                                  Tensor* output,
                                  const std::vector<int>& dst_size,
                                  float pad_val,
                                  LetterBoxRecord* letter_box_record,
                                  cudaStream_t stream = nullptr);

    bool yolo_preprocess_nv12_cuda(const uint8_t* src_y,
                                   const uint8_t* src_uv,
                                   const std::vector<int>& src_size,
                                   int step_y,
                                   int step_uv,
                                   Tensor* output,
                                   const std::vector<int>& dst_size,
                                   float pad_value,
                                   LetterBoxRecord* letter_box_record,
                                   cudaStream_t stream = nullptr);
}
