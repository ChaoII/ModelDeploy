//
// Created by aichao on 2025/7/22.
//

#pragma once
#include <cuda_runtime.h>
#include "core/tensor.h"
#include "vision/common/struct.h"
#include "vision/common/image_data.h"

cudaError_t yolo_preproc(
    const uint8_t* src, int src_h, int src_w,
    float* dst, int dst_h, int dst_w, cudaStream_t stream,
    const float mean[3], const float std[3], float pad_value,
    modeldeploy::vision::LetterBoxRecord* info_out);

namespace modeldeploy::vision {
    bool yolo_preprocess_cuda(ImageData* image, Tensor* output,
                              const std::vector<int>& dst_size,
                              const std::vector<float>& pad_val,
                              LetterBoxRecord* letter_box_record);
}
