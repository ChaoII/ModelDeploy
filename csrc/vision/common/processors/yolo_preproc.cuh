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

    // 整批融合预处理（3D grid 一次 launch）：对 batch 内每张图做 letterbox+resize+normalize+CHW
    // 输出 Tensor [batch, 3, dst_h, dst_w] FP32 GPU
    bool yolo_preprocess_batch_cuda(const std::vector<ImageData>& images,
                                    Tensor* output,
                                    const std::vector<int>& dst_size,
                                    float pad_value,
                                    std::vector<LetterBoxRecord>* letter_box_records,
                                    cudaStream_t stream = nullptr);
}
