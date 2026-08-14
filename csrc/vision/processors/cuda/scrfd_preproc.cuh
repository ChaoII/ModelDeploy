#pragma once
#include <cuda_runtime.h>
#include "core/tensor.h"
#include "vision/common/struct.h"
#include "vision/common/image_data.h"
#include "vision/processors/cuda/cuda_output_pool.h"

namespace modeldeploy::vision {
    bool scrfd_preprocess_cuda(const ImageData& image,
                               Tensor* output,
                               const std::vector<int>& dst_size,
                               float pad_val,
                               LetterBoxRecord* letter_box_record,
                               cudaStream_t stream = nullptr,
                               CudaOutputBufferPool* dst_pool = nullptr);

    bool scrfd_preprocess_bgr_cuda(const uint8_t* src,
                                   const std::vector<int>& src_size,
                                   Tensor* output,
                                   const std::vector<int>& dst_size,
                                   float pad_val,
                                   LetterBoxRecord* letter_box_record,
                                   cudaStream_t stream = nullptr,
                                   CudaOutputBufferPool* dst_pool = nullptr);

    bool scrfd_preprocess_nv12_cuda(const uint8_t* src_y,
                                    const uint8_t* src_uv,
                                    const std::vector<int>& src_size,
                                    int step_y,
                                    int step_uv,
                                    Tensor* output,
                                    const std::vector<int>& dst_size,
                                    float pad_value,
                                    LetterBoxRecord* letter_box_record,
                                    cudaStream_t stream = nullptr,
                                    CudaOutputBufferPool* dst_pool = nullptr);

    // 整批 SCRFD 预处理（3D grid 一次 launch），输出 [batch, 3, dst_h, dst_w] FP32 GPU
    bool scrfd_preprocess_batch_cuda(const std::vector<ImageData>& images,
                                     Tensor* output,
                                     const std::vector<int>& dst_size,
                                     float pad_value,
                                     std::vector<LetterBoxRecord>* letter_box_records,
                                     CudaOutputBufferPool* dst_pool = nullptr,
                                     cudaStream_t stream = nullptr);
}