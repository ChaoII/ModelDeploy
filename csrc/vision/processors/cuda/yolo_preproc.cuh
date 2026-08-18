//
// Created by aichao on 2025/7/22.
//

#pragma once
#include <cuda_runtime.h>
#include "core/tensor.h"
#include "vision/common/struct.h"
#include "vision/common/image_data.h"
#include "vision/processors/cuda/cuda_output_pool.h"

namespace modeldeploy::vision {
    bool yolo_preprocess_cuda(const ImageData& image,
                              Tensor* output,
                              const std::vector<int>& dst_size,
                              float pad_val,
                              LetterBoxRecord* letter_box_record,
                              cudaStream_t stream = nullptr,
                              CudaOutputBufferPool* dst_pool = nullptr);

    bool yolo_preprocess_bgr_cuda(const uint8_t* src,
                                  const std::vector<int>& src_size,
                                  Tensor* output,
                                  const std::vector<int>& dst_size,
                                  float pad_val,
                                  LetterBoxRecord* letter_box_record,
                                  cudaStream_t stream = nullptr,
                                  CudaOutputBufferPool* dst_pool = nullptr);

    bool yolo_preprocess_nv12_cuda(const uint8_t* src_y,
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

    // 整批融合预处理（3D grid 一次 launch）：对 batch 内每张图做 letterbox+resize+normalize+CHW
    // 输出 Tensor [batch, 3, dst_h, dst_w] FP32 GPU
    bool yolo_preprocess_batch_cuda(const std::vector<ImageData>& images,
                                    Tensor* output,
                                    const std::vector<int>& dst_size,
                                    float pad_value,
                                    std::vector<LetterBoxRecord>* letter_box_records,
                                    cudaStream_t stream = nullptr,
                                    CudaOutputBufferPool* dst_pool = nullptr);

    // NV12 整批融合预处理（3D grid 一次 launch）：逐图 Y/UV 平面（设备零拷贝或 host H2D）
    // 设备帧：plane 指针为显存，kernel 直读（零 PCIe）；host 帧：聚合 H2D 后 kernel 读设备槽。
    // 输出 Tensor [batch, 3, dst_h, dst_w] FP32 GPU
    bool yolo_preprocess_nv12_batch_cuda(const std::vector<ImageData>& images,
                                         Tensor* output,
                                         const std::vector<int>& dst_size,
                                         float pad_value,
                                         std::vector<LetterBoxRecord>* letter_box_records,
                                         cudaStream_t stream = nullptr,
                                         CudaOutputBufferPool* dst_pool = nullptr);
}
