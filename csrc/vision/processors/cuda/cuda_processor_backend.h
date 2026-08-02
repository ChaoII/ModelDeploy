//
// Created by aichao on 2025/8/2.
//
#pragma once

#include "vision/processors/cpu/cpu_processor_backend.h"

namespace modeldeploy::vision {

// CUDA backend 继承 CPU 实现，仅覆写 yolo 系算子为 CUDA kernel
class CudaProcessorBackend : public CpuProcessorBackend {
public:
    CudaProcessorBackend() = default;
    ~CudaProcessorBackend() override = default;

    bool yolo_preprocess(const ImageData& image, Tensor* out,
                         const std::vector<int>& dst_size,
                         float pad_val, LetterBoxRecord* record) override;
    bool yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                              const std::vector<int>& src_size,
                              int step_y, int step_uv, Tensor* out,
                              const std::vector<int>& dst_size,
                              float pad_val, LetterBoxRecord* record) override;
};

} // namespace modeldeploy::vision
