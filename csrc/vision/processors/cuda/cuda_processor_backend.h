//
// Created by aichao on 2025/8/2.
//
#pragma once

#include "vision/processors/cpu/cpu_processor_backend.h"

namespace modeldeploy::vision {

// CUDA backend 继承 CPU 实现，仅覆写 yolo 系算子为 CUDA kernel
class MODELDEPLOY_CXX_EXPORT CudaProcessorBackend : public CpuProcessorBackend {
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
    bool fused_preprocess(
        const ImageData& image, Tensor* out,
        const std::vector<int>& dst_size,
        float origin_x, float origin_y,
        float scale_x, float scale_y,
        const std::vector<float>& alpha,
        const std::vector<float>& beta,
        bool swap_rb, float pad_value) override;
    bool yolo_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                               const std::vector<int>& dst_size,
                               float pad_val,
                               std::vector<LetterBoxRecord>* records) override;
    bool fused_preprocess_batch(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<int>& dst_size,
        const std::vector<float>& origins_x, const std::vector<float>& origins_y,
        const std::vector<float>& scales_x, const std::vector<float>& scales_y,
        const std::vector<float>& alpha, const std::vector<float>& beta,
        bool swap_rb, float pad_value) override;
    bool scrfd_preprocess(const ImageData& image, Tensor* out,
                          const std::vector<int>& dst_size,
                          float pad_val, LetterBoxRecord* record) override;
};

} // namespace modeldeploy::vision
