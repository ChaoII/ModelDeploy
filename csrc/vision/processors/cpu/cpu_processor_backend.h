//
// Created by aichao on 2025/8/2.
//
#pragma once

#include "vision/processors/processor_backend.h"
#include "core/md_decl.h"

namespace modeldeploy::vision {

class MODELDEPLOY_CXX_EXPORT CpuProcessorBackend : public VisionProcessorBackend {
public:
    CpuProcessorBackend() = default;
    ~CpuProcessorBackend() override = default;

    bool yolo_preprocess(const ImageData& image, Tensor* out,
                         const std::vector<int>& dst_size,
                         float pad_val, LetterBoxRecord* record) override;
    bool yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                              const std::vector<int>& src_size,
                              int step_y, int step_uv, Tensor* out,
                              const std::vector<int>& dst_size,
                              float pad_val, LetterBoxRecord* record) override;
    bool letterbox(const ImageData& image, ImageData* out,
                   const std::vector<int>& dst_size,
                   const std::vector<float>& padding_value,
                   LetterBoxRecord* record) override;
    bool yolo_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                               const std::vector<int>& dst_size,
                               float pad_val,
                               std::vector<LetterBoxRecord>* records) override;
    bool scrfd_preprocess(const ImageData& image, Tensor* out,
                          const std::vector<int>& dst_size,
                          float pad_val, LetterBoxRecord* record) override;
    bool resize(const ImageData& image, ImageData* out,
                int width, int height) override;
    bool convert(const ImageData& image, ImageData* out,
                 const std::vector<float>& alpha,
                 const std::vector<float>& beta) override;
    bool cast(const ImageData& image, ImageData* out,
              const std::string& dtype, bool scale = true) override;
    bool convert_and_permute(const ImageData& image, Tensor* out,
                             const std::vector<float>& alpha,
                             const std::vector<float>& beta,
                             bool swap_rb) override;
    bool fusion_resize_pad_normalize_permute(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<std::array<int, 2>>& resize_sizes,
        const std::vector<int>& dst_size,
        const std::vector<float>& mean, const std::vector<float>& std,
        float pad_value) override;
    bool normalize(const ImageData& image, ImageData* out,
                   const std::vector<float>& mean,
                   const std::vector<float>& std,
                   bool scale, bool swap_rb) override;
    bool convert_to(const ImageData& image, ImageData* out,
                    const std::string& dst_format) override;
    bool center_crop(const ImageData& image, ImageData* out,
                     int width, int height) override;
    bool pad(const ImageData& image, ImageData* out,
             int top, int bottom, int left, int right,
             float value = 0.0f) override;
    bool hwc2chw(const ImageData& image, Tensor* out) override;
    bool normalize_and_permute(const ImageData& image, Tensor* out,
                               const std::vector<float>& mean,
                               const std::vector<float>& std,
                               bool scale = true) override;
    bool nv12_to_bgr(const uint8_t* y, const uint8_t* uv,
                     int width, int height, ImageData* out) override;
    bool fused_preprocess(
        const ImageData& image, Tensor* out,
        const std::vector<int>& dst_size,
        float origin_x, float origin_y,
        float scale_x, float scale_y,
        const std::vector<float>& alpha,
        const std::vector<float>& beta,
        bool swap_rb, float pad_value) override;
};

} // namespace modeldeploy::vision
