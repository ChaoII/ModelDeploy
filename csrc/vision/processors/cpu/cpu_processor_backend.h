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
    bool scrfd_preprocess(const ImageData& image, Tensor* out,
                          const std::vector<int>& dst_size,
                          float pad_val, LetterBoxRecord* record) override;
    bool resize(const ImageData& image, ImageData* out,
                int width, int height) override;
    bool convert(const ImageData& image, ImageData* out,
                 const std::vector<float>& alpha,
                 const std::vector<float>& beta) override;
    bool cast(const ImageData& image, ImageData* out,
              const std::string& dtype) override;
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
             const std::vector<int>& top,
             const std::vector<int>& bottom) override;
    bool hwc2chw(const ImageData& image, Tensor* out) override;
    bool normalize_and_permute(const ImageData& image, Tensor* out,
                               const std::vector<float>& mean,
                               const std::vector<float>& std,
                               bool scale = true) override;
    bool nv12_to_bgr(const uint8_t* y, const uint8_t* uv,
                     int width, int height, ImageData* out) override;
};

} // namespace modeldeploy::vision
