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
                                  float pad_val, LetterBoxRecord* record,
                                  Device src_device = Device::CPU) override;
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
        bool scrfd_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                    const std::vector<int>& dst_size,
                                    float pad_val, std::vector<LetterBoxRecord>* records) override;
        bool resize(const ImageData& image, ImageData* out, int width, int height) override;
        bool crop(const ImageData& image, float x, float y, float w, float h, ImageData* out) override;
        bool rotate(const ImageData& image, RotateFlags flag, ImageData* out) override;
        bool cvt_color(const ImageData& image, ColorConvertType type, ImageData* out) override;
        bool rotate_crop(const ImageData& image, std::array<float, 8> box, ImageData* out) override;
        bool fusion_resize_pad_normalize_permute(
            const std::vector<ImageData>& images, Tensor* out,
            const std::vector<std::array<int, 2>>& resize_sizes,
            const std::vector<int>& dst_size,
            const std::vector<float>& mean, const std::vector<float>& std,
            float pad_value) override;
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
        bool fused_preprocess_bilinear(
            const ImageData& image, Tensor* out,
            const std::vector<int>& dst_size,
            float origin_x, float origin_y,
            float scale_x, float scale_y,
            const std::vector<float>& alpha,
            const std::vector<float>& beta,
            bool swap_rb, float pad_value) override;
        bool fused_color_matrix_preprocess(
            const ImageData& image, Tensor* out,
            const std::vector<int>& dst_size,
            float origin_x, float origin_y,
            float scale_x, float scale_y,
            const float mat[3][3], const float bias[3],
            float pad_value) override;

        bool draw_rect_nv12(ImageData& frame, float x, float y, float w, float h,
                            float r, float g, float b, int thickness) override;
        bool draw_polygon_nv12(ImageData& frame, const std::vector<Point2f>& pts,
                               float r, float g, float b, int thickness) override;
        bool draw_points_nv12(ImageData& frame, const std::vector<Point3f>& pts,
                              float r, float g, float b, int radius) override;
        bool draw_text_nv12(ImageData& frame, float x, float y, const std::string& text,
                            float r, float g, float b, int font_size) override;
    };
} // namespace modeldeploy::vision
