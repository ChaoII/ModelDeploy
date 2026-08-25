//
// Created by aichao on 2025/8/2.
//
#pragma once

#include "vision/processors/cpu/cpu_processor_backend.h"
#include "vision/processors/cuda/cuda_output_pool.h"

namespace modeldeploy::vision {
    // CUDA backend 继承 CPU 实现，仅覆写 yolo 系算子为 CUDA kernel
    class MODELDEPLOY_CXX_EXPORT CudaProcessorBackend : public CpuProcessorBackend {
    public:
        CudaProcessorBackend() = default;

        bool yolo_preprocess(const ImageData& image, Tensor* out,
                             const std::vector<int>& dst_size,
                             float pad_val, LetterBoxRecord* record) override;
    bool yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                              const std::vector<int>& src_size,
                              int step_y, int step_uv, Tensor* out,
                              const std::vector<int>& dst_size,
                              float pad_val, LetterBoxRecord* record,
                              Device src_device = Device::CPU) override;
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
        bool scrfd_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                    const std::vector<int>& dst_size,
                                    float pad_val,
                                    std::vector<LetterBoxRecord>* records) override;
        bool fusion_resize_pad_normalize_permute(
            const std::vector<ImageData>& images, Tensor* out,
            const std::vector<std::array<int, 2>>& resize_sizes,
            const std::vector<int>& dst_size,
            const std::vector<float>& mean, const std::vector<float>& std,
            float pad_value) override;

        // ── NV12 设备侧就地绘制（覆写为 CUDA kernel）──
        bool draw_rect_nv12(ImageData& frame,
                            float x, float y, float w, float h,
                            float r, float g, float b, int thickness) override;
        bool draw_polygon_nv12(ImageData& frame,
                               const std::vector<Point2f>& pts,
                               float r, float g, float b, int thickness) override;
        bool draw_points_nv12(ImageData& frame,
                              const std::vector<Point3f>& pts,
                              float r, float g, float b, int radius) override;
        bool draw_text_nv12(ImageData& frame,
                            float x, float y, const std::string& text,
                            float r, float g, float b, int font_size) override;

        // ── 中间图像算子：设备帧 → NV12 设备侧裁剪；其它格式未实现 → false ──
        bool crop(const ImageData& image, float x, float y, float w, float h,
                  ImageData* out) override;
        bool rotate(const ImageData& image, RotateFlags flag, ImageData* out) override {
            (void)image; (void)flag; (void)out;
            return false;
        }
        bool cvt_color(const ImageData& image, ColorConvertType type, ImageData* out) override {
            (void)image; (void)type; (void)out;
            return false;
        }

        ~CudaProcessorBackend() override;

    private:
        // 持久 CUDA stream（避免每帧 create/destroy；.cpp 中惰性创建）
        void* stream_ = nullptr;
        // 预处理输出设备缓冲池（与 stream 同生命周期，析构自动释放）
        CudaOutputBufferPool out_pool_;
    };
} // namespace modeldeploy::vision
