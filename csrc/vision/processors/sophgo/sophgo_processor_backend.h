//
// Created by aichao on 2025/8/2.
//
#pragma once

#include "vision/processors/processor_backend.h"
#include "core/md_decl.h"

namespace modeldeploy::vision {
    // Sophgo 算能 TPU 前后处理后端：BMCV（resize/letterbox/仿射/通道重排）+ bm_image。
    // 仅在 ENABLE_SOPHGO 编译（Linux + SOPHON-Sail）。
    class MODELDEPLOY_CXX_EXPORT SophgoProcessorBackend : public VisionProcessorBackend {
    public:
        explicit SophgoProcessorBackend(int device_id = 0);
        ~SophgoProcessorBackend() override;

        bool yolo_preprocess(const ImageData& image, Tensor* out,
                             const std::vector<int>& dst_size,
                             float pad_val, LetterBoxRecord* record) override;
        bool fused_preprocess(const ImageData& image, Tensor* out,
                              const std::vector<int>& dst_size,
                              float origin_x, float origin_y,
                              float scale_x, float scale_y,
                              const std::vector<float>& alpha,
                              const std::vector<float>& beta,
                              bool swap_rb, float pad_value) override;

        // 其余算子：BMCV 可表达的在此实现，否则走 CPU 兜底
        bool yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                                  const std::vector<int>& src_size,
                                  int step_y, int step_uv, Tensor* out,
                                  const std::vector<int>& dst_size,
                                  float pad_val, LetterBoxRecord* record) override;
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
        bool letterbox(const ImageData& image, ImageData* out,
                       const std::vector<int>& dst_size,
                       const std::vector<float>& padding_value,
                       LetterBoxRecord* record) override;
        bool resize(const ImageData& image, ImageData* out,
                    int width, int height) override;
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
                 float value) override;
        bool hwc2chw(const ImageData& image, Tensor* out) override;
        bool normalize_and_permute(const ImageData& image, Tensor* out,
                                   const std::vector<float>& mean,
                                   const std::vector<float>& std,
                                   bool scale) override;
        bool convert(const ImageData& image, ImageData* out,
                     const std::vector<float>& alpha,
                     const std::vector<float>& beta) override;
        bool cast(const ImageData& image, ImageData* out,
                  const std::string& dtype, bool scale) override;
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
        bool nv12_to_bgr(const uint8_t* y, const uint8_t* uv,
                         int width, int height, ImageData* out) override;
        bool process_device_image(void* device_image, int width, int height,
                                  Tensor* out, LetterBoxRecord* record) override;
        bool scrfd_preprocess(const ImageData& image, Tensor* out,
                              const std::vector<int>& dst_size,
                              float pad_val, LetterBoxRecord* record) override;

        // 设备内存融合预处理（零拷贝，参考官方 SOPHON-DEMO）：BMCV 结果直接写入 bmrt_tensor
        // 分配的输入设备内存（input_mem = bm_device_mem_t*），跳过 D2H/H2D。
        // out_img: 输出 bm_image*（FP32 RGB_PLANAR，已 attach 到 input_mem，由本函数创建，
        //          调用方用完需 md_bmcv_image_destroy 释放；其设备内存即 input_mem 由 backend 管理）。
        // 返回 true 成功（后续调用方直接用 input_mem + shape 调 infer_device 推理）
        bool fused_preprocess_device(const ImageData& image, void** out_img, void* input_mem,
                                     int* dst_w, int* dst_h,
                                     float origin_x, float origin_y,
                                     float scale_x, float scale_y,
                                     const std::vector<float>& alpha,
                                     const std::vector<float>& beta,
                                     bool swap_rb, float pad_value);

        // 使用外部 bm_handle（SophgoBackend 的 bmrt handle），保证 D2D 零拷贝在同一设备上下文。
        // 替代构造时自行 bm_dev_request 的 handle。
        void use_external_handle(void* handle);

    private:
        int device_id_ = 0;
        // 不透明句柄：实际为 bm_handle_t（见 .cpp，避免头文件引入 libsophon）
        void* handle_ = nullptr;
        void* bmcv_ = nullptr;
        // 是否为外部共享 handle（SophgoBackend 的），析构时不释放
        bool external_handle_ = false;
        // CPU 兜底
        std::unique_ptr<VisionProcessorBackend> cpu_fallback_;
    };
} // namespace modeldeploy::vision
