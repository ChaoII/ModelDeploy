//
// Created by aichao on 2025/8/2.
//
#pragma once

#include "vision/processors/cpu/cpu_processor_backend.h"
#include "core/md_decl.h"

namespace modeldeploy::vision {
    // Sophgo 算能 TPU 预处理后端：BMCV（letterbox/仿射/通道重排/NV12）+ 设备内存零拷贝。
    // 继承 CpuProcessorBackend：只有 yolo_preprocess / fused_preprocess / yolo_preprocess_nv12
    // 用 BMCV 硬件实现并产出 Device::TPU Tensor（零拷贝直接喂 SophgoBackend::infer），
    // 其余算子自动回退到 CPU 实现；BMCV 调用失败时同样回退 CPU。
    // 仅在 ENABLE_SOPHGO 编译。
    class MODELDEPLOY_CXX_EXPORT SophgoProcessorBackend : public CpuProcessorBackend {
    public:
        explicit SophgoProcessorBackend(int device_id = 0);
        ~SophgoProcessorBackend() override;

        bool yolo_preprocess(const ImageData& image, Tensor* out,
                             const std::vector<int>& dst_size,
                             float pad_val, LetterBoxRecord* record) override;
        bool yolo_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                   const std::vector<int>& dst_size,
                                   float pad_val,
                                   std::vector<LetterBoxRecord>* records) override;
        bool fused_preprocess(const ImageData& image, Tensor* out,
                              const std::vector<int>& dst_size,
                              float origin_x, float origin_y,
                              float scale_x, float scale_y,
                              const std::vector<float>& alpha,
                              const std::vector<float>& beta,
                              bool swap_rb, float pad_value) override;
        bool yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                                  const std::vector<int>& src_size,
                                  int step_y, int step_uv, Tensor* out,
                                  const std::vector<int>& dst_size,
                                  float pad_val, LetterBoxRecord* record,
                                  Device src_device = Device::CPU) override;

    private:
        // 确保已分配可容纳单张 dst 尺寸图像的设备内存（bm_device_mem_t* 或 nullptr）
        void* ensure_input_mem(int dst_w, int dst_h);
        // 内部将 BMCV 写入的 in_mem_ 包装为 Device::TPU Tensor
        bool finish_tpu_tensor(Tensor* out, int dst_w, int dst_h,
                               const std::string& name = "");

        int device_id_ = 0;
        // 不透明句柄：实际为 bm_handle_t（见 .cpp，避免头文件引入 libsophon）
        void* handle_ = nullptr;
        // 缓存的输入设备内存（bm_device_mem_t*，零拷贝推理输入），按需分配、复用、析构释放
        void* in_mem_ = nullptr;
        size_t in_mem_bytes_ = 0;
        int cached_w_ = 0;
        int cached_h_ = 0;
    };
} // namespace modeldeploy::vision
