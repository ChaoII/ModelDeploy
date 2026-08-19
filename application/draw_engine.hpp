#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>

#include "config.hpp"
#include "inference_engine.hpp"
#include "csrc/vision/common/image_data.h"
#include "csrc/vision/common/result.h"
#include "csrc/vision/processors/processor_backend.h"

/// 绘制引擎：使用 ModelDeploy 的 vis_det 绘制检测结果
class DrawEngine {
public:
    explicit DrawEngine(const DrawConfig& cfg);
    ~DrawEngine() = default;

    /// 在 ImageData 上绘制所有模型的结果
    void draw(modeldeploy::vision::ImageData& image,
              const std::vector<InferResult>& results);

    /// 统一设备/CPU NV12 就地绘制（按 frame.device() 分派到 CUDA/TPU/CPU backend 的
    /// draw_*_nv12，零拷贝写设备 y/uv 平面）。仅对 NV12 帧可用；非 NV12（如 packed BGR）
    /// 返回 false，由调用方回退 CPU draw()。
    bool draw_gpu(modeldeploy::vision::ImageData& image,
                  const std::vector<InferResult>& results,
                  bool show_label = true, bool show_score = true);

private:
    DrawConfig cfg_;

    void draw_detection(modeldeploy::vision::ImageData& image,
                        const InferResult& result);
    void draw_face(modeldeploy::vision::ImageData& image,
                   const InferResult& result);

    // 复用处理器 backend，避免每帧 create_processor_backend（其含惰性建流等
    // CUDA 上下文同步操作，高频调用会与 batch_predict 争用）
    mutable std::mutex backend_mtx_;
    mutable std::map<modeldeploy::Device,
                     std::unique_ptr<modeldeploy::vision::VisionProcessorBackend>> backends_;
};
