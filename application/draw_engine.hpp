#pragma once
#include <string>
#include <vector>
#include <map>

#include "config.hpp"
#include "inference_engine.hpp"
#include "csrc/vision/common/image_data.h"
#include "csrc/vision/common/result.h"

/// 绘制引擎：使用 ModelDeploy 的 vis_det 绘制检测结果
class DrawEngine {
public:
    explicit DrawEngine(const DrawConfig& cfg);
    ~DrawEngine() = default;

    /// 在 ImageData 上绘制所有模型的结果
    void draw(modeldeploy::vision::ImageData& image,
              const std::vector<InferResult>& results);

    /// GPU 绘制：CUDA 不可用时返回 false（调用方回退 CPU draw）
    bool draw_gpu(modeldeploy::vision::ImageData& image,
                  const std::vector<InferResult>& results,
                  bool show_label = true, bool show_score = true);

private:
    DrawConfig cfg_;

    void draw_detection(modeldeploy::vision::ImageData& image,
                        const InferResult& result);
    void draw_face(modeldeploy::vision::ImageData& image,
                   const InferResult& result);
};
