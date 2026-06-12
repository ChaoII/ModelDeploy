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

private:
    DrawConfig cfg_;

    void draw_detection(modeldeploy::vision::ImageData& image,
                        const InferResult& result);
};
