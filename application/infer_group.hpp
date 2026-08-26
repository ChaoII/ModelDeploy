#pragma once
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "config.hpp"
#include "inference_engine.hpp"
#include "csrc/vision/common/image_data.h"
#include "csrc/vision/common/result.h"

/// 多模型调度组：管理同一路视频上的多个推理模型（纯模型薄封装）
/// 每帧串行跑各模型；detection 模型额外缓存 SDK DetectionResult 供 pipeline 设备绘制
class InferGroup {
public:
    using ModelFactory = std::function<std::unique_ptr<InferenceEngine>(const ModelConfig&)>;
    bool load_models(const std::vector<ModelConfig>& mcfgs, ModelFactory factory);
    bool add_model(const ModelConfig& mcfg, ModelFactory factory);
    bool remove_model(const std::string& name);
    void clear();
    bool empty() const;
    /// 对一帧依次跑全部模型；sdk_dets 收集各 detection 模型的 SDK DetectionResult（供设备绘制）；
    /// non_det 收集各非 detection 模型（face/classification）的 InferResult（供 DrawEngine 标注，可空）
    bool run_models(const modeldeploy::vision::ImageData& frame,
                    std::vector<std::pair<std::string, std::vector<modeldeploy::vision::DetectionResult>>>* sdk_dets,
                    std::vector<std::pair<std::string, InferResult>>* non_det = nullptr);
    modeldeploy::vision::detection::UltralyticsDet* det_model(const std::string& name);
    /// 按模型名取 ModelConfig（含动态 add_model 加入的模型），未找到返回 nullptr
    const ModelConfig* config_of(const std::string& name) const;
private:
    std::vector<std::unique_ptr<InferenceEngine>> engines_;
};
