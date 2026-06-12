#pragma once
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "capi/common/md_types.h"
#include "capi/common/md_decl.h"
#include "capi/utils/md_image_capi.h"
#include "config.hpp"

/// 模型推理结果
struct InferResult {
    std::string model_name;
    std::string type;  // "detection" | "classification" | etc.
    std::vector<struct DetectionBox> boxes;
};

struct DetectionBox {
    float x, y, w, h;        // 归一化或像素坐标
    float score;
    int label_id;
    std::string label_name;
};

/// ModelDeploy 推理引擎封装
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    /// 加载模型
    bool load(const ModelConfig& cfg);

    /// 卸载模型
    void unload();

    /// 是否已加载
    bool is_loaded() const { return loaded_; }

    /// 对 MDImage 执行推理
    bool infer(MDImage* image, InferResult* result);

    /// 获取模型配置
    const ModelConfig& config() const { return cfg_; }

    /// 获取模型输入尺寸
    std::pair<int, int> input_size() const {
        return {cfg_.input_size[0], cfg_.input_size[1]};
    }

private:
    bool loaded_ = false;
    ModelConfig cfg_;
    MDModel model_{};

    // 模型类型对应的 CAPI 函数
    bool infer_detection(MDImage* image, InferResult* result);
    bool infer_classification(MDImage* image, InferResult* result);
};
