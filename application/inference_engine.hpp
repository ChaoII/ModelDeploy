#pragma once
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "capi/common/md_types.h"
#include "csrc/vision/detection/ultralytics_det.h"
#include "csrc/vision/classification/classification.h"
#include "csrc/vision/common/image_data.h"

#include "config.hpp"

struct DetectionBox {
    float x, y, w, h;
    float score;
    int label_id;
    std::string label_name;
};

struct InferResult {
    std::string model_name;
    std::string type;
    std::vector<DetectionBox> boxes;
};

class InferenceEngine {
public:
    InferenceEngine() = default;
    ~InferenceEngine() { unload(); }

    bool load(const ModelConfig& cfg);
    void unload();
    bool is_loaded() const { return loaded_; }

    bool infer(const modeldeploy::vision::ImageData& image, InferResult* result);

    const ModelConfig& config() const { return cfg_; }
    std::pair<int, int> input_size() const {
        return {cfg_.input_size[0], cfg_.input_size[1]};
    }

private:
    bool loaded_ = false;
    ModelConfig cfg_;

    // C++ 模型实例（共用体风格，实际只用一个）
    std::unique_ptr<modeldeploy::vision::detection::UltralyticsDet> det_model_;
    std::unique_ptr<modeldeploy::vision::classification::Classification> cls_model_;

    bool infer_detection(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_classification(const modeldeploy::vision::ImageData& image, InferResult* result);
};
