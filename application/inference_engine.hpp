#pragma once
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "capi/common/md_types.h"
#include "csrc/vision/detection/ultralytics_det.h"
#include "csrc/vision/classification/classification.h"
#include "csrc/vision/face/face_det/scrfd.h"
#include "csrc/vision/common/image_data.h"

#include "config.hpp"

struct DetectionBox {
    float x, y, w, h;
    float score;
    int label_id;
    std::string label_name;
};

/// 人脸关键点（最多 5 个：左右眼/鼻尖/左右嘴角）
struct FaceKeypoint {
    float x, y;
};

struct InferResult {
    std::string model_name;
    std::string type;                 // detection / classification / face_detection
    std::vector<DetectionBox> boxes;
    // 仅人脸：每个 box 对应一组关键点
    std::vector<std::vector<FaceKeypoint>> keypoints;
};

/// 推理引擎包装器：每路 Pipeline 独占一个实例
/// 通过 model.clone() 共享底层 ORT Session，但 pre/post processor 各实例独立（线程安全）
class InferenceEngine {
public:
    InferenceEngine() = default;
    ~InferenceEngine() { unload(); }

    /// 从配置加载（会创建新的 ORT Session，适合第一次加载）
    bool load(const ModelConfig& cfg);

    /// 从已有实例克隆 detection 模型（共享 ORT Session），SCRFD独立加载
    /// 适用于多路场景：第一路 load，后续 clone
    bool clone_detection_from(const modeldeploy::vision::detection::UltralyticsDet& proto,
                              const ModelConfig& cfg);

    void unload();
    bool is_loaded() const { return loaded_; }

    bool infer(const modeldeploy::vision::ImageData& image, InferResult* result);

    const ModelConfig& config() const { return cfg_; }
    std::pair<int, int> input_size() const {
        return {cfg_.input_size[0], cfg_.input_size[1]};
    }

    static std::string make_cache_key(const ModelConfig& cfg);

private:
    bool loaded_ = false;
    ModelConfig cfg_;

    std::unique_ptr<modeldeploy::vision::detection::UltralyticsDet> det_model_;
    std::unique_ptr<modeldeploy::vision::classification::Classification> cls_model_;
    std::unique_ptr<modeldeploy::vision::face::Scrfd> face_model_;

    bool infer_detection(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_classification(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_face(const modeldeploy::vision::ImageData& image, InferResult* result);
};
