#pragma once
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "csrc/vision/detection/ultralytics_det.h"
#include "csrc/vision/classification/classification.h"
#include "csrc/vision/face/face_det/scrfd.h"
#include "csrc/vision/common/image_data.h"
#include "csrc/utils/benchmark.h"

#include "config.hpp"
#include "batched_detector.hpp"

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

    /// 接管一份已 clone 的 face 模型（共享 Runtime）
    void adopt_face_model(std::unique_ptr<modeldeploy::vision::face::Scrfd> model,
                          const ModelConfig& cfg);

    /// 接管共享批处理检测器（多路共享一个模型实例 + 批推理；GPU 高利用率路径）
    void set_shared_detector(std::shared_ptr<BatchedDetector> det, const ModelConfig& cfg);
    BatchedDetector* shared_detector() const { return shared_det_.get(); }

    void unload();
    bool is_loaded() const { return loaded_; }

    bool infer(const modeldeploy::vision::ImageData& image, InferResult* result);

    /// 获取底层 detection 模型指针（BatchScheduler 需要直接调用 batch_predict）
    modeldeploy::vision::detection::UltralyticsDet* det_model() {
        return shared_det_ ? shared_det_->model() : det_model_.get();
    }

    /// detection 统一推理入口：共享批处理检测器优先（批推理），否则本实例模型
    bool predict_detection(const modeldeploy::vision::ImageData& image,
                           std::vector<modeldeploy::vision::DetectionResult>* out);

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
    std::shared_ptr<BatchedDetector> shared_det_;   // 共享批处理检测器（多路共享，优先）

    bool infer_detection(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_classification(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_face(const modeldeploy::vision::ImageData& image, InferResult* result);
};
