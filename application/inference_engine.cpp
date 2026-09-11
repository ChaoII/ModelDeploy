#include "inference_engine.hpp"
#include "runtime_factory.hpp"
#include "csrc/runtime/runtime_option.h"
#include "csrc/vision/common/result.h"
#include "csrc/vision/common/struct.h"
#include <iostream>
#include <filesystem>

using namespace modeldeploy;
using namespace modeldeploy::vision;

bool InferenceEngine::load(const ModelConfig& cfg) {
    if (loaded_) unload();
    cfg_ = cfg;

    RuntimeOption opt = build_runtime_option(cfg);

    try {
        if (cfg.type == "detection") {
            det_model_ = std::make_unique<detection::UltralyticsDet>(cfg.path, opt);
            if (!det_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] Detection model init failed: " << cfg.path << std::endl;
                det_model_.reset();
                return false;
            }
            if (cfg_.input_size.size() == 2)
                det_model_->get_preprocessor().set_size(cfg_.input_size);
        } else if (cfg.type == "face_detection") {
            face_model_ = std::make_unique<face::Scrfd>(cfg.path, opt);
            if (!face_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] Face model init failed: " << cfg.path << std::endl;
                face_model_.reset();
                return false;
            }
            if (cfg_.input_size.size() == 2)
                face_model_->get_preprocessor().set_size(cfg_.input_size);
        } else if (cfg.type == "classification") {
            cls_model_ = std::make_unique<classification::Classification>(cfg.path, opt);
            if (!cls_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] Classification model init failed: " << cfg.path << std::endl;
                cls_model_.reset();
                return false;
            }
        } else {
            std::cerr << "[InferenceEngine] Unsupported model type: " << cfg.type << std::endl;
            return false;
        }

        // 将配置中的置信度阈值传递到模型后处理
        if (det_model_) {
            det_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);
        }
        if (face_model_) {
            face_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);
        }
    } catch (const std::exception& e) {
        std::cerr << "[InferenceEngine] Exception loading model: " << e.what() << std::endl;
        return false;
    }

    loaded_ = true;
    std::cout << "[InferenceEngine] Loaded " << cfg.name << " [" << cfg.type
              << "] " << cfg.path << " backend=" << cfg.backend << std::endl;
    return true;
}

std::string InferenceEngine::make_cache_key(const ModelConfig& cfg) {
    return cfg.path + "|" + cfg.backend + "|" + cfg.device + "|" + cfg.type
           + "|" + std::to_string(cfg.input_size[0]) + "x" + std::to_string(cfg.input_size[1]);
}

bool InferenceEngine::clone_detection_from(
    const detection::UltralyticsDet& proto, const ModelConfig& cfg) {
    if (loaded_) this->unload();
    cfg_ = cfg;

    // clone() = 新 instance + 共享 Runtime (含 ORT Session)
    auto cloned = proto.clone();
    if (!cloned || !cloned->is_initialized()) {
        std::cerr << "[InferenceEngine] Failed to clone detection model" << std::endl;
        return false;
    }
    det_model_ = std::move(cloned);
    if (cfg_.input_size.size() == 2)
        det_model_->get_preprocessor().set_size(cfg_.input_size);
    det_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);

    loaded_ = true;
    std::cout << "[InferenceEngine] Cloned detection: " << cfg.name
              << " (shared ORT session)" << std::endl;
    return true;
}

void InferenceEngine::adopt_face_model(
    std::unique_ptr<face::Scrfd> model, const ModelConfig& cfg) {
    if (loaded_) this->unload();
    cfg_ = cfg;
    face_model_ = std::move(model);
    face_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);
    loaded_ = true;
    std::cout << "[InferenceEngine] Adopted face: " << cfg.name
              << " (shared ORT session)" << std::endl;
}

void InferenceEngine::set_shared_detector(std::shared_ptr<BatchedDetector> det,
                                          const ModelConfig& cfg) {
    if (loaded_) this->unload();
    cfg_ = cfg;
    shared_det_ = std::move(det);
    loaded_ = shared_det_ && shared_det_->ok();
    if (!loaded_) {
        std::cerr << "[InferenceEngine] shared detector unavailable: "
                  << (shared_det_ ? shared_det_->error() : "null") << std::endl;
    }
}

bool InferenceEngine::predict_detection(const ImageData& image,
                                        std::vector<DetectionResult>* out) {
    if (shared_det_) return shared_det_->predict(image, out);
    if (!det_model_) return false;
    return det_model_->predict(image, out);
}

void InferenceEngine::unload() {
    det_model_.reset();
    cls_model_.reset();
    face_model_.reset();
    shared_det_.reset();
    loaded_ = false;
}

bool InferenceEngine::infer(const ImageData& image, InferResult* result) {
    if (!loaded_ || !result) return false;
    result->model_name = cfg_.name;
    result->type = cfg_.type;

    if (cfg_.type == "detection")
        return infer_detection(image, result);
    if (cfg_.type == "face_detection")
        return infer_face(image, result);
    if (cfg_.type == "classification")
        return infer_classification(image, result);
    return false;
}

bool InferenceEngine::infer_detection(const ImageData& image, InferResult* result) {
    if (!shared_det_ && !det_model_) return false;
    std::vector<DetectionResult> det_results;
    if (!predict_detection(image, &det_results)) return false;
    for (auto& d : det_results) {
        DetectionBox box;
        box.x = d.box.x; box.y = d.box.y;
        box.w = d.box.width; box.h = d.box.height;
        box.score = d.score;
        box.label_id = d.label_id;
        result->boxes.push_back(box);
    }
    return true;
}

bool InferenceEngine::infer_classification(const ImageData& image, InferResult* result) {
    if (!cls_model_) return false;
    ClassifyResult cls_result;
    if (!cls_model_->predict(image, &cls_result)) return false;
    for (size_t i = 0; i < cls_result.label_ids.size(); ++i) {
        DetectionBox box;
        box.label_id = cls_result.label_ids[i];
        box.score = cls_result.scores[i];
        result->boxes.push_back(box);
    }
    return true;
}

bool InferenceEngine::infer_face(const ImageData& image, InferResult* result) {
    if (!face_model_) return false;
    std::vector<KeyPointsResult> face_results;
    if (!face_model_->predict(image, &face_results)) return false;
    float thresh = cfg_.confidence_threshold;
    for (auto& f : face_results) {
        if (f.score < thresh) continue;
        DetectionBox box;
        box.x = f.box.x; box.y = f.box.y;
        box.w = f.box.width; box.h = f.box.height;
        box.score = f.score;
        box.label_id = f.label_id;
        result->boxes.push_back(box);
        std::vector<FaceKeypoint> kps;
        kps.reserve(f.keypoints.size());
        for (const auto& p : f.keypoints) {
            FaceKeypoint kp;
            kp.x = p.x;
            kp.y = p.y;
            kps.push_back(kp);
        }
        result->keypoints.push_back(std::move(kps));
    }
    return true;
}
