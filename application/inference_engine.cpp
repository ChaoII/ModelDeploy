#include "inference_engine.hpp"
#include "csrc/runtime/runtime_option.h"
#include "csrc/vision/common/result.h"
#include <iostream>

using namespace modeldeploy;
using namespace modeldeploy::vision;

bool InferenceEngine::load(const ModelConfig& cfg) {
    if (loaded_) unload();
    cfg_ = cfg;

    RuntimeOption opt;
    if (cfg.device == "gpu") {
        opt.use_gpu(0);
    }
    opt.set_cpu_thread_num(4);

    if (cfg.backend == "trt") {
        opt.use_trt_backend();
    } else if (cfg.backend == "mnn") {
        opt.use_mnn_backend();
    } else {
        opt.use_ort_backend();
    }

    try {
        if (cfg.type == "detection" || cfg.type == "face_detection") {
            det_model_ = std::make_unique<detection::UltralyticsDet>(cfg.path, opt);
            if (!det_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] Detection model init failed: " << cfg.path << std::endl;
                det_model_.reset();
                return false;
            }
            // 设置预处理
            if (cfg_.input_size.size() == 2)
                det_model_->get_preprocessor().set_size(cfg_.input_size);
            if (cfg.device == "gpu")
                det_model_->get_preprocessor().use_cuda_preproc();
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
    } catch (const std::exception& e) {
        std::cerr << "[InferenceEngine] Exception loading model: " << e.what() << std::endl;
        return false;
    }

    loaded_ = true;
    std::cout << "[InferenceEngine] Loaded " << cfg.name << " [" << cfg.type
              << "] " << cfg.path << std::endl;
    return true;
}

void InferenceEngine::unload() {
    det_model_.reset();
    cls_model_.reset();
    loaded_ = false;
}

bool InferenceEngine::infer(const ImageData& image, InferResult* result) {
    if (!loaded_ || !result) return false;
    result->model_name = cfg_.name;
    result->type = cfg_.type;

    if (cfg_.type == "detection" || cfg_.type == "face_detection")
        return infer_detection(image, result);
    if (cfg_.type == "classification")
        return infer_classification(image, result);
    return false;
}

bool InferenceEngine::infer_detection(const ImageData& image, InferResult* result) {
    if (!det_model_) return false;

    std::vector<DetectionResult> det_results;
    if (!det_model_->predict(image, &det_results)) {
        return false;
    }

    for (auto& d : det_results) {
        DetectionBox box;
        box.x = d.box.x;
        box.y = d.box.y;
        box.w = d.box.width;
        box.h = d.box.height;
        box.score = d.score;
        box.label_id = d.label_id;
        result->boxes.push_back(box);
    }
    return true;
}

bool InferenceEngine::infer_classification(const ImageData& image, InferResult* result) {
    if (!cls_model_) return false;

    ClassifyResult cls_result;
    if (!cls_model_->predict(image, &cls_result)) {
        return false;
    }

    for (size_t i = 0; i < cls_result.label_ids.size(); ++i) {
        DetectionBox box;
        box.label_id = cls_result.label_ids[i];
        box.score = cls_result.scores[i];
        result->boxes.push_back(box);
    }
    return true;
}
