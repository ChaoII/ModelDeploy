#include "inference_engine.hpp"
#include <cstring>
#include <iostream>

#include "capi/vision/detection/detection_capi.h"
#include "capi/vision/classification/classification_capi.h"

InferenceEngine::InferenceEngine() {
    memset(&model_, 0, sizeof(model_));
}

InferenceEngine::~InferenceEngine() {
    unload();
}

bool InferenceEngine::load(const ModelConfig& cfg) {
    if (loaded_) unload();
    cfg_ = cfg;

    MDRuntimeOption opt;
    memset(&opt, 0, sizeof(opt));
    opt.cpu_thread_num = 4;
    opt.device = (cfg.device == "gpu") ? MD_DEVICE_GPU : MD_DEVICE_CPU;
    opt.backend = (cfg.backend == "trt") ? MD_BACKEND_TRT
                 : (cfg.backend == "mnn") ? MD_BACKEND_MNN
                 : MD_BACKEND_ORT;
    opt.enable_trt = 0;
    opt.ort_log_severity = 3;
    opt.trt_min_shape = "";
    opt.trt_opt_shape = "";
    opt.trt_max_shape = "";
    opt.trt_engine_cache_path = "";
    opt.password = "";

    MDStatusCode status = MDStatusCode::Success;

    if (cfg.type == "detection" || cfg.type == "face_detection") {
        status = md_create_detection_model(&model_, cfg.path.c_str(), &opt);
    } else if (cfg.type == "classification") {
        status = md_create_classification_model(&model_, cfg.path.c_str(), &opt);
    } else if (cfg.type == "lpr_pipeline") {
        status = md_create_detection_model(&model_, cfg.path.c_str(), &opt);
    } else {
        std::cerr << "[InferenceEngine] Unsupported model type: " << cfg.type << std::endl;
        return false;
    }

    if (status != MDStatusCode::Success) {
        std::cerr << "[InferenceEngine] Failed to load model: " << cfg.path
                  << " (error " << status << ")" << std::endl;
        return false;
    }

    loaded_ = true;
    std::cout << "[InferenceEngine] Loaded " << cfg.name << " [" << cfg.type
              << "] " << cfg.path << std::endl;
    return true;
}

void InferenceEngine::unload() {
    if (model_.model_content) {
        md_free_detection_model(&model_);
        model_.model_content = nullptr;
    }
    if (model_.model_name) {
        free(model_.model_name);
        model_.model_name = nullptr;
    }
    loaded_ = false;
}

bool InferenceEngine::infer(MDImage* image, InferResult* result) {
    if (!loaded_ || !image || !result) return false;
    result->model_name = cfg_.name;
    result->type = cfg_.type;

    if (cfg_.type == "detection" || cfg_.type == "face_detection") {
        return infer_detection(image, result);
    } else if (cfg_.type == "classification") {
        return infer_classification(image, result);
    }
    return false;
}

bool InferenceEngine::infer_detection(MDImage* image, InferResult* result) {
    MDDetectionResults c_results;
    memset(&c_results, 0, sizeof(c_results));

    auto st = md_detection_predict(&model_, image, &c_results);
    if (st != MDStatusCode::Success) {
        std::cerr << "[InferenceEngine] Detection predict failed: " << st << std::endl;
        return false;
    }

    for (int i = 0; i < c_results.size; ++i) {
        auto& d = c_results.data[i];
        DetectionBox box;
        box.x = static_cast<float>(d.box.x);
        box.y = static_cast<float>(d.box.y);
        box.w = static_cast<float>(d.box.width);
        box.h = static_cast<float>(d.box.height);
        box.score = d.score;
        box.label_id = d.label_id;
        result->boxes.push_back(box);
    }

    md_free_detection_result(&c_results);
    return true;
}

bool InferenceEngine::infer_classification(MDImage* image, InferResult* result) {
    MDClassificationResults c_results;
    memset(&c_results, 0, sizeof(c_results));

    auto st = md_classification_predict(&model_, image, &c_results);
    if (st != MDStatusCode::Success) return false;

    for (int i = 0; i < c_results.size; ++i) {
        DetectionBox box;
        box.score = c_results.data[i].score;
        box.label_id = c_results.data[i].label_id;
        result->boxes.push_back(box);
    }

    md_free_classification_result(&c_results);
    return true;
}
