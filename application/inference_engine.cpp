#include "inference_engine.hpp"
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

    RuntimeOption opt;
    if (cfg.device == "gpu") {
        opt.use_gpu(0);
    }
    opt.set_cpu_thread_num(4);

    // ── TRT 后端（纯 TRT，非 ORT-TRT） ──
    if (cfg.backend == "trt" || cfg.backend == "tensorrt") {
        opt.use_trt_backend();
        opt.enable_fp16 = true;                    // RuntimeOption 级 FP16
        std::string cache_dir = "data/trt_cache";
        try { std::filesystem::create_directories(cache_dir); } catch (...) {}
        std::string model_name = cfg.path.substr(cfg.path.find_last_of("/\\") + 1);
        opt.trt_option.cache_file_path = cache_dir + "/" + model_name + ".engine";
        opt.trt_option.enable_fp16 = true;
        opt.trt_option.max_workspace_size = 1ULL << 30; // 1GB
        if (cfg_.input_size.size() == 2) {
            std::string min_s = "1x3x" + std::to_string(cfg_.input_size[0]) + "x" + std::to_string(cfg_.input_size[1]);
            opt.set_trt_min_shape(min_s);
            opt.set_trt_opt_shape(min_s);
            opt.set_trt_max_shape(min_s);
        }
    }
    // ── MNN 后端 ──
    else if (cfg.backend == "mnn") {
        opt.use_mnn_backend();
    }
    // ── ORT 后端（默认）—— 开启 TensorRT EP + FP16 + 缓存 ──
    else {
        opt.use_ort_backend();
        if (cfg.device == "gpu") {
            opt.enable_fp16 = true;                    // FP16 推理（关键性能优化）
            opt.ort_option.enable_trt = true;           // ORT 的 TensorRT EP
            opt.ort_option.enable_fp16 = true;          // FP16 for TRT EP
            std::string cache_dir = "data/ort_trt_cache";
            try { std::filesystem::create_directories(cache_dir); } catch (...) {}
            opt.ort_option.trt_engine_cache_path = cache_dir;
            // 设置动态 shape（TRT 编译用）
            if (cfg_.input_size.size() == 2) {
                int w = cfg_.input_size[0];
                int h = cfg_.input_size[1];
                std::string shape = "1x3x" + std::to_string(h) + "x" + std::to_string(w);
                opt.ort_option.trt_min_shape = shape;
                opt.ort_option.trt_opt_shape = shape;
                opt.ort_option.trt_max_shape = shape;
            }
        }
    }

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
            if (cfg.device == "gpu")
                det_model_->get_preprocessor().use_cuda_preproc();
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
    if (cfg.device == "gpu")
        det_model_->get_preprocessor().use_cuda_preproc();

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
    loaded_ = true;
    std::cout << "[InferenceEngine] Adopted face: " << cfg.name
              << " (shared ORT session)" << std::endl;
}

void InferenceEngine::unload() {
    det_model_.reset();
    cls_model_.reset();
    face_model_.reset();
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
    if (!det_model_) return false;
    std::vector<DetectionResult> det_results;
    if (!det_model_->predict(image, &det_results)) return false;
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
