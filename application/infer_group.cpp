#include "infer_group.hpp"
#include <iostream>

using namespace modeldeploy::vision;

namespace {
modeldeploy::vision::detection::UltralyticsDet* find_det_model(
    std::vector<std::unique_ptr<InferenceEngine>>& engines, const std::string& name) {
    for (auto& e : engines) {
        if (e->config().name == name) return e->det_model();
    }
    return nullptr;
}
}

bool InferGroup::load_models(const std::vector<ModelConfig>& mcfgs, ModelFactory factory) {
    clear();
    bool all_ok = true;
    for (const auto& mcfg : mcfgs) {
        if (!add_model(mcfg, factory)) {
            std::cerr << "[InferGroup] Failed to load model: " << mcfg.name << std::endl;
            all_ok = false;
        }
    }
    return all_ok;
}

bool InferGroup::add_model(const ModelConfig& mcfg, ModelFactory factory) {
    std::unique_ptr<InferenceEngine> engine;
    if (factory) {
        engine = factory(mcfg);
        if (engine && !engine->is_loaded()) engine.reset();
    }
    if (!engine) {
        engine = std::make_unique<InferenceEngine>();
        if (!engine->load(mcfg)) return false;
    }
    engines_.push_back(std::move(engine));
    return true;
}

bool InferGroup::remove_model(const std::string& name) {
    for (auto it = engines_.begin(); it != engines_.end(); ++it) {
        if ((*it)->config().name == name) {
            engines_.erase(it);
            return true;
        }
    }
    return false;
}

void InferGroup::clear() {
    engines_.clear();
}

bool InferGroup::empty() const {
    return engines_.empty();
}

bool InferGroup::run_models(
    const ImageData& frame,
    std::vector<std::pair<std::string, std::vector<DetectionResult>>>* sdk_dets) {
    if (sdk_dets) sdk_dets->clear();
    bool any = false;
    for (auto& e : engines_) {
        const auto& mc = e->config();
        if (mc.type == "detection" && e->det_model()) {
            std::vector<DetectionResult> dets;
            if (!e->det_model()->predict(frame, &dets)) continue;
            if (sdk_dets) sdk_dets->push_back({mc.name, std::move(dets)});
            any = true;
        } else {
            InferResult r;
            if (!e->infer(frame, &r)) continue;
            any = true;
        }
    }
    return any;
}

modeldeploy::vision::detection::UltralyticsDet* InferGroup::det_model(const std::string& name) {
    return find_det_model(engines_, name);
}
