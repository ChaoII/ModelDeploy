#include "infer_group.hpp"
#include <iostream>

using namespace modeldeploy::vision;

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
    Entry en;
    en.engine = std::move(engine);
    entries_.push_back(std::move(en));
    return true;
}

bool InferGroup::remove_model(const std::string& name) {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->engine->config().name == name) {
            entries_.erase(it);
            return true;
        }
    }
    return false;
}

void InferGroup::clear() {
    entries_.clear();
}

bool InferGroup::empty() const {
    return entries_.empty();
}

bool InferGroup::run_models(
    const ImageData& frame,
    std::vector<std::pair<std::string, std::vector<DetectionResult>>>* sdk_dets,
    std::vector<std::pair<std::string, InferResult>>* non_det) {
    if (sdk_dets) sdk_dets->clear();
    if (non_det) non_det->clear();
    bool any = false;
    for (auto& en : entries_) {
        auto* e = en.engine.get();
        const auto& mc = e->config();
        const int interval = mc.interval > 0 ? mc.interval : 1;
        // interval 抽帧：每 interval 帧推理一次，其余帧复用上次结果（无结果时强制先跑一次）
        const bool do_infer = !en.has_last || (en.frame_idx % interval == 0);
        const bool is_det = (mc.type == "detection" && e->det_model() != nullptr);

        if (do_infer) {
            if (is_det) {
                // ROI 区域裁剪推理：仅 CPU 帧（设备 NV12 不做裁剪）；结果按偏移回映射到全图
                const bool roi_valid = mc.roi.size() >= 4 && mc.roi[2] > 0 && mc.roi[3] > 0;
                ImageData infer_frame = frame;
                float ox = 0.f, oy = 0.f;
                if (roi_valid && frame.device() == modeldeploy::Device::CPU) {
                    infer_frame = frame.crop({static_cast<float>(mc.roi[0]), static_cast<float>(mc.roi[1]),
                                              static_cast<float>(mc.roi[2]), static_cast<float>(mc.roi[3])});
                    ox = static_cast<float>(mc.roi[0]);
                    oy = static_cast<float>(mc.roi[1]);
                }
                std::vector<DetectionResult> dets;
                if (e->det_model()->predict(infer_frame, &dets)) {
                    if (ox != 0.f || oy != 0.f)
                        for (auto& d : dets) { d.box.x += ox; d.box.y += oy; }
                    en.last_dets = std::move(dets);
                    en.has_last = true;
                }
            } else {
                InferResult r;
                if (e->infer(frame, &r)) {
                    en.last_non_det = std::move(r);
                    en.has_last = true;
                }
            }
        }
        ++en.frame_idx;

        if (is_det) {
            if (sdk_dets) sdk_dets->push_back({mc.name, en.last_dets});
        } else {
            if (non_det) non_det->push_back({mc.name, en.last_non_det});
        }
        if (en.has_last) any = true;
    }
    return any;
}

modeldeploy::vision::detection::UltralyticsDet* InferGroup::det_model(const std::string& name) {
    for (auto& en : entries_) {
        if (en.engine->config().name == name) return en.engine->det_model();
    }
    return nullptr;
}

const ModelConfig* InferGroup::config_of(const std::string& name) const {
    for (const auto& en : entries_) {
        if (en.engine->config().name == name) return &en.engine->config();
    }
    return nullptr;
}
