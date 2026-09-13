#include "infer_group.hpp"
#include <iostream>

using namespace modeldeploy::vision;

namespace {
// 计算有效像素 ROI：优先 roi_norm（归一化，按帧宽高换算），否则用像素 roi
struct RoiRect { float x, y, w, h; bool valid; };
RoiRect effective_roi(const ModelConfig& mc, const ImageData& frame) {
    if (mc.roi_norm.size() >= 4 && mc.roi_norm[2] > 0.f && mc.roi_norm[3] > 0.f &&
        frame.width() > 0 && frame.height() > 0) {
        return {mc.roi_norm[0] * frame.width(), mc.roi_norm[1] * frame.height(),
                mc.roi_norm[2] * frame.width(), mc.roi_norm[3] * frame.height(), true};
    }
    if (mc.roi.size() >= 4 && mc.roi[2] > 0 && mc.roi[3] > 0)
        return {static_cast<float>(mc.roi[0]), static_cast<float>(mc.roi[1]),
                static_cast<float>(mc.roi[2]), static_cast<float>(mc.roi[3]), true};
    return {0, 0, 0, 0, false};
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
                const RoiRect rr = effective_roi(mc, frame);
                ImageData infer_frame = frame;
                float ox = 0.f, oy = 0.f;
                if (rr.valid && frame.device() == modeldeploy::Device::CPU) {
                    infer_frame = frame.crop({rr.x, rr.y, rr.w, rr.h});
                    ox = rr.x; oy = rr.y;
                }
                std::vector<DetectionResult> dets;
                if (e->predict_detection(infer_frame, &dets)) {
                    if (ox != 0.f || oy != 0.f)
                        for (auto& d : dets) { d.box.x += ox; d.box.y += oy; }
                    en.last_dets = std::move(dets);
                    en.has_last = true;
                }
            } else {
                // ROI：非 detection 族（如人脸）按区域裁剪，几何结果按偏移回映射；classification 无几何不做
                const RoiRect rr = effective_roi(mc, frame);
                const bool can_roi = rr.valid && frame.device() == modeldeploy::Device::CPU &&
                                     mc.type != "classification";
                ImageData infer_frame = frame;
                float ox = 0.f, oy = 0.f;
                if (can_roi) {
                    infer_frame = frame.crop({rr.x, rr.y, rr.w, rr.h});
                    ox = rr.x; oy = rr.y;
                }
                InferResult r;
                if (e->infer(infer_frame, &r)) {
                    if (ox != 0.f || oy != 0.f) {
                        for (auto& b : r.boxes) { b.x += ox; b.y += oy; }
                        for (auto& kps : r.keypoints)
                            for (auto& k : kps) { k.x += ox; k.y += oy; }
                    }
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
