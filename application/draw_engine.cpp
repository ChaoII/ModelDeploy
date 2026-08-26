#include "draw_engine.hpp"
#include "csrc/vision/common/visualize/visualize.h"
#include "csrc/vision/processors/processor_factory.h"
#ifdef WITH_GPU
#include "csrc/vision/processors/cuda/cuda_processor_backend.h"
#endif
#ifdef ENABLE_SOPHGO
#include "csrc/vision/processors/sophgo/sophgo_processor_backend.h"
#endif
#include <iostream>
#include <cstdio>
#include <algorithm>

using namespace modeldeploy::vision;

DrawEngine::DrawEngine(const DrawConfig& cfg) : cfg_(cfg) {
}

void DrawEngine::draw(ImageData& image,
                       const std::vector<InferResult>& results) {
    for (const auto& r : results) {
        if (r.type == "detection") {
            draw_detection(image, r);
        } else if (r.type == "face_detection") {
            draw_face(image, r);
        }
    }
}

void DrawEngine::draw_detection(ImageData& image, const InferResult& result) {
    std::vector<DetectionResult> det_results;
    det_results.reserve(result.boxes.size());
    for (const auto& b : result.boxes) {
        DetectionResult dr;
        dr.box = {b.x, b.y, b.w, b.h};
        dr.score = b.score;
        dr.label_id = b.label_id;
        det_results.push_back(dr);
    }
    vis_det(image, det_results,
            cfg_.show_score ? 0.0 : 0.5,
            {}, cfg_.font_path, 12, 0.15, false);
}

void DrawEngine::draw_face(ImageData& image, const InferResult& result) {
    std::vector<KeyPointsResult> kp_results;
    kp_results.reserve(result.boxes.size());
    for (size_t i = 0; i < result.boxes.size(); ++i) {
        const auto& b = result.boxes[i];
        KeyPointsResult kp;
        kp.box = {b.x, b.y, b.w, b.h};
        kp.score = b.score;
        kp.label_id = b.label_id;
        if (i < result.keypoints.size()) {
            for (const auto& p : result.keypoints[i]) {
                kp.keypoints.emplace_back(p.x, p.y, 0.0f);
            }
        }
        kp_results.push_back(std::move(kp));
    }
    vis_keypoints(image, kp_results, cfg_.font_path, 12, 3, 0.15, false, false);
}

bool DrawEngine::draw_gpu(ImageData& image,
                          const std::vector<InferResult>& results,
                          bool show_label, bool show_score) {
    if (image.empty()) return true;
    // 统一设备/CPU NV12 就地绘制（零拷贝，不 D2H/H2D）：
    // 按 frame.device() 分派到对应 processor backend 的 draw_*_nv12，直接写设备 y/uv 平面。
    // 仅对 NV12 帧可用（CPU packed / 非 NV12 无 NV12 内核）→ 返回 false，由调用方回退 CPU 绘制。
    if (image.type() != MdImageType::NV12 || image.plane_count() < 2) return false;

    const auto device = image.device();
    // 复用 backend：避免每帧 create_processor_backend（建流/分配池为 CUDA 同步操作，
    // 高频调用与 batch_predict 争用、拖慢整条批推理链路）
    VisionProcessorBackend* backend = nullptr;
    {
        std::lock_guard<std::mutex> lock(backend_mtx_);
        auto it = backends_.find(device);
        if (it == backends_.end()) {
            auto b = create_processor_backend(
                device, device == modeldeploy::Device::TPU ? modeldeploy::Backend::SOPHGO
                                                           : modeldeploy::Backend::ORT, 0);
            if (!b) return false;
            backend = b.get();
            backends_.emplace(device, std::move(b));
        } else {
            backend = it->second.get();
        }
    }

    // 设备帧必须确认真实设备后端，杜绝工厂回退 CPU 后在设备内存上跑 CPU 内核（越界/UB）。
    if (device != modeldeploy::Device::CPU) {
        bool device_ready = false;
#ifdef WITH_GPU
        if (device == modeldeploy::Device::GPU &&
            dynamic_cast<CudaProcessorBackend*>(backend) != nullptr) device_ready = true;
#endif
#ifdef ENABLE_SOPHGO
        if (device == modeldeploy::Device::TPU &&
            dynamic_cast<SophgoProcessorBackend*>(backend) != nullptr) device_ready = true;
#endif
        if (!device_ready) return false;
    }

    bool any = false;
    // 高层设备绘制:与 CPU vis_* 语义一致(整框填充 + 类色 + 标签 + 阈值)
    VisionProcessorBackend::VisOptions vo;
    vo.threshold = show_score ? 0.0 : 0.5;
    vo.alpha = 0.15;
    for (const auto& r : results) {
        if (r.type == "detection") {
            std::vector<DetectionResult> dets;
            dets.reserve(r.boxes.size());
            for (const auto& b : r.boxes) {
                if (b.score < vo.threshold) continue;
                DetectionResult dr;
                dr.box = {b.x, b.y, b.w, b.h};
                dr.score = b.score;
                dr.label_id = b.label_id;
                if (!b.label_name.empty()) vo.label_map[b.label_id] = b.label_name;
                dets.push_back(std::move(dr));
            }
            if (!dets.empty() && backend->vis_det_nv12(image, dets, vo)) any = true;
        } else if (r.type == "face_detection") {
            std::vector<KeyPointsResult> kps;
            kps.reserve(r.boxes.size());
            for (size_t i = 0; i < r.boxes.size(); ++i) {
                const auto& b = r.boxes[i];
                if (b.score < vo.threshold) continue;
                KeyPointsResult kp;
                kp.box = {b.x, b.y, b.w, b.h};
                kp.score = b.score;
                kp.label_id = b.label_id;
                if (i < r.keypoints.size()) {
                    for (const auto& p : r.keypoints[i]) kp.keypoints.emplace_back(p.x, p.y, 0.0f);
                }
                kps.push_back(std::move(kp));
            }
            if (!kps.empty() && backend->vis_keypoints_nv12(image, kps, vo, false)) any = true;
        }
    }
    return any || results.empty();
}
