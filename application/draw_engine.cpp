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

namespace {
    // 与 vis_det 一致的 label_id → BGR 颜色（确定性调色板，GPU 内核使用）
    // 按 GpuDrawBox 字段顺序 r,g,b 写入：rgb[0]=R, rgb[1]=G, rgb[2]=B
    void color_for_label(int label_id, uint8_t* rgb) {
        static const uint8_t palette[8][3] = {
            {0, 0, 255},       // red
            {0, 255, 0},       // green
            {255, 0, 0},       // blue
            {0, 255, 255},     // yellow
            {255, 0, 255},     // magenta
            {255, 255, 0},     // cyan
            {0, 165, 255},     // orange
            {255, 128, 0},     // violet
        };
        const auto* c = palette[static_cast<unsigned>(label_id % 8)];
        rgb[0] = c[2];   // r
        rgb[1] = c[1];   // g
        rgb[2] = c[0];   // b
    }

    std::string format_label(const DetectionBox& b, bool show_label, bool show_score) {
        std::string label;
        if (show_label) {
            label = b.label_name.empty() ? std::to_string(b.label_id) : b.label_name;
        }
        if (show_score) {
            if (!label.empty()) label += ": ";
            label += std::to_string(b.score).substr(0, 4);
        }
        if (label.size() > 31) label.resize(31);
        return label;
    }
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
    uint8_t rgb[3];
    const float threshold = show_score ? 0.0f : 0.5f;   // 与 draw_detection 阈值一致
    for (const auto& r : results) {
        if (r.type == "detection") {
            for (const auto& b : r.boxes) {
                if (b.score < threshold) continue;
                color_for_label(b.label_id, rgb);
                backend->draw_rect_nv12(image, b.x, b.y, b.w, b.h,
                                        rgb[0], rgb[1], rgb[2], 2);
                if (show_label) {
                    const std::string label = format_label(b, show_label, show_score);
                    backend->draw_text_nv12(image, b.x, std::max(0.0f, b.y - 16), label,
                                            255, 255, 255, 1);
                }
                any = true;
            }
        }
    }
    return any || results.empty();
}
