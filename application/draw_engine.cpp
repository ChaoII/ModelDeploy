#include "draw_engine.hpp"
#include "csrc/vision/common/visualize/visualize.h"
#include "csrc/vision/processors/cuda/draw_gpu.cuh"
#include <iostream>
#include <cstdio>

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
        kp.type = ResultType::FACE_DETECTION;
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
    const int width = image.width();
    const int height = image.height();

    for (const auto& r : results) {
        // face_detection 关键点尚无 GPU 绘制实现：任一此类结果回退 CPU 路径（vis_keypoints）
        if (r.type == "face_detection") return false;
    }

    std::vector<GpuDrawBox> boxes;
    for (const auto& r : results) {
        if (r.type != "detection") continue;
        for (const auto& b : r.boxes) {
            GpuDrawBox gb{};
            gb.x1 = static_cast<int>(b.x);
            gb.y1 = static_cast<int>(b.y);
            gb.x2 = static_cast<int>(b.x + b.w);
            gb.y2 = static_cast<int>(b.y + b.h);
            gb.score = b.score;
            gb.label_id = b.label_id;
            color_for_label(b.label_id, &gb.r);   // 按 r,g,b 字段顺序写入
            const std::string label = format_label(b, show_label, show_score);
            std::snprintf(gb.label, sizeof(gb.label), "%s", label.c_str());
            boxes.push_back(gb);
        }
    }
    if (boxes.empty()) return true;

    // bgr/boxes 均为 host 指针 → draw_boxes_gpu 内部自动上传、绘制、回拷
    return draw_boxes_gpu(image.data(), width, height, boxes.data(),
                          static_cast<int>(boxes.size()), 0.15f, nullptr);
}
