#include "draw_engine.hpp"
#include "csrc/vision/common/visualize/visualize.h"
#include <iostream>

using namespace modeldeploy::vision;

DrawEngine::DrawEngine(const DrawConfig& cfg) : cfg_(cfg) {
}

void DrawEngine::draw(ImageData& image,
                       const std::vector<InferResult>& results) {
    for (const auto& r : results) {
        if (r.type == "detection" || r.type == "face_detection") {
            draw_detection(image, r);
        }
    }
}

void DrawEngine::draw_detection(ImageData& image, const InferResult& result) {
    std::vector<DetectionResult> det_results;
    for (const auto& b : result.boxes) {
        DetectionResult dr;
        dr.box = {b.x, b.y, b.w, b.h};
        dr.score = b.score;
        dr.label_id = b.label_id;
        det_results.push_back(dr);
    }

    // 使用 ModelDeploy 的 vis_det 绘制
    vis_det(image, det_results,
            cfg_.show_score ? 0.0 : 0.5,
            {}, cfg_.font_path, 12, 0.15, false);
}
