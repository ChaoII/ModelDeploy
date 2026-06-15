#include "draw_engine.hpp"
#include "csrc/vision/common/visualize/visualize.h"
#include <iostream>

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
