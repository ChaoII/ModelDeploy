#include "draw_engine.hpp"
#include <opencv2/imgproc.hpp>
#include <random>
#include <iostream>

DrawEngine::DrawEngine(const DrawConfig& cfg) : cfg_(cfg) {
    // 预生成颜色表
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(50, 255);
    colors_.reserve(80);
    for (int i = 0; i < 80; ++i) {
        colors_.emplace_back(dist(rng), dist(rng), dist(rng));
    }
}

cv::Scalar DrawEngine::get_color(int label_id) {
    static std::vector<cv::Scalar> table;
    if (table.empty()) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(50, 255);
        for (int i = 0; i < 80; ++i)
            table.emplace_back(dist(rng), dist(rng), dist(rng));
    }
    return table[static_cast<size_t>(label_id) % table.size()];
}

void DrawEngine::draw(cv::Mat& bgr_image,
                       const std::vector<InferResult>& results) {
    for (const auto& r : results) {
        draw_detection(bgr_image, r);
    }
}

void DrawEngine::draw_detection(cv::Mat& image, const InferResult& result) {
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.5;
    int thickness = 2;

    for (const auto& box : result.boxes) {
        auto color = get_color(box.label_id);
        cv::Rect rect(cv::Point(static_cast<int>(box.x), static_cast<int>(box.y)),
                       cv::Point(static_cast<int>(box.x + box.w),
                                 static_cast<int>(box.y + box.h)));
        cv::rectangle(image, rect, color, thickness);

        if (cfg_.show_label || cfg_.show_score) {
            std::string label;
            if (cfg_.show_label)
                label = box.label_name.empty()
                    ? std::to_string(box.label_id)
                    : box.label_name;
            if (cfg_.show_score)
                label += " " + std::to_string(static_cast<int>(box.score * 100)) + "%";

            int baseline;
            auto text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);
            cv::rectangle(image,
                          cv::Point(rect.x, rect.y - text_size.height - 5),
                          cv::Point(rect.x + text_size.width, rect.y),
                          color, -1);
            cv::putText(image, label,
                        cv::Point(rect.x, rect.y - 5),
                        font_face, font_scale,
                        cv::Scalar(255, 255, 255), thickness);
        }
    }
}
