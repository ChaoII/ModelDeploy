//
// Created by aichao on 2025/4/1.
//
#include "csrc/core/md_log.h"
#include "csrc/vision/common/visualize/visualize.h"

namespace modeldeploy::vision {
    void draw_rectangle_and_text(cv::Mat& image, const std::array<float, 4> box, const float score,
                                 const cv::Scalar& color, cv::FontFace font, const int font_size,
                                 const int thickness, const bool draw_text = false) {
        const auto x1 = static_cast<int>(box[0]);
        const auto y1 = static_cast<int>(box[1]);
        const auto x2 = static_cast<int>(box[2]);
        const auto y2 = static_cast<int>(box[3]);
        // 绘制对象矩形框
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness);
        const std::string text = "score: " + std::to_string(score).substr(0, 4);
        const auto size = cv::getTextSize(cv::Size(0, 0),
                                          text, cv::Point(x1, y1), font, font_size);
        // 绘制标签背景
        cv::rectangle(image, size, color, thickness);
        if (draw_text) {
            cv::putText(image, text, cv::Point(x1, y1 - 2),
                        cv::Scalar(255 - color[0], 255 - color[1], 255 - color[2]),
                        font, font_size);
        }
    }


    cv::Mat VisFaceDetection(cv::Mat cv_image, const DetectionLandmarkResult& result,
                             const std::string& font_path, const int font_size,
                             const int landmark_radius, const double alpha, const bool save_result) {
        cv::Mat overlay;
        cv_image.copyTo(overlay);
        const cv::FontFace font(font_path);
        const cv::Scalar cv_color = get_random_color();
        // 绘制半透明部分（填充矩形）
        for (int i = 0; i < result.boxes.size(); ++i) {
            draw_rectangle_and_text(overlay, result.boxes[i], result.scores[i], cv_color, font, font_size, -1, false);
        }
        cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
        // 绘制对象矩形矩形边框、文字背景边框、文字、关键点
        for (int i = 0; i < result.boxes.size(); ++i) {
            draw_rectangle_and_text(cv_image, result.boxes[i], result.scores[i], cv_color, font, font_size, 1, true);
            for (size_t j = 0; j < result.landmarks_per_instance; ++j) {
                cv::Point landmark;
                auto landmark_color = get_random_color();
                landmark.x = static_cast<int>(
                    result.landmarks[i * result.landmarks_per_instance + j][0]);
                landmark.y = static_cast<int>(
                    result.landmarks[i * result.landmarks_per_instance + j][1]);
                cv::circle(cv_image, landmark, landmark_radius, landmark_color, -1);
            }
        }
        if (save_result) {
            MD_LOG_INFO << "Save detection result to [vis_result.jpg]" << std::endl;
            cv::imwrite("vis_result.jpg", cv_image);
        }
        return cv_image;
    }
}
