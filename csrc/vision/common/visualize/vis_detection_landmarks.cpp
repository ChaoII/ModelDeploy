//
// Created by aichao on 2025/4/3.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/common/visualize/visualize.h"

namespace modeldeploy::vision {
    cv::Mat vis_det_landmarks(cv::Mat cv_image, const std::vector<DetectionLandmarkResult>& result,
                              const std::string& font_path, const int font_size,
                              const int landmark_radius, const double alpha, const bool save_result) {
        cv::Mat overlay;
        cv_image.copyTo(overlay);
        const cv::FontFace font(font_path);
        const cv::Scalar cv_color = get_random_color();
        // 绘制半透明部分（填充矩形）
        for (int i = 0; i < result.size(); ++i) {
            const std::string text = "score: " + std::to_string(result[i].score).substr(0, 4);
            draw_rectangle_and_text(overlay, result[i].box, text, cv_color, font, font_size, -1, false);
        }
        cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
        // 绘制对象矩形矩形边框、文字背景边框、文字、关键点
        for (int i = 0; i < result.size(); ++i) {
            const std::string text = "score: " + std::to_string(result[i].score).substr(0, 4);
            draw_rectangle_and_text(cv_image, result[i].box, text, cv_color, font, font_size, 1, true);
            draw_landmarks(cv_image, result[i].landmarks, landmark_radius);
        }
        if (save_result) {
            MD_LOG_INFO << "Save landmark result to [vis_result.jpg]" << std::endl;
            cv::imwrite("vis_result.jpg", cv_image);
        }
        return cv_image;
    }
}
