//
// Created by 84945 on 2025/11/5.
//

#include "vision/utils.h"
#include "core/md_log.h"
#include "vision/common/visualize/utils.h"
#include "vision/common/visualize/visualize.h"

namespace modeldeploy::vision {
    ImageData vis_keypoints(ImageData& image, const std::vector<KeyPointsResult>& result,
                            const std::string& font_path, const int font_size,
                            const int landmark_radius, const double alpha,
                            const bool save_result) {
        cv::Mat cv_image, overlay;
        image.to_mat(&cv_image);
        cv_image.copyTo(overlay);
        const cv::FontFace font(font_path);
        static std::map<int, cv::Scalar_<int>> color_map; // ← 每类颜色只初始化一次
        // 绘制半透明部分（填充矩形）
        for (const auto& _result : result) {
            auto class_id = _result.label_id;
            if (color_map.find(class_id) == color_map.end()) {
                color_map[class_id] = get_random_color();
            }
            auto cv_color = color_map[class_id];
            const std::string text = "score: " + std::to_string(_result.score).substr(0, 4);
            draw_rectangle_and_text(overlay, utils::rect2f_to_cv_type(_result.box), text, cv_color, font, font_size, -1,
                                    false);
        }
        cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
        // 绘制对象矩形矩形边框、文字背景边框、文字、关键点
        for (const auto& _result : result) {
            auto class_id = _result.label_id;
            auto cv_color = color_map[class_id];
            const std::string text = "score: " + std::to_string(_result.score).substr(0, 4);
            draw_rectangle_and_text(cv_image, utils::rect2f_to_cv_type(_result.box), text, cv_color,
                                    font, font_size, 1, true);
            std::vector<cv::Point3f> cv_keypoints;
            std::transform(_result.keypoints.begin(), _result.keypoints.end(),
                           std::back_inserter(cv_keypoints),
                           [](const Point3f& point) {
                               return utils::point3f_to_cv_type(point);
                           });
            draw_landmarks(cv_image, cv_keypoints, landmark_radius);
        }
        if (save_result) {
            MD_LOG_INFO << "Save pose result to [vis_result.jpg]" << std::endl;
            cv::imwrite("vis_result.jpg", cv_image);
        }
        return image;
    }
}
