//
// Created by aichao on 2026/08/22.
//

#include "vision/utils.h"
#include "core/md_log.h"
#include "vision/common/visualize/utils.h"
#include "vision/common/visualize/visualize.h"

namespace modeldeploy::vision {
    struct HandParams {
        static constexpr float kpt_threshold = 0.5;
        // MediaPipe 21 点骨架连接表（端点 0-indexed，WRIST=0）：
        //   拇指 1..4，食指 0,5..8，中指 5,9..12，无名指 9,13..16，小指 13,17..20，掌根 0-17
        inline static std::vector<std::pair<int, int>> skeleton = {
            {1, 2}, {2, 3}, {3, 4},                                    // 拇指
            {0, 5}, {5, 6}, {6, 7}, {7, 8},                            // 食指
            {5, 9}, {9, 10}, {10, 11}, {11, 12},                       // 中指
            {9, 13}, {13, 14}, {14, 15}, {15, 16},                     // 无名指
            {13, 17}, {17, 18}, {18, 19}, {19, 20},                    // 小指
            {0, 17}                                                    // 掌根
        };
        // 21 点手部调色板（BGR，每关键点一个颜色）
        inline static std::vector<cv::Scalar> palette = {
            cv::Scalar(0, 0, 255),
            cv::Scalar(0, 85, 255),
            cv::Scalar(0, 170, 255),
            cv::Scalar(0, 255, 255),
            cv::Scalar(0, 255, 170),
            cv::Scalar(0, 255, 85),
            cv::Scalar(0, 255, 0),
            cv::Scalar(85, 255, 0),
            cv::Scalar(170, 255, 0),
            cv::Scalar(255, 255, 0),
            cv::Scalar(255, 170, 0),
            cv::Scalar(255, 85, 0),
            cv::Scalar(255, 0, 0),
            cv::Scalar(255, 0, 85),
            cv::Scalar(255, 0, 170),
            cv::Scalar(255, 0, 255),
            cv::Scalar(170, 0, 255),
            cv::Scalar(85, 0, 255),
            cv::Scalar(0, 0, 255),
            cv::Scalar(0, 0, 170),
            cv::Scalar(0, 255, 0),
        };
    };

    void draw_hand_keypoints(cv::Mat& cv_image,
                             const std::vector<cv::Point3f>& keypoints,
                             const int keypoint_radius) {
        for (int j = 0; j < keypoints.size(); ++j) {
            const cv::Point3f keypoint = keypoints[j];
            if (keypoint.z < HandParams::kpt_threshold)
                continue;
            const cv::Scalar& kpt_color = HandParams::palette[j % HandParams::palette.size()];
            cv::circle(cv_image, cv::Point2f(keypoint.x, keypoint.y), keypoint_radius, kpt_color, -1, 8);
        }

        for (int j = 0; j < HandParams::skeleton.size(); ++j) {
            const auto i0 = HandParams::skeleton[j].first;
            const auto i1 = HandParams::skeleton[j].second;
            if (i0 >= keypoints.size() || i1 >= keypoints.size())
                continue;
            const auto kpt0 = keypoints[i0];
            const auto kpt1 = keypoints[i1];
            if (kpt0.z < HandParams::kpt_threshold || kpt1.z < HandParams::kpt_threshold)
                continue;
            const cv::Scalar& kpt_color = HandParams::palette[i0 % HandParams::palette.size()];
            cv::line(cv_image, cv::Point2f(kpt0.x, kpt0.y),
                     cv::Point2f(kpt1.x, kpt1.y),
                     kpt_color,
                     1,
                     cv::LINE_AA);
        }
    }

    ImageData vis_hand(ImageData& image, const std::vector<KeyPointsResult>& result,
                       const std::string& font_path, const int font_size,
                       const int landmark_radius, const double alpha,
                       const bool save_result) {
        cv::Mat cv_image, overlay;
        image.asMat(&cv_image);
        cv_image.copyTo(overlay);
        cv::FontFace& font = get_font_face(font_path);
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
        // 绘制对象矩形边框、文字背景边框、文字、关键点
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
            draw_hand_keypoints(cv_image, cv_keypoints, landmark_radius);
        }
        if (save_result) {
            MD_LOG_INFO << "Save hand result to [vis_result.jpg]" << std::endl;
            cv::imwrite("vis_result.jpg", cv_image);
        }
        return image;
    }
} // namespace modeldeploy::vision
