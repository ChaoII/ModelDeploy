//
// Created by aichao on 2025/4/3.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/common/visualize/visualize.h"

namespace modeldeploy::vision {
    void draw_rectangle_and_text(cv::Mat& image, const std::array<float, 4> box, const std::string& text,
                                 const cv::Scalar& color, cv::FontFace font, const int font_size,
                                 const int thickness, const bool draw_text = false) {
        const auto x1 = static_cast<int>(box[0]);
        const auto y1 = static_cast<int>(box[1]);
        const auto x2 = static_cast<int>(box[2]);
        const auto y2 = static_cast<int>(box[3]);
        // 绘制对象矩形框
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness);
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

    void draw_landmarks(cv::Mat& cv_image,
                        const std::vector<std::array<float, 2>>& landmarks,
                        const size_t
                        current_obj_index,
                        const size_t landmark_num,
                        const int landmark_radius) {
        for (size_t j = 0; j < landmark_num; ++j) {
            cv::Point landmark;
            auto landmark_color = get_random_color();
            landmark.x = static_cast<int>(
                landmarks[current_obj_index * landmark_num + j][0]);
            landmark.y = static_cast<int>(
                landmarks[current_obj_index * landmark_num + j][1]);
            cv::circle(cv_image, landmark, landmark_radius, landmark_color, -1);
        }
    }


    cv::Mat vis_lpr(cv::Mat& cv_image, const LprResult& result,
                    const std::string& font_path, const int font_size,
                    const int landmark_radius, const double alpha, const bool save_result) {
        cv::Mat overlay;
        cv_image.copyTo(overlay);
        const cv::FontFace font(font_path);
        const cv::Scalar cv_color = get_random_color();
        // 绘制半透明部分（填充矩形）
        for (int i = 0; i < result.boxes.size(); ++i) {
            // const std::string text = plate_str + " " + plate_color + " " + std::to_string(score).substr(0, 4);
            const std::string text = result.car_plate_strs[i] + " " + result.car_plate_colors[i] + " " +
                std::to_string(result.scores[i]).substr(0, 4);
            draw_rectangle_and_text(overlay, result.boxes[i], text, cv_color, font, font_size, -1, false);
        }
        cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
        // 绘制对象矩形矩形边框、文字背景边框、文字、关键点
        for (int i = 0; i < result.boxes.size(); ++i) {
            const std::string text = result.car_plate_strs[i] + " " + result.car_plate_colors[i] + " " +
                std::to_string(result.scores[i]).substr(0, 4);
            draw_rectangle_and_text(cv_image, result.boxes[i], text, cv_color, font, font_size, 1, true);
            // draw landmark;
            draw_landmarks(cv_image, result.landmarks, i, 4, landmark_radius);
        }
        if (save_result) {
            MD_LOG_INFO << "Save lpr result to [vis_result.jpg]" << std::endl;
            cv::imwrite("vis_result.jpg", cv_image);
        }
        return cv_image;
    }


    cv::Mat vis_det_landmarks(cv::Mat cv_image, const DetectionLandmarkResult& result,
                              const std::string& font_path, const int font_size,
                              const int landmark_radius, const double alpha, const bool save_result) {
        cv::Mat overlay;
        cv_image.copyTo(overlay);
        const cv::FontFace font(font_path);
        const cv::Scalar cv_color = get_random_color();
        // 绘制半透明部分（填充矩形）
        for (int i = 0; i < result.boxes.size(); ++i) {
            const std::string text = "score: " + std::to_string(result.scores[i]).substr(0, 4);
            draw_rectangle_and_text(overlay, result.boxes[i], text, cv_color, font, font_size, -1, false);
        }
        cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
        // 绘制对象矩形矩形边框、文字背景边框、文字、关键点
        for (int i = 0; i < result.boxes.size(); ++i) {
            const std::string text = "score: " + std::to_string(result.scores[i]).substr(0, 4);
            draw_rectangle_and_text(cv_image, result.boxes[i], text, cv_color, font, font_size, 1, true);
            draw_landmarks(cv_image, result.landmarks, i, result.landmarks_per_instance, landmark_radius);
        }
        if (save_result) {
            MD_LOG_INFO << "Save detection result to [vis_result.jpg]" << std::endl;
            cv::imwrite("vis_result.jpg", cv_image);
        }
        return cv_image;
    }
}
