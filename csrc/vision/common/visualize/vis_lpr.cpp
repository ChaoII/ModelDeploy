//
// Created by 84945 on 2025/6/2.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/common/visualize/visualize.h"

namespace modeldeploy::vision
{
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
            std::vector current_landmarks(result.landmarks.begin() + i * 4,
                                          result.landmarks.begin() + (i + 1) * 4);
            draw_landmarks(cv_image, current_landmarks, landmark_radius);
        }
        if (save_result) {
            MD_LOG_INFO << "Save lpr result to [vis_result.jpg]" << std::endl;
            cv::imwrite("vis_result.jpg", cv_image);
        }
        return cv_image;
    }
}
