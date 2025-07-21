//
// Created by aichao on 2025/4/1.
//

#include "vision/utils.h"
#include "core/md_log.h"
#include "vision/common/visualize/utils.h"
#include "vision/common/visualize/visualize.h"


namespace modeldeploy::vision {
    ImageData vis_obb(ImageData& image, const std::vector<ObbResult>& result,
                      const double threshold,
                      const std::string& font_path, const int font_size,
                      const double alpha, const bool save_result) {
        cv::Mat cv_image, overlay;
        image.to_mat(&cv_image);
        cv_image.copyTo(overlay);
        cv::FontFace font(font_path);
        // 根据label_id获取颜色
        static std::map<int, cv::Scalar_<int>> color_map;
        // 绘制半透明部分（填充矩形）

        // for roted_box
        for (int i = 0; i < result.size(); ++i) {
            if (result[i].score < threshold) {
                continue;
            }
            auto class_id = result[i].label_id;
            if (color_map.find(class_id) == color_map.end()) {
                color_map[class_id] = get_random_color();
            }
            cv::Point2f _points[4];
            utils::rotated_rect_to_cv_type(result[i].rotated_box).points(_points);
            std::vector<cv::Point> points;
            points.reserve(4);
            for (auto& _point : _points) {
                points.emplace_back(cvRound(_point.x), cvRound(_point.y));
            }
            auto cv_color = color_map[class_id];
            // 绘制对象矩形框
            cv::fillPoly(overlay, points, cv_color, cv::LINE_AA, 0);
            std::string text = std::to_string(class_id) + ": " + std::to_string(result[i].score).substr(0, 4);
            const auto size = cv::getTextSize(cv::Size(0, 0), text, points[1], font, font_size);
            // 绘制标签背景
            cv::rectangle(overlay, size, cv_color, -1);
        }


        cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
        // 绘制对象矩形矩形边框、文字背景边框、文字
        // for roted_box
        for (int c = 0; c < result.size(); ++c) {
            if (result[c].score < threshold) {
                continue;
            }
            auto class_id = result[c].label_id;
            auto roted_boxes = result[c].rotated_box;
            cv::Point2f _points[4];
            utils::rotated_rect_to_cv_type(roted_boxes).points(_points);
            auto cv_color = color_map[class_id];
            std::vector<cv::Point> points;
            points.reserve(4);
            for (auto& _point : _points) {
                points.emplace_back(cvRound(_point.x), cvRound(_point.y));
            }
            cv::polylines(cv_image, points, true, cv_color, 1, cv::LINE_AA, 0);
            std::string text = std::to_string(class_id) + ": " + std::to_string(result[c].score).substr(0, 4);
            const auto size = cv::getTextSize(cv::Size(0, 0), text, points[1], font, font_size);
            cv::rectangle(cv_image, size, cv_color, 1, cv::LINE_AA, 0);
            cv::putText(cv_image, text, points[1],
                        cv::Scalar(255 - cv_color[0], 255 - cv_color[1], 255 - cv_color[2]), font, font_size);
        }


        if (save_result) {
            MD_LOG_INFO << "Save detection result to [vis_result.jpg]" << std::endl;
            cv::imwrite("vis_result.jpg", cv_image);
        }
        return image;
    }
} // namespace modeldeploy::vision
