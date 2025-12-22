//
// Created by aichao on 2025/4/1.
//

#include "core/md_log.h"
#include "vision/common/visualize/utils.h"
#include "vision/common/visualize/visualize.h"

namespace modeldeploy::vision {
    ImageData vis_attr(ImageData& image, const std::vector<AttributeResult>& result,
                       const double threshold,
                       const std::unordered_map<int, std::string>& label_map,
                       const std::string& font_path, const int font_size,
                       const double alpha, const bool save_result, const std::vector<int>& abnormal_ids,
                       const bool show_attr) {
        cv::Mat cv_image, overlay;
        image.to_mat(&cv_image);
        cv_image.copyTo(overlay);
        cv::FontFace font(font_path);
        if (abnormal_ids.size() > result.size()) {
            MD_LOG_WARN << "abnormal_ids size is larger than result size" << std::endl;
        }
        // 根据label_id获取颜色
        static std::map<int, cv::Scalar_<int>> color_map; // ← 每类颜色只初始化一次
        static auto obj_color = cv::Scalar(0, 255, 0);
        // 绘制半透明部分（填充矩形）
        for (int i = 0; i < result.size(); ++i) {
            if (result[i].box_score < threshold) {
                continue;
            }
            bool found = std::any_of(abnormal_ids.begin(), abnormal_ids.end(),
                                     [i](const int x) { return x == i; });
            if (found) {
                obj_color = cv::Scalar(0, 0, 255);
            }
            else {
                obj_color = cv::Scalar(0, 255, 0);
            }
            // 绘制对象矩形框
            cv::Rect2f rect = {result[i].box.x, result[i].box.y, result[i].box.width, result[i].box.height};
            // 绘制对象矩形框
            cv::rectangle(overlay, rect, obj_color, -1);
            // 绘制多标签分类标签
            if (show_attr) {
                for (int attr_id = 0; attr_id < result[i].attr_scores.size(); ++attr_id) {
                    int text_y = rect.y + (attr_id + 1) * (font_size + 2);
                    if (color_map.find(attr_id) == color_map.end()) {
                        color_map[attr_id] = get_random_color();
                    }
                    auto attr_color = color_map[attr_id];
                    std::string attr_name = label_map.find(attr_id) != label_map.end()
                                                ? label_map.at(attr_id)
                                                : std::to_string(attr_id);
                    std::string score_str = std::to_string(result[i].attr_scores[attr_id]).substr(0, 4);
                    std::string text = attr_name + ": " + score_str;
                    const auto size = cv::getTextSize(cv::Size(0, 0),
                                                      text, cv::Point(rect.x + 1, text_y), font, font_size);
                    // 绘制标签背景
                    cv::rectangle(overlay, size, attr_color, -1);
                }
            }
        }

        cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
        // 绘制对象矩形矩形边框、文字背景边框、文字
        for (int i = 0; i < result.size(); ++i) {
            if (result[i].box_score < threshold) {
                continue;
            }
            bool found = std::any_of(abnormal_ids.begin(), abnormal_ids.end(),
                                     [i](const int x) { return x == i; });
            if (found) {
                obj_color = cv::Scalar(0, 0, 255);
            }
            else {
                obj_color = cv::Scalar(0, 255, 0);
            }
            cv::Rect2f rect = {result[i].box.x, result[i].box.y, result[i].box.width, result[i].box.height};
            cv::rectangle(cv_image, rect, obj_color, 1, cv::LINE_AA, 0);
            // 绘制多标签分类标签
            if (show_attr) {
                for (int attr_id = 0; attr_id < result[i].attr_scores.size(); ++attr_id) {
                    int text_y = rect.y + (attr_id + 1) * (font_size + 2);
                    auto attr_color = color_map[attr_id];
                    std::string attr_name = label_map.find(attr_id) != label_map.end()
                                                ? label_map.at(attr_id)
                                                : std::to_string(attr_id);
                    std::string score_str = std::to_string(result[i].attr_scores[attr_id]).substr(0, 4);
                    std::string text = attr_name + ": " + score_str;
                    const auto size = cv::getTextSize(cv::Size(0, 0),
                                                      text, cv::Point(rect.x + 1, text_y), font, font_size);
                    // 绘制标签背景
                    cv::rectangle(cv_image, size, attr_color, -1, cv::LINE_AA, 0);
                    cv::putText(cv_image, text, {size.x + 1, text_y - 1}, {255, 255, 255}, font, font_size);
                }
            }
        }
        if (save_result) {
            MD_LOG_INFO << "Save attribute result to [vis_result.jpg]" << std::endl;
            cv::imwrite("vis_result.jpg", cv_image);
        }
        return image;
    }
} // namespace modeldeploy::vision
