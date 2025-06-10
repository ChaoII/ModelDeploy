//
// Created by aichao on 2025/5/26.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/common/visualize/visualize.h"

namespace modeldeploy::vision {
    cv::Mat vis_cls(cv::Mat& cv_image,
                               const ClassifyResult& result,
                               const int top_k,
                               const float score_threshold,
                               const std::string& font_path,
                               const int font_size,
                               const double alpha, const bool save_result) {
        cv::Mat overlay;
        cv_image.copyTo(overlay);
        cv::FontFace font(font_path);
        constexpr int margin = 5;
        // 根据label_id获取颜色
        std::map<int, cv::Scalar_<int>> color_map;
        // 绘制半透明部分（填充矩形）
        for (int i = 0; i < top_k; ++i) {
            if (result.scores[i] < score_threshold) {
                continue;
            }
            auto class_id = result.label_ids[i];
            if (!color_map.contains(class_id)) {
                color_map[class_id] = get_random_color();
            }
            auto cv_color = color_map[class_id];
            const cv::Point origin(margin, margin + font_size * (i + 1));
            std::string text = std::to_string(class_id) + ": " + std::to_string(result.scores[i]).substr(0, 4);
            const auto size = cv::getTextSize(cv::Size(0, 0),
                                              text, origin, font, font_size);
            // 绘制标签背景
            cv::rectangle(overlay, size, cv_color, -1);
        }
        cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
        // 绘制对象矩形矩形边框、文字背景边框、文字
        for (int c = 0; c < top_k; ++c) {
            if (result.scores[c] < score_threshold) {
                continue;
            }
            auto class_id = result.label_ids[c];
            auto cv_color = color_map[class_id];
            std::string text = std::to_string(class_id) + ": " + std::to_string(result.scores[c]).substr(0, 4);
            const cv::Point origin(margin, margin + font_size * (c + 1));
            const auto size = cv::getTextSize(cv::Size(0, 0), text,
                                              origin, font, font_size);
            cv::rectangle(cv_image, size, cv_color, 1, cv::LINE_AA, 0);
            cv::putText(cv_image, text, cv::Point(origin.x, origin.y-1),
                        cv::Scalar(255 - cv_color[0], 255 - cv_color[1], 255 - cv_color[2]), font, font_size);
        }

        if (save_result) {
            MD_LOG_INFO << "Save classification result to [vis_result.jpg]" << std::endl;
            cv::imwrite("vis_result.jpg", cv_image);
        }
        return cv_image;
    }
}
