//
// Created by aichao on 2025/4/1.
//

#include "core/md_log.h"
#include "vision/common/visualize/visualize.h"


namespace modeldeploy::vision {
    cv::Mat vis_iseg(cv::Mat& cv_image, const std::vector<InstanceSegResult>& result,
                     const double threshold,
                     const std::string& font_path, const int font_size,
                     const double alpha, const bool save_result) {
        cv::Mat overlay;
        cv_image.copyTo(overlay);
        cv::FontFace font(font_path);
        // 根据label_id获取颜色
        static std::map<int, cv::Scalar_<int>> color_map;
        // 绘制半透明部分（填充矩形）
        for (int i = 0; i < result.size(); ++i) {
            if (result[i].score < threshold) {
                continue;
            }
            auto class_id = result[i].label_id;
            if (color_map.find(class_id) == color_map.end()) {
                color_map[class_id] = get_random_color();
            }
            cv::Rect2f rect = {result[i].box.x, result[i].box.y, result[i].box.width, result[i].box.height};
            auto cv_color = color_map[class_id];
            // 绘制对象矩形框
            cv::rectangle(overlay, rect, cv_color, -1);
            std::string text = std::to_string(class_id) + ": " + std::to_string(result[i].score).substr(0, 4);
            const auto size = cv::getTextSize(cv::Size(0, 0),
                                              text, cv::Point(rect.x, rect.y), font, font_size);
            // 绘制标签背景
            cv::rectangle(overlay, size, cv_color, -1);
        }

        cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
        // 绘制对象矩形矩形边框、文字背景边框、文字
        for (int c = 0; c < result.size(); ++c) {
            if (result[c].score < threshold) {
                continue;
            }
            auto class_id = result[c].label_id;
            cv::Rect2f rect = {result[c].box.x, result[c].box.y, result[c].box.width, result[c].box.height};
            auto cv_color = color_map[class_id];
            cv::rectangle(cv_image, rect, cv_color, 1, cv::LINE_AA, 0);
            std::string text = std::to_string(class_id) + ": " + std::to_string(result[c].score).substr(0, 4);
            const auto size = cv::getTextSize(cv::Size(0, 0), text,
                                              cv::Point2f(rect.x, rect.y), font, font_size);
            cv::rectangle(cv_image, size, cv_color, 1, cv::LINE_AA, 0);
            cv::putText(cv_image, text, cv::Point2f(rect.x, rect.y - 2),
                        cv::Scalar(255 - cv_color[0], 255 - cv_color[1], 255 - cv_color[2]), font, font_size);

            int mask_h = static_cast<int>(result[c].mask.shape[0]);
            int mask_w = static_cast<int>(result[c].mask.shape[1]);
            cv::Mat mask(mask_h, mask_w, CV_8UC1,
                         const_cast<uint8_t*>(static_cast<const uint8_t*>(result[c].mask.data())));
            if (mask_h != rect.height || mask_w != rect.width) {
                cv::resize(mask, mask, rect.size());
            }
            // 创建一个彩色 mask 图层
            cv::Mat color_mask(rect.size(), CV_8UC3);
            int mc0 = 255 - cv_color[0] >= 127 ? 255 - cv_color[0] : 127;
            int mc1 = 255 - cv_color[1] >= 127 ? 255 - cv_color[1] : 127;
            int mc2 = 255 - cv_color[2] >= 127 ? 255 - cv_color[2] : 127;
            color_mask.setTo(cv::Scalar(mc0, mc1, mc2));
            // 将 mask 应用到 color_mask 上
            cv::Mat colored_mask;
            cv::bitwise_and(color_mask, color_mask, colored_mask, mask);
            // 定义 ROI 区域
            cv::Mat roi(cv_image, rect);
            // 使用 addWeighted 混合原始图像和 mask
            cv::addWeighted(roi, 0.2, colored_mask, 0.8, 0, roi);
        }
        // for roted_box
        if (save_result) {
            MD_LOG_INFO << "Save detection result to [vis_result.jpg]" << std::endl;
            cv::imwrite("vis_result.jpg", cv_image);
        }
        return cv_image;
    }
} // namespace modeldeploy::vision
