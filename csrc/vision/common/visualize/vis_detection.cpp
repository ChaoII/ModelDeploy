//
// Created by aichao on 2025/4/1.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/common/visualize/visualize.h"


namespace modeldeploy::vision {
    cv::Mat vis_detection(cv::Mat& cv_image, const DetectionResult& result,
                          const std::string& font_path, const int font_size,
                          const double alpha, const int save_result) {
        cv::Mat overlay;
        cv_image.copyTo(overlay);
        cv::FontFace font(font_path);
        // 根据label_id获取颜色
        std::map<int, cv::Scalar_<int>> color_map;
        // 绘制半透明部分（填充矩形）
        for (int i = 0; i < result.boxes.size(); ++i) {
            auto class_id = result.label_ids[i];
            if (!color_map.contains(class_id)) {
                color_map[class_id] = get_random_color();
            }
            auto x1 = static_cast<int>(result.boxes[i][0]);
            auto y1 = static_cast<int>(result.boxes[i][1]);
            auto x2 = static_cast<int>(result.boxes[i][2]);
            auto y2 = static_cast<int>(result.boxes[i][3]);
            auto cv_color = color_map[class_id];
            // 绘制对象矩形框
            cv::rectangle(overlay, cv::Point(x1, y1), cv::Point(x2, y2), cv_color, -1);
            std::string text = std::to_string(class_id) + ": " + std::to_string(result.scores[i]).substr(0, 4);
            const auto size = cv::getTextSize(cv::Size(0, 0),
                                              text, cv::Point(x1, y1), font, font_size);
            // 绘制标签背景
            cv::rectangle(overlay, size, cv_color, -1);
        }
        cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
        // 绘制对象矩形矩形边框、文字背景边框、文字
        for (int c = 0; c < result.boxes.size(); ++c) {
            auto class_id = result.label_ids[c];
            int x1 = static_cast<int>(round(result.boxes[c][0]));
            int y1 = static_cast<int>(round(result.boxes[c][1]));
            int x2 = static_cast<int>(round(result.boxes[c][2]));
            int y2 = static_cast<int>(round(result.boxes[c][3]));
            auto cv_color = color_map[class_id];
            cv::rectangle(cv_image, cv::Point(x1, y1), cv::Point(x2, y2), cv_color, 1, cv::LINE_AA, 0);
            std::string text = std::to_string(class_id) + ": " + std::to_string(result.scores[c]).substr(0, 4);
            const auto size = cv::getTextSize(cv::Size(0, 0), text,
                                              cv::Point(x1, y1), font, font_size);
            cv::rectangle(cv_image, size, cv_color, 1, cv::LINE_AA, 0);
            cv::putText(cv_image, text, cv::Point(x1, y1 - 2),
                        cv::Scalar(255 - cv_color[0], 255 - cv_color[1], 255 - cv_color[2]), font, font_size);
            int box_h = y2 - y1;
            int box_w = x2 - x1;
            int im_w = cv_image.cols;
            if (result.contain_masks) {
                int mask_h = static_cast<int>(result.masks[c].shape[0]);
                int mask_w = static_cast<int>(result.masks[c].shape[1]);
                // non-const pointer for cv:Mat constructor
                auto* mask_raw_data = const_cast<uint8_t*>(
                    static_cast<const uint8_t*>(result.masks[c].data()));
                // only reference to mask data (zero copy)
                cv::Mat mask(mask_h, mask_w, CV_8UC1, mask_raw_data);
                if (mask_h != box_h || mask_w != box_w) {
                    cv::resize(mask, mask, cv::Size(box_w, box_h));
                }
                // use a bright color for instance mask
                int mc0 = 255 - cv_color[0] >= 127 ? 255 - cv_color[0] : 127;
                int mc1 = 255 - cv_color[1] >= 127 ? 255 - cv_color[1] : 127;
                int mc2 = 255 - cv_color[2] >= 127 ? 255 - cv_color[2] : 127;
                auto* mask_data = mask.data;
                // inplace blending (zero copy)
                auto* vis_im_data = cv_image.data;
                for (size_t i = y1; i < y2; ++i) {
                    for (size_t j = x1; j < x2; ++j) {
                        if (mask_data[(i - y1) * mask_w + (j - x1)] != 0) {
                            vis_im_data[i * im_w * 3 + j * 3 + 0] = cv::saturate_cast<uchar>(
                                static_cast<float>(mc0) * 0.5f +
                                static_cast<float>(vis_im_data[i * im_w * 3 + j * 3 + 0]) * 0.5f);
                            vis_im_data[i * im_w * 3 + j * 3 + 1] = cv::saturate_cast<uchar>(
                                static_cast<float>(mc1) * 0.5f +
                                static_cast<float>(vis_im_data[i * im_w * 3 + j * 3 + 1]) * 0.5f);
                            vis_im_data[i * im_w * 3 + j * 3 + 2] = cv::saturate_cast<uchar>(
                                static_cast<float>(mc2) * 0.5f +
                                static_cast<float>(vis_im_data[i * im_w * 3 + j * 3 + 2]) * 0.5f);
                        }
                    }
                }
            }
        }

        if (save_result) {
            MD_LOG_INFO << "Save detection result to [vis_result.jpg]" << std::endl;
            cv::imwrite("vis_result.jpg", cv_image);
        }
        return cv_image;
    }
} // namespace modeldeploy::vision
