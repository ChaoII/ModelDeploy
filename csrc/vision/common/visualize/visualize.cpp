//
// Created by aichao on 2025/5/22.
//

#include <random>
#include "csrc/vision/common/visualize/visualize.h"

namespace modeldeploy::vision {
    cv::Scalar get_random_color() {
        std::random_device rd; // 获取随机数种子
        std::mt19937 gen(rd()); // 使用Mersenne Twister算法生成随机数
        std::uniform_int_distribution dis(0, 255); // 定义随机数范围为1到255
        return {
            static_cast<double>(dis(gen)),
            static_cast<double>(dis(gen)),
            static_cast<double>(dis(gen))
        };
    }

    void draw_rectangle_and_text(cv::Mat& image, const cv::Rect2f box, const std::string& text,
                                 const cv::Scalar& color, cv::FontFace font, const int font_size,
                                 const int thickness, const bool draw_text) {
        // 绘制对象矩形框
        cv::rectangle(image, box, color, thickness);
        const auto size = cv::getTextSize(cv::Size(0, 0),
                                          text, cv::Point2f(box.x, box.y), font, font_size);
        // 绘制标签背景
        cv::rectangle(image, size, color, thickness);
        if (draw_text) {
            cv::putText(image, text, cv::Point2f(box.x, box.y - 2),
                        cv::Scalar(255 - color[0], 255 - color[1], 255 - color[2]),
                        font, font_size);
        }
    }

    void draw_landmarks(cv::Mat& cv_image,
                        const std::vector<cv::Point2f>& landmarks,
                        const int landmark_radius) {
        static std::map<size_t, cv::Scalar_<int>> color_map; // ← 每类颜色只初始化一次
        for (size_t i = 0; i < landmarks.size(); ++i) {
            if (color_map.find(i) == color_map.end()) {
                color_map[i] = get_random_color();
            }
            auto landmark_color = color_map[i];
            cv::circle(cv_image, landmarks[i], landmark_radius, landmark_color, -1);
        }
    }
}
