//
// Created by aichao on 2025/7/21.
//
#include <random>
#include "vision/common/visualize/utils.h"

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
                                 const cv::Scalar& color, MD_FONT_OBJ font, const int font_size,
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

    void draw_filled_rect(cv::Mat& image, const cv::Rect& rect, const cv::Scalar& color, const double alpha) {
        cv::Mat overlay;
        image.copyTo(overlay);
        cv::rectangle(overlay, rect, color, -1);
        cv::addWeighted(overlay, alpha, image, 1 - alpha, 0, image);
        cv::rectangle(image, rect, color, 1, cv::LINE_AA, 0);
    }

    void draw_filled_polygon(cv::Mat& image, const std::vector<cv::Point>& points,
                             const cv::Scalar& color, const double alpha) {
        cv::Mat overlay;
        image.copyTo(overlay);
        cv::fillPoly(overlay, points, color, cv::LINE_AA, 0);
        cv::addWeighted(overlay, alpha, image, 1 - alpha, 0, image);
        cv::polylines(image, points, true, color, 1, cv::LINE_AA, 0);
    }

    void draw_text(cv::Mat& image, const std::string& text, const std::string& font_path,
                   const int font_size, const cv::Scalar& color, const cv::Point& origin) {
        MD_FONT_OBJ font = MD_FONT_SIMPLEX;
        if (!font_path.empty()) {
            font = cv::FontFace(font_path);   // OpenCV 4.11+ / 5.x 支持从字体文件构造
        }
        cv::putText(image, text, origin, color, font, font_size);
    }

    void draw_landmarks(cv::Mat& cv_image,
                        const std::vector<cv::Point3f>& landmarks,
                        const int landmark_radius, const bool draw_lines) {
        static std::map<size_t, cv::Scalar_<int>> color_map; // ← 每类颜色只初始化一次
        for (size_t i = 0; i < landmarks.size(); ++i) {
            if (color_map.find(i) == color_map.end()) {
                color_map[i] = get_random_color();
            }
            auto landmark_color = color_map[i];
            cv::circle(cv_image, cv::Point2f(landmarks[i].x, landmarks[i].y), landmark_radius, landmark_color, -1);
        }

        if (draw_lines && landmarks.size() > 1) {
            for (int j = 0; j < landmarks.size() - 1; ++j) {
                const auto kpt0 = landmarks[j];
                const auto kpt1 = landmarks[j + 1];
                cv::Scalar line_color = color_map[j];
                cv::arrowedLine(cv_image, cv::Point(kpt0.x, kpt0.y), cv::Point(kpt1.x, kpt1.y), line_color, 1,
                                cv::LINE_AA, 0, 0.03);
            }
        }
    }
}
