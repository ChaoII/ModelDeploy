//
// Created by aichao on 2025/4/1.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/common/visualize/visualize.h"

namespace modeldeploy ::vision {
    cv::Mat VisOcr(cv::Mat image, const OCRResult& results, const std::string& font_path,
                   const int font_size, const double alpha, const int save_result) {
        cv::Mat overlay;
        image.copyTo(overlay);
        cv::FontFace font(font_path);
        cv::Scalar cv_color = get_random_color();
        // 绘制半透明部分（填充多边形）和文字背景色
        for (int i = 0; i < results.boxes.size(); ++i) {
            const auto polygon = results.boxes[i];
            std::vector<cv::Point> points;
            points.reserve(polygon.size() / 2);
            for (int j = 0; j < polygon.size(); j += 2) {
                points.emplace_back(polygon[j], polygon[j + 1]);
            }
            cv::fillPoly(overlay, points, cv_color, cv::LINE_AA, 0);
            const auto size = cv::getTextSize(cv::Size(0, 0), results.text[i],
                                              {points[0].x, points[0].y}, font, font_size);
            cv::rectangle(image, size, cv_color, -1, cv::LINE_AA, 0);
        }
        cv::addWeighted(overlay, alpha, image, 1 - alpha, 0, image);
        // 绘制多边形边框，文字背景边框，文字
        for (int i = 0; i < results.boxes.size(); ++i) {
            const auto polygon = results.boxes[i];
            std::vector<cv::Point> points;
            points.reserve(polygon.size());
            for (int j = 0; j < polygon.size(); j += 2) {
                points.emplace_back(polygon[j], polygon[j + 1]);
            }
            cv::polylines(image, points, true, cv_color, 1, cv::LINE_AA, 0);
            const auto size = cv::getTextSize(cv::Size(0, 0), results.text[i],
                                              {points[0].x, points[0].y}, font, font_size);
            cv::rectangle(image, size, cv_color, 1, cv::LINE_AA, 0);
            const auto inv_color = cv::Scalar(255 - cv_color[0], 255 - cv_color[1], 255 - cv_color[2]);
            cv::putText(image, results.text[i], {points[0].x, points[0].y - 2},
                        inv_color, font, font_size);
        }
        if (save_result) {
            cv::imwrite("vis_result.jpg", image);
        }
        return image;
    }
}
