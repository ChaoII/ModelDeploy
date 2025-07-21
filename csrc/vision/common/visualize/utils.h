//
// Created by aichao on 2025/7/21.
//

#pragma once
#include <opencv2/opencv.hpp>

namespace modeldeploy::vision {
    cv::Scalar get_random_color();

    void draw_rectangle_and_text(cv::Mat& image, cv::Rect2f box, const std::string& text,
                                 const cv::Scalar& color, cv::FontFace font, int font_size,
                                 int thickness, bool draw_text = false);

    void draw_landmarks(cv::Mat& cv_image,
                        const std::vector<cv::Point2f>& landmarks,
                        int landmark_radius);
}
