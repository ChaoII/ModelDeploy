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
                        const std::vector<cv::Point3f>& landmarks,
                        int landmark_radius, bool draw_lines = false);

    inline static std::vector<cv::Scalar> kps_palette =
    {
        cv::Scalar(255, 128, 0),
        cv::Scalar(255, 153, 51),
        cv::Scalar(255, 178, 102),
        cv::Scalar(230, 230, 0),
        cv::Scalar(255, 153, 255),
        cv::Scalar(153, 204, 255),
        cv::Scalar(255, 102, 255),
        cv::Scalar(255, 51, 255),
        cv::Scalar(102, 178, 255),
        cv::Scalar(51, 153, 255),
        cv::Scalar(255, 153, 153),
        cv::Scalar(255, 102, 102),
        cv::Scalar(255, 51, 51),
        cv::Scalar(153, 255, 153),
        cv::Scalar(102, 255, 102),
        cv::Scalar(51, 255, 51),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 0, 0),
        cv::Scalar(255, 255, 255),
    };
}
