//
// Created by aichao on 2025/7/21.
//

#pragma once
#include <opencv2/opencv.hpp>

// OpenCV 5 用 cv::FontFace（可从字体文件构造）；OpenCV 4 用 int 字体枚举
#if defined(CV_VERSION_MAJOR) && CV_VERSION_MAJOR >= 5
#define MD_FONT_OBJ cv::FontFace
#define MD_FONT_SIMPLEX cv::FontFace::HersheySimplex
#else
#define MD_FONT_OBJ int
#define MD_FONT_SIMPLEX cv::FONT_HERSHEY_SIMPLEX
#endif

namespace modeldeploy::vision {
    cv::Scalar get_random_color();

    void draw_rectangle_and_text(cv::Mat& image, cv::Rect2f box, const std::string& text,
                                 const cv::Scalar& color, MD_FONT_OBJ font, int font_size,
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
