//
// Created by aichao on 2025/7/21.
//

#pragma once
#include <opencv2/opencv.hpp>

// OpenCV 5 用 cv::FontFace（可从字体文件构造）；OpenCV 4 用 int 字体枚举
#if defined(CV_VERSION_MAJOR) && CV_VERSION_MAJOR >= 5
#define MD_FONT_OBJ cv::FontFace
#define MD_FONT_SIMPLEX cv::FontFace()
#else
#define MD_FONT_OBJ int
#define MD_FONT_SIMPLEX cv::FONT_HERSHEY_SIMPLEX
#endif

namespace modeldeploy::vision {
    cv::Scalar get_random_color();

    void draw_rectangle_and_text(cv::Mat& image, cv::Rect2f box, const std::string& text,
                                 const cv::Scalar& color, MD_FONT_OBJ font, int font_size,
                                 int thickness, bool draw_text = false);

    // 填充矩形 + 边框（alpha 混合，与 vis_* 系一致的半透明绘制）
    void draw_filled_rect(cv::Mat& image, const cv::Rect& rect, const cv::Scalar& color, double alpha);

    // 填充多边形 + 边框（alpha 混合，与 vis_ocr/vis_obb 一致）
    void draw_filled_polygon(cv::Mat& image, const std::vector<cv::Point>& points,
                             const cv::Scalar& color, double alpha);

    // 用字体文件渲染文本（OpenCV 4 需 FontFace；5 用 cv::FontFace）
    void draw_text(cv::Mat& image, const std::string& text, const std::string& font_path,
                   int font_size, const cv::Scalar& color, const cv::Point& origin);

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
