//
// Created by AC on 2024-12-24.
//

#include "opencv2/opencv.hpp"

#ifdef BUILD_FACE
#include "seeta/CStruct.h"
#endif

#include "types.h"

/// 将MDImage转换为cv::Mat
/// \param image
/// \return
cv::Mat md_image_to_mat(MDImage* image);

/// 将cv::Mat转换为MDImage
/// \param mat
/// \return
MDImage* mat_to_md_image(const cv::Mat &mat);

#ifdef BUILD_FACE
SeetaImageData md_image_to_seeta_image(MDImage *image);
#endif

void draw_rect_internal(cv::Mat &cv_image, const cv::Rect &rect, const cv::Scalar &cv_color, double alpha);

void draw_polygon_internal(cv::Mat &cv_image, const std::vector<cv::Point> &points,
                           const cv::Scalar &color, double alpha);

void draw_text_internal(cv::Mat &cv_image, const cv::Rect &rect, const std::string &text, const std::string &font_path,
                        int font_size, const cv::Scalar &cv_color, double alpha);

void draw_text_internal(cv::Mat &cv_image, const cv::Point &point, const std::string &text,
                        const std::string &font_path, int font_size, const cv::Scalar &cv_color);

/// 判断字符串是否包含子串
/// \param str 原始字符串
/// \param sub_str 子串
/// \return 是否包含子串
bool contains_substring(const std::string &str, const std::string &sub_str);

/// 格式化输出多边形
std::string format_polygon(MDPolygon polygon);

/// 格式化输出矩形
std::string format_rect(MDRect rect);

/// 生成随机色
cv::Scalar get_random_color();
