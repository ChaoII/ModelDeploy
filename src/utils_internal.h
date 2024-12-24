//
// Created by AC on 2024-12-24.
//

#include "opencv2/opencv.hpp"
#include "types.h"

/// 将WImage转换为cv::Mat
/// \param image
/// \return
cv::Mat wimage_to_mat(WImage *image);

/// 将cv::Mat转换为WImage
/// \param mat
/// \return
WImage *mat_to_wimage(const cv::Mat &mat);


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
std::string format_polygon(WPolygon polygon);

/// 格式化输出矩形
std::string format_rect(WRect rect);

/// 生成随机色
cv::Scalar get_random_color();
