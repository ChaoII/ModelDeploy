//
// Created by AC on 2024/12/16.
//
#pragma once

#include "wrzs_capi.h"
#include <opencv2/opencv.hpp>
#include "decl.h"


EXPORT_DECL cv::Mat wimage_to_mat(WImage *image);


EXPORT_DECL WImage *mat_to_wimage(const cv::Mat &mat);


EXPORT_DECL void print_rect(WRect rect);


EXPORT_DECL void draw_rect(WImage *image, WRect rect, WColor color);

EXPORT_DECL void draw_transparent_rectangle(cv::Mat &image, const std::vector<cv::Point> &points,
                                            const cv::Scalar &color, double alpha);

EXPORT_DECL void draw_text(WImage *image, WRect rect, const char *text, WColor color, double alpha);

EXPORT_DECL void show_image(WImage *image);

EXPORT_DECL bool contains_substring(const std::string &str, const std::string &sub_str);

/// 获取按钮的可用状态，当像素点的阈值小于pix_threshold，并且此值的像素点的数量所占的百分比大于rate_threshold，则认为按钮是可用的
/// \param image 灰度图像或彩色图像
/// \param pix_threshold 像素值的阈值
/// \param rate_threshold 对应阈值内的像素所占百分比
/// \return 按钮是否可用
EXPORT_DECL bool get_button_enable_status(WImage *image, int pix_threshold = 50, double rate_threshold = 0.05);

EXPORT_DECL WImage *crop_image(WImage *image, WRect rect);

EXPORT_DECL WImage *clone_image(WImage *image);

EXPORT_DECL WImage *from_compressed_bytes(const unsigned char *bytes, int size);