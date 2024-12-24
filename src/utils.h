//
// Created by AC on 2024/12/16.
//
#pragma once

#include "decl.h"
#include "types.h"
#include <opencv2/opencv.hpp>


/// 将WImage转换为cv::Mat
/// \param image
/// \return
EXPORT_DECL cv::Mat wimage_to_mat(WImage *image);

/// 将cv::Mat转换为WImage
/// \param mat
/// \return
EXPORT_DECL WImage *mat_to_wimage(const cv::Mat &mat);

/// 打印位置信息
/// \param rect
EXPORT_DECL void print_rect(WRect rect);

/// 在图像上绘制矩形
/// \param image 原始图像
/// \param rect 矩形
/// \param color 绘制颜色
EXPORT_DECL void draw_rect(WImage *image, WRect rect, WColor color);

/// 在图像上绘制半透明矩形框
/// \param image cv::Mat
/// \param points 多边形点集合
/// \param color 颜色
/// \param alpha 不透明度
EXPORT_DECL void draw_transparent_rectangle(cv::Mat &image, const std::vector<cv::Point> &points,
                                            const cv::Scalar &color, double alpha);
/// 在图像上绘制文字
/// \param image 原始图像
/// \param rect 位置
/// \param text 文本
/// \param color 颜色
/// \param alpha 不透明度
EXPORT_DECL void draw_text(WImage *image, WRect rect, const char *text, WColor color, double alpha);

/// 弹窗显示image
/// \param image
EXPORT_DECL void show_image(WImage *image);

/// 判断字符串是否包含子串
/// \param str 原始字符串
/// \param sub_str 子串
/// \return 是否包含子串
EXPORT_DECL bool contains_substring(const std::string &str, const std::string &sub_str);

/// 获取按钮的可用状态，当像素点的阈值小于pix_threshold，并且此值的像素点的数量所占的百分比大于rate_threshold，则认为按钮是可用的
/// \param image 灰度图像或彩色图像
/// \param pix_threshold 像素值的阈值
/// \param rate_threshold 对应阈值内的像素所占百分比
/// \return 按钮是否可用
EXPORT_DECL bool get_button_enable_status(WImage *image, int pix_threshold = 50, double rate_threshold = 0.05);

/// 根据对应位置的图像裁剪（注意需要手动释放WImage）
/// \param image 原始图像
/// \param rect 待裁剪的区域
/// \return 裁剪后的图像
EXPORT_DECL WImage *crop_image(WImage *image, WRect rect);

/// 克隆一个WImage(注意需要手动释放WIMage指针)
/// \param image
/// \return WImage 指针
EXPORT_DECL WImage *clone_image(WImage *image);

/// 从压缩字节生成一个WImage指针（需要手动释放）
/// \param bytes 压缩字节，比如.jpg的buffer数据
/// \param size 字节长度
/// \return WImage指针
EXPORT_DECL WImage *from_compressed_bytes(const unsigned char *bytes, int size);

/// 获取矩形的中心点
/// \param rect
/// \return
EXPORT_DECL WPoint get_center_point(WRect rect);

/// 释放WImage指针
/// \param img
EXPORT_DECL void free_wimage(WImage *img);

/// 从文件读取WImage
/// \param path 图像路径
/// \return WImage指针
EXPORT_DECL WImage *read_image(const char *path);

/// 格式化输出多边形
/// \param polygon
/// \return
EXPORT_DECL std::string format_polygon(WPolygon polygon);

EXPORT_DECL std::string format_rect(WRect rect);

EXPORT_DECL WModel *allocate_model();

