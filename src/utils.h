//
// Created by AC on 2024/12/16.
//
#pragma once

#include "decl.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif


/// 打印位置信息
/// \param rect
EXPORT_DECL void print_rect(WRect rect);

/// 在图像上绘制矩形
/// \param image 原始图像
/// \param rect 矩形
/// \param color 绘制颜色
EXPORT_DECL void draw_rect(WImage *image, WRect rect, WColor color, double alpha);

/// 在图像上绘制半透明多边形
/// \param image cv::Mat
/// \param points 多边形点集合
/// \param color 颜色
/// \param alpha 不透明度
EXPORT_DECL void draw_polygon(WImage *image, WPolygon *polygon, WColor color, double alpha);
/// 在图像上绘制文字
/// \param image 原始图像
/// \param rect 位置
/// \param font_path 字体路径
/// \param font_size 字体大小
/// \param text 文本
/// \param color 颜色
/// \param alpha 不透明度
EXPORT_DECL void
draw_text(WImage *image, WRect rect, const char *text, const char *font_path, int font_size, WColor color,
          double alpha);

/// 弹窗显示image
/// \param image
EXPORT_DECL void show_image(WImage *image);

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

/// 开辟一个WModel指针
EXPORT_DECL WModel *allocate_model();

#ifdef __cplusplus
}
#endif