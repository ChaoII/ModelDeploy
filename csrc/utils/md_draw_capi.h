//
// Created by aichao on 2025/2/8.
//
#pragma once
#include "csrc/common/md_decl.h"
#include "csrc/common/md_types.h"


#ifdef __cplusplus
extern "C" {
#endif

/// 在图像上绘制矩形
/// \param image 原始图像
/// \param rect 矩形
/// \param color 绘制颜色
EXPORT_DECL void md_draw_rect(MDImage* image, const MDRect* rect, const MDColor* color, double alpha);

/// 在图像上绘制半透明多边形
/// \param image cv::Mat
/// \param points 多边形点集合
/// \param color 颜色
/// \param alpha 不透明度
EXPORT_DECL void md_draw_polygon(MDImage* image, const MDPolygon* polygon, const MDColor* color, double alpha);
/// 在图像上绘制文字
/// \param image 原始图像
/// \param rect 位置
/// \param font_path 字体路径
/// \param font_size 字体大小
/// \param text 文本
/// \param color 颜色
/// \param alpha 不透明度
EXPORT_DECL void md_draw_text(MDImage* image, const MDRect* rect, const char* text, const char* font_path,
                              int font_size, const MDColor* color, double alpha);




#ifdef __cplusplus
}
#endif