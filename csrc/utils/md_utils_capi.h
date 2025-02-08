//
// Created by AC on 2024/12/16.
//
#pragma once

#include "csrc/common/md_decl.h"
#include "csrc/common/md_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 打印位置信息
/// \param rect
EXPORT_DECL void md_print_rect(const MDRect* rect);

/// 获取按钮的可用状态，当像素点的阈值小于pix_threshold，并且此值的像素点的数量所占的百分比大于rate_threshold，则认为按钮是可用的
/// \param image 灰度图像或彩色图像
/// \param pix_threshold 像素值的阈值
/// \param rate_threshold 对应阈值内的像素所占百分比
/// \return 按钮是否可用
EXPORT_DECL bool md_get_button_enable_status(MDImage* image, int pix_threshold = 50, double rate_threshold = 0.05);

/// 获取矩形的中心点
/// \param rect
/// \return
EXPORT_DECL MDPoint md_get_center_point(const MDRect* rect);


EXPORT_DECL bool is_quantize_model(const char* model_dir);


#ifdef __cplusplus
}
#endif
