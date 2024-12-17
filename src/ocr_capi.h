//
// Created by AC on 2024/12/16.
//
#pragma once

#include "decl.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef void *OCRModelHandle;

/// 初始化模型
/// \param model_dir 模型所在目录（最好使用posix的格式，即单个正斜杠'c:/xxx/xxx'），目录最后不要带斜杠
/// \param dict_file 文本识别字典文件
/// \param thread_num 增加线程数，效率并不会线性增加，默认为1
EXPORT_DECL OCRModelHandle create_ocr_model(const char *model_dir, const char *dict_file, int thread_num = 1);

EXPORT_DECL StatusCode text_rec_buffer(OCRModelHandle model, WImage *image, WOCRResult *out_data,
                                       int draw_result, WColor color, double alpha, int is_save_result);

/// 获取文本所在位置
/// \param model 模型句柄，由create_ocr_model生成
/// \param image 原始图像
/// \param text 文本
/// \return 返回文本所在位置[x,y,w,h]
EXPORT_DECL WRect get_text_position(OCRModelHandle model, WImage *image, const char *text);

/// 根据模板图像获取模板所在原始图像中的位置
/// \param shot_img 原始图像
/// \param template_img 模板图像
/// \return 返回模板所在位置[x,y,w,h]
EXPORT_DECL WRect get_template_position(WImage *shot_img, WImage *template_img);

/// 打印文本识别结果
/// \param result
EXPORT_DECL void print_ocr_result(WOCRResult *result);

/// 释放文本识别结果
/// \param result
EXPORT_DECL void free_ocr_result(WOCRResult *result);
/// 释放模型
/// \param model
EXPORT_DECL void free_ocr_model(OCRModelHandle model);

#ifdef __cplusplus
}
#endif
