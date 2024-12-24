//
// Created by AC on 2024/12/16.
//
#pragma once

#include "decl.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 初始化模型
/// \param model_dir 模型所在目录（最好使用posix的格式，即单个正斜杠'c:/xxx/xxx'），目录最后不要带斜杠
/// \param dict_file 文本识别字典文件
/// \param thread_num 增加线程数，效率并不会线性增加，默认为1
EXPORT_DECL StatusCode create_ocr_model(WModel *model, OCRModelParameters *parameters);

EXPORT_DECL StatusCode ocr_model_predict(WModel *model, WImage *image, WOCRResults *results,
                                         int draw_result, WColor color, double alpha, int is_save_result);


/// 获取文本所在位置
/// \param model 模型句柄，由create_ocr_model生成
/// \param image 原始图像
/// \param text 文本
/// \return 返回文本所在位置[x,y,w,h]
EXPORT_DECL WRect get_text_position(WModel *model, WImage *image, const char *text);

/// 打印文本识别结果
/// \param result
EXPORT_DECL void print_ocr_result(WOCRResults *results);

/// 释放文本识别结果
/// \param result
EXPORT_DECL void free_ocr_result(WOCRResults *results);
/// 释放模型
/// \param model
EXPORT_DECL void free_ocr_model(WModel *model);

#ifdef __cplusplus
}
#endif
