//
// Created by AC on 2024/12/16.
//
#pragma once

#include "../decl.h"
#include "../types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 初始化模型
/// \param model 模型，请在外部申请内存比如 malloc，new等等，该方法会在内部开辟模型内存，并赋值
/// \param parameters 文本识别模型参数结构体，参考OCRModelParameters结构体实现，注意在初始化时，一定按照注释的参数来初始化
/// \return 0 为成功，其他为失败
EXPORT_DECL MDStatusCode md_create_ocr_model(MDModel* model, MDOCRModelParameters* parameters);

/// 文本识别
/// \param model 在 create_ocr_model方法中创建的模型句柄
/// \param image 图像数据
/// \param results MDOCRResults结果，包含了OCR识别的坐标文本和置信度，可以通过print_ocr_result方法打印出来
/// \return 0 为成功，其他为失败
EXPORT_DECL MDStatusCode md_ocr_model_predict(MDModel* model, MDImage* image, MDOCRResults* results);

/// 获取文本所在位置
/// \param model 模型句柄，由create_ocr_model生成
/// \param image 原始图像
/// \param text 文本
/// \return 返回文本所在位置[x,y,w,h]
EXPORT_DECL MDRect md_get_text_position(MDModel* model, MDImage* image, const char* text);

/// 打印文本识别结果
EXPORT_DECL void md_print_ocr_result(MDOCRResults* results);

/// 绘制文本识别结果
/// \param image 原始图像
/// \param results ocr识别结果
/// \param font_path 字体路径
/// \param font_size 字体大小
/// \param color 字体颜色
/// \param alpha 不透明度
/// \param save_result 是否保存结果
EXPORT_DECL void md_draw_ocr_result(MDImage* image, MDOCRResults* results, const char* font_path,
                                    int font_size, MDColor* color, double alpha, int save_result);

/// 释放文本识别结果
EXPORT_DECL void md_free_ocr_result(MDOCRResults* results);

/// 释放模型
EXPORT_DECL void md_free_ocr_model(MDModel* model);

#ifdef __cplusplus
}
#endif
