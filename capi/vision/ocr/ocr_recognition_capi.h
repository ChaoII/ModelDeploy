//
// Created by aichao on 2025/3/3.
//

#pragma once

#include "capi/common/md_decl.h"
#include "capi/common/md_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 创建OCR的识别模型，可以根据一张已经裁切好的图片或者原始图片和多边形选区来识别文字
/// @param model 创建后开辟的内存，模型格式，名称等信息存储的地址的指针
/// @param model_path  模型路径
/// @param dict_path 字典路径，即词汇表
/// @param thread_num 线程数，线程数越大，识别速度越快，但是资源占用越高，并且速度并不是线性增长的请酌情修改
/// @return 创建成功返回MDStatusCode::Success，否则返回错误码
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_create_ocr_recognition_model(MDModel* model, const char* model_path,
                                                                     const char* dict_path, int thread_num = 8);

/// 单张图片识别，图片中仅包含单行文字的识别，直接给出识别结果
/// @param model 由md_create_ocr_recognition_model生成的模型
/// @param image 图像，仅包含单行文字的图像
/// @param result 识别结果
/// @return 识别成功返回MDStatusCode::Success，否则返回错误码
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_ocr_recognition_model_predict(const MDModel* model, MDImage* image,
                                                                      MDOCRResult* result);

/// 图片中包含大量的单行文字，识别给定区域内的文字，可以批量识别
/// @param model 由md_create_ocr_recognition_model生成的模型
/// @param image 图像，包含多行文字的图像
/// @param batch_size 识别的批次大小，一般情况批次大小与模型指定的线程数一致能提升识别速度
/// @param polygon 多边形选区，可以指定识别区域
/// @param size 多边形选区的数量
/// @param results 模型识别结果
/// @return 识别成功返回MDStatusCode::Success，否则返回错误码
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_ocr_recognition_model_predict_batch(
    const MDModel* model, MDImage* image, int batch_size, MDPolygon* polygon, int size, MDOCRResults* results);

/// 释放文本识别结果
/// @param result 模型识别结果，由于在填充结构体时进行了手动内存分配，所以需要手动释放
/// @return 无
MODELDEPLOY_CAPI_EXPORT void md_free_ocr_recognition_result(MDOCRResult* result);

/// 释放模型 由md_create_ocr_recognition_model生成的模型
/// @param model 模型指针
MODELDEPLOY_CAPI_EXPORT void md_free_ocr_recognition_model(MDModel* model);

#ifdef __cplusplus
}
#endif
