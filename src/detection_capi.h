//
// Created by AC on 2024-12-17.
//

#pragma once


#include "decl.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 创建检测模型
/// \param model  模型，请在外部申请内存比如 malloc，new等等，该方法会在内部开辟模型内存，并赋值
/// \param model_dir 模型所在目录如果format为MDModelFormat::ONNX model_dir为模型路径，如果是MDModelFormat::PaddlePaddle model_dir为模型目录
/// \param thread_num 线程数
/// \param format 模型格式
/// \return 0 成功，其他失败
EXPORT_DECL MDStatusCode md_create_detection_model(MDModel *model, const char *model_dir, int thread_num = 8,
                                              MDModelFormat format = MDModelFormat::ONNX);
///
/// \param model 由create_detection_model创建的模型
/// \param size 模型输入的大小，在onnx模型中，检测模型默认输入大小为640*640，如果模型输入大小不一致，需要调用该方法设置
/// \return 0 成功，其他失败
EXPORT_DECL MDStatusCode md_set_detection_input_size(const MDModel *model,  MDSize size);
/// 执行预测，检测结果保存在results中
/// \param model 由create_detection_model创建的模型
/// \param results 检测结果结构体，请在调用该方法前申请内存，该方法会在内部申请内存，并释放
/// \param image 原始图像
/// \return 0 成功，其他失败
EXPORT_DECL MDStatusCode md_detection_predict(const MDModel *model, MDImage *image, MDDetectionResults *results);

/// 打印检测结果
EXPORT_DECL void md_print_detection_result(const MDDetectionResults *result);

/// 绘制检测结果
/// \param image 原始图像
/// \param result 检测结果
/// \param font_path 字体路径
/// \param font_size 字体大小
/// \param alpha 不透明度
/// \param save_result 保存结果
EXPORT_DECL void md_draw_detection_result(MDImage *image,const  MDDetectionResults *result, const char *font_path,
                                          int font_size, double alpha, int save_result);

/// 释放检测结果
EXPORT_DECL void md_free_detection_result(MDDetectionResults* result);

///  释放检测模型
EXPORT_DECL void md_free_detection_model(MDModel* model);

#ifdef __cplusplus
}
#endif
