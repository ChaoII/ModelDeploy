//
// Created by AC on 2024/12/16.
//
#pragma once

#include "capi/common/md_decl.h"
#include "capi/common/md_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 初始化模型
/// \param model 模型，请在外部申请内存比如 malloc，new等等，该方法会在内部开辟模型内存，并赋值
/// \param parameters 文本识别模型参数结构体，参考OCRModelParameters结构体实现，注意在初始化时，一定参考注释的参数来初始化
/// \param option
/// \return 成功时返回MDStatusCode::Success，其他为失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_create_structure_table_model(
    MDModel* model, const MDStructureTableModelParameters* parameters, const MDRuntimeOption* option);

/// 文本识别
/// \param model 在 create_structure_table_model方法中创建的模型句柄
/// \param image 图像数据
/// \param c_results MDOCRResults结果，包含了OCR识别的坐标文本和置信度，可以通过print_structure_table_result方法打印出来
/// \return 成功时返回MDStatusCode::Success，其他为失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_structure_table_model_predict(
    const MDModel* model, MDImage* image,
    MDOCRResults* c_results);


/// 打印文本识别结果
MODELDEPLOY_CAPI_EXPORT void md_print_structure_table_result(
    const MDOCRResults* c_results);

/// 绘制文本识别结果
/// \param image 原始图像
/// \param c_results structure_table识别结果
/// \param font_path 字体路径
/// \param font_size 字体大小
/// \param alpha 不透明度
/// \param save_result 是否保存结果
MODELDEPLOY_CAPI_EXPORT void md_draw_structure_table_result(
    const MDImage* image, const MDOCRResults* c_results,
    const char* font_path,
    int font_size, double alpha, int save_result);

/// 释放文本识别结果
/// @param c_results 模型识别结果，通过md_structure_table_model_predict方法手动开辟了内存空间，需要手动释放
MODELDEPLOY_CAPI_EXPORT void md_free_structure_table_result(MDOCRResults* c_results);

/// 释放模型
/// @param model 由create_structure_table_model生成
MODELDEPLOY_CAPI_EXPORT void md_free_structure_table_model(MDModel* model);

#ifdef __cplusplus
}
#endif
