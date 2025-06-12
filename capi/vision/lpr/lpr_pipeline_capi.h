//
// Created by aichao on 2025-4-8.
//

#pragma once


#include "capi/common/md_decl.h"
#include "capi/common/md_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 创建检测模型
/// \param model 模型，请在外部申请内存比如 malloc，new等等，该方法会在内部开辟模型内存，并赋值
/// \param lpr_det_model_file
/// \param lpr_rec_model_file
/// \param option
/// \return MDStatusCode::Success成功，其他失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_create_lpr_pipeline_model(
    MDModel* model, const char* lpr_det_model_file, const char* lpr_rec_model_file, const MDRuntimeOption* option);

/// 执行预测，检测结果保存在results中
/// \param model 由create_detection_model创建的模型
/// \param image 原始图像
/// \param c_results
/// \return MDStatusCode::Success 成功，其他失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_lpr_pipeline_predict(
    const MDModel* model, MDImage* image,
    MDLPRResults* c_results);

MODELDEPLOY_CAPI_EXPORT void md_print_lpr_pipeline_result(
    const MDLPRResults* c_results);


/// 绘制检测结果
/// \param image 原始图像
/// \param c_results 检测结果
/// \param font_path 字体路径
/// \param font_size 字体大小
/// \param landmark_radius
/// \param alpha 不透明度
/// \param save_result 保存结果, 如果>0将保存绘制后的图片vis_result.jpg
MODELDEPLOY_CAPI_EXPORT void md_draw_lpr_pipeline_result(
    const MDImage* image, const MDLPRResults* c_results,
    const char* font_path, int font_size = 14,
    int landmark_radius = 4, double alpha = 0.5,
    int save_result = 0);

/// 释放检测结果
/// @param c_results 由md_detection_predict生成的检测结果
MODELDEPLOY_CAPI_EXPORT void md_free_lpr_pipeline_result(MDLPRResults* c_results);
/// 释放检测模型
/// @param model 由create_detection_model创建的模型
MODELDEPLOY_CAPI_EXPORT void md_free_lpr_pipeline_model(MDModel* model);

#ifdef __cplusplus
}
#endif
