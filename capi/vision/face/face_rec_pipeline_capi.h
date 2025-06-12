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
/// \param face_det_model_file
/// \param face_rec_model_file
/// \param option
/// \return MDStatusCode::Success成功，其他失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_create_face_rec_pipeline_model(
    MDModel* model, const char* face_det_model_file, const char* face_rec_model_file, const MDRuntimeOption* option);

/// 执行预测，检测结果保存在results中
/// \param model 由create_detection_model创建的模型
/// \param image 原始图像
/// \param c_results
/// \return MDStatusCode::Success 成功，其他失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_face_rec_pipeline_predict(
    const MDModel* model,
    MDImage* image,
    MDFaceRecognizerResults* c_results);

MODELDEPLOY_CAPI_EXPORT void md_print_face_rec_pipeline_result(const MDFaceRecognizerResults* c_results);

/// 释放检测结果
/// @param c_results 由md_detection_predict生成的检测结果
MODELDEPLOY_CAPI_EXPORT void md_free_face_rec_pipeline_result(MDFaceRecognizerResults* c_results);
/// 释放检测模型
/// @param model 由create_detection_model创建的模型
MODELDEPLOY_CAPI_EXPORT void md_free_face_rec_pipeline_model(MDModel* model);

#ifdef __cplusplus
}
#endif
