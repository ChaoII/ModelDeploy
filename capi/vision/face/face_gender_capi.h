//
// Created by aichao on 2025-4-7.
//

#pragma once


#include "capi/common/md_decl.h"
#include "capi/common/md_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 创建检测模型
/// \param model 模型，请在外部申请内存比如 malloc，new等等，该方法会在内部开辟模型内存，并赋值
/// \param model_path 模型路径
/// \param thread_num 线程数
/// \return MDStatusCode::Success成功，其他失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_create_face_gender_model(MDModel* model,
                                                              const char* model_path,
                                                              int thread_num = 8);

/// 执行预测，检测结果保存在results中
/// \param model 由create_detection_model创建的模型
/// \param c_result 检测结果结构体，请在调用该方法前申请内存，该方法会在内部申请内存，并释放
/// \param image 原始图像
/// \return MDStatusCode::Success 成功，其他失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_face_gender_predict(const MDModel* model, MDImage* image,
                                                         MDFaceGenderResult* c_result);


/// 释放检测结果
/// @param c_result 由md_detection_predict生成的检测结果
MODELDEPLOY_CAPI_EXPORT void md_free_face_gender_result(MDFaceGenderResult* c_result);

/// 释放检测模型
/// @param model 由create_detection_model创建的模型
MODELDEPLOY_CAPI_EXPORT void md_free_face_gender_model(MDModel* model);

#ifdef __cplusplus
}
#endif
