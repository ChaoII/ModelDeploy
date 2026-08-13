//
// Created by aichao on 2026/8/13.
//

#pragma once

#include "capi/common/md_decl.h"
#include "capi/common/md_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 创建语义分割模型
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_create_sem_model(
    MDModel* model, const char* model_path, const MDRuntimeOption* option);

/// 设置输入尺寸
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_set_sem_input_size(
    const MDModel* model, MDSize size);

/// 执行预测
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_sem_predict(
    const MDModel* model, MDImage* image, MDSemSegResult* c_result);
/// NV12 直接输入预测（硬解码/摄像头直通，省去 BGR 转换）。
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_sem_predict_nv12(
    const MDModel* model,
    const unsigned char* src_y, const unsigned char* src_uv,
    int width, int height, int step_y, int step_uv,
    MDDevice src_device,
    MDSemSegResult* c_results);


/// 释放结果
MODELDEPLOY_CAPI_EXPORT void md_free_sem_result(MDSemSegResult* c_result);

/// 释放模型
MODELDEPLOY_CAPI_EXPORT void md_free_sem_model(MDModel* model);

#ifdef __cplusplus
}
#endif
