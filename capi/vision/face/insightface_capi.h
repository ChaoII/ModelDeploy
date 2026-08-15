//
// insightface buffalo_l 人脸分析 C API。
//

#pragma once

#include "capi/common/md_decl.h"
#include "capi/common/md_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 创建 insightface 综合人脸分析模型（det + 2d106 + 3d68 + recognition）
/// \param model 模型
/// \param det_model_path det_10g.onnx 路径
/// \param rec_model_path w600k_r50.onnx 路径
/// \param lmk2d_model_path 2d106det.onnx 路径
/// \param lmk3d_model_path 1k3d68.onnx 路径
/// \param option
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_create_insightface_model(
    MDModel* model,
    const char* det_model_path,
    const char* rec_model_path,
    const char* lmk2d_model_path,
    const char* lmk3d_model_path,
    const MDRuntimeOption* option);

/// 创建 insightface 检测模型（仅 det_10g）
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_create_insightface_det_model(
    MDModel* model, const char* model_path, const MDRuntimeOption* option);

/// 人脸分析：检测 + 2D/3D 关键点 + 识别
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_insightface_analyze(
    const MDModel* model, MDImage* image, MDInsightFaceResults* c_results);

/// 仅检测
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_insightface_detect(
    const MDModel* model, MDImage* image, MDKeyPointResults* c_results);

/// 设置检测阈值
MODELDEPLOY_CAPI_EXPORT void md_insightface_set_det_thresh(MDModel* model, float thresh);

/// 释放人脸分析结果
MODELDEPLOY_CAPI_EXPORT void md_free_insightface_result(MDInsightFaceResults* c_results);

/// 释放模型
MODELDEPLOY_CAPI_EXPORT void md_free_insightface_model(MDModel* model);

#ifdef __cplusplus
}
#endif
