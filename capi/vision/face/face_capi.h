//
// Created by AC on 2024-12-25.
//


#pragma once

#include "capi/common/md_decl.h"
#include "capi/common/md_types.h"
#include "capi/common/md_micro.h"


#ifdef __cplusplus
extern "C" {
#endif

/// 创建人脸识别模型
/// @param model 人脸识别模型
/// @param model_dir 模型路径（注意模型路径如果有中文会报错--路径乱码），模型文件必须放在该文件夹下,并且模型名称不能修改
/// @param flags 人脸模型启用标志，人脸模型包括人脸检测、人脸识别、人脸属性、人脸质量评估、活体检测等，按需开启，也可以默认全开
/// @param thread_num 线程数，线程数越大越快
/// @return MDStatusCode
MODELDEPLOY_CAPI_EXPORT MDStatusCode
md_create_face_model(MDModel* model, const char* model_dir, int flags = MD_MASK, int thread_num = 1);

/// 人脸检测
/// @param model 由md_create_face_model创建的模型
/// @param image 包含人脸的图片
/// @param result 人脸检测框、阈值等信息
/// @return MDStatusCode
MODELDEPLOY_CAPI_EXPORT MDStatusCode
md_face_detection(const MDModel* model, MDImage* image, MDDetectionResults* result);

/// 人脸关键点模型，采用五点关键点，双眼、鼻尖、俩嘴角
/// @param model 由md_create_face_model创建的模型
/// @param image 包含人脸的图片
/// @param rect 由md_face_detection返回的矩形框
/// @param result 人脸关键点结果（5个点）
/// @return MDStatusCode
MODELDEPLOY_CAPI_EXPORT MDStatusCode
md_face_marker(const MDModel* model, MDImage* image, const MDRect* rect, MDLandMarkResult* result);

/// 人脸特征抽取，根据人脸关键点进行特征抽取
/// @param model 由md_create_face_model创建的模型
/// @param image 包含人脸的图片
/// @param points 由md_face_marker返回的人脸关键点
/// @param feature 人脸特征信息，512维的特征向量
/// @return MDStatusCode
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_face_feature(const MDModel* model, MDImage* image,
                                                     const MDLandMarkResult* points, MDFaceFeature* feature);

/// 端到端的人脸特征抽取，直接传入一张图片即可
/// @param model 由md_create_face_model创建的模型
/// @param image 包含人脸的图片
/// @param feature 人脸特征信息，512维的特征向量
/// @return MDStatusCode
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_face_feature_e2e(const MDModel* model, MDImage* image, MDFaceFeature* feature);

/// 人脸特征对比，也就是相似性计算，采用cosine距离，越接近1（向量夹角为0 cos0°=1），越相似
/// @param model 由md_create_face_model创建的模型
/// @param feature1 人脸特征信息，512维的特征向量
/// @param feature2 人脸特征信息，512维的特征向量
/// @param similarity 相似度[0-1]
/// @return MDStatusCode
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_face_feature_compare(const MDModel* model, const MDFaceFeature* feature1,
                                                             const MDFaceFeature* feature2, float* similarity);

/// 人脸活体检测
/// @param model 由md_create_face_model创建的模型
/// @param image 包含人脸的图片
/// @param result MDFaceAntiSpoofingResult枚举结果
/// @return MDStatusCode
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_face_anti_spoofing(const MDModel* model, MDImage* image,
                                                           MDFaceAntiSpoofingResult* result);

/// 人脸质量评估
/// @param model 由md_create_face_model创建的模型
/// @param image 包含人脸的图片
/// @param type 质量评估的类型包含亮度、清晰度、完整性、遮挡、角度
/// @param result 人脸质量评估结果MDFaceQualityEvaluateResult 高、中、低
/// @return MDStatusCode
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_face_quality_evaluate(const MDModel* model, MDImage* image,
                                                              MDFaceQualityEvaluateType type,
                                                              MDFaceQualityEvaluateResult* result);

/// 人脸年龄预测
/// @param model 由md_create_face_model创建的模型
/// @param image 包含人脸的图片
/// @param age 输出年龄
/// @return MDStatusCode
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_face_age_predict(const MDModel* model, MDImage* image, int* age);

/// 人脸性别预测
/// @param model 由md_create_face_model创建的模型
/// @param image 包含人脸的图片
/// @param gender 输出性别
/// @return MDStatusCode
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_face_gender_predict(const MDModel* model, MDImage* image,
                                                            MDGenderResult* gender);

/// 眼睛状态预测（睁眼闭眼）
/// @param model 由md_create_face_model创建的模型
/// @param image 包含人脸的图片
/// @param eye_state 输出双眼的状态
/// @return MDStatusCode
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_face_eye_state_predict(const MDModel* model, MDImage* image,
                                                               MDEyeStateResult* eye_state);

/// 打印人脸活体检测结果
/// @param result 人脸活体检测结果
MODELDEPLOY_CAPI_EXPORT void md_print_face_anti_spoofing_result(MDFaceAntiSpoofingResult result);

/// 释放人脸关键点结果，在关键点预测处开辟了内存
/// @param result
MODELDEPLOY_CAPI_EXPORT void md_free_face_landmark(MDLandMarkResult* result);

/// 释放人脸特征结果，在特征预测处开辟了内存
/// @param feature 由md_face_feature或md_face_feature_e2e返回
MODELDEPLOY_CAPI_EXPORT void md_free_face_feature(MDFaceFeature* feature);

/// 释放人脸模型，在创建人脸模型处开辟了内存
/// @param model 由md_create_face_model创建的模型
MODELDEPLOY_CAPI_EXPORT void md_free_face_model(MDModel* model);


#ifdef __cplusplus
}
#endif
