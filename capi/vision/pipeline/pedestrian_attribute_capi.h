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
/// \param det_model_path 检测模型路径
/// \param cls_model_path 分标签类模型路径
/// \param option 和runtime相关的参数，比如GPU推理，fp16半精度推理等
/// \return 成功时返回MDStatusCode::Success，其他为失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_create_attr_model(
    MDModel* model, const char* det_model_path, const char* cls_model_path, const MDRuntimeOption* option);

///
/// \param model md_create_attr_model创建的模型
/// \param size 模型输入的大小，在onnx模型中，检测模型默认输入大小为640*640，如果模型输入大小不一致，需要调用该方法设置
/// \return MDStatusCode::Success 成功，其他失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_set_attr_det_input_size(
    const MDModel* model, MDSize size);

///
/// \param model md_create_attr_model创建的模型
/// \param size 模型输入的大小，在onnx模型中，多标签分类模型默认输入大小为224*224，如果模型输入大小不一致，需要调用该方法设置
/// \return MDStatusCode::Success 成功，其他失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_set_attr_cls_input_size(
    const MDModel* model, MDSize size);
///
/// @param model md_create_attr_model创建的模型
/// @param batch_size  模型输入的batch_size
/// @return MDStatusCode::Success 成功，其他失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_set_attr_cls_batch_size(const MDModel* model, int batch_size);

///
/// @param model 在 md_create_attr_model创建的模型
/// @param threshold 模型输入的阈值
/// @return  MDStatusCode::Success 成功，其他失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_set_attr_det_threshold(const MDModel* model, float threshold);


/// 文本识别
/// \param model 在 md_create_attr_model创建的模型
/// \param image 图像数据
/// \param c_results MDOCRResults结果，包含了OCR识别的坐标文本和置信度，可以通过print_ocr_result方法打印出来
/// \return 成功时返回MDStatusCode::Success，其他为失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode
md_attr_model_predict(const MDModel* model, MDImage* image, MDAttributeResults* c_results);


/// 打印文本识别结果
MODELDEPLOY_CAPI_EXPORT void md_print_attr_result(const MDAttributeResults* c_results);

/// 绘制文本识别结果
/// \param image 原始图像
/// \param c_results attr识别结果
/// \param threshold 阈值
/// \param c_label_map 标签映射
/// \param font_path 字体路径
/// \param font_size 字体大小
/// \param alpha 不透明度
/// \param save_result 是否保存结果
MODELDEPLOY_CAPI_EXPORT void md_draw_attr_result(const MDImage* image,
                                                 const MDAttributeResults* c_results,
                                                 double threshold,
                                                 MDMapData* c_label_map,
                                                 const char* font_path,
                                                 int font_size,
                                                 double alpha,
                                                 int save_result);

/// 释放文本识别结果
/// @param c_results 模型识别结果，通过md_ocr_model_predict方法手动开辟了内存空间，需要手动释放
MODELDEPLOY_CAPI_EXPORT void md_free_attr_result(MDAttributeResults* c_results);

/// 释放模型
/// @param model 由create_ocr_model生成
MODELDEPLOY_CAPI_EXPORT void md_free_attr_model(MDModel* model);

#ifdef __cplusplus
}
#endif
