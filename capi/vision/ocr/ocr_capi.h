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
/// \param option 和runtime相关的参数，比如GPU推理，fp16半精度推理等
/// \return 成功时返回MDStatusCode::Success，其他为失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_create_ocr_model(
    MDModel* model, const MDOCRModelParameters* parameters, const MDRuntimeOption* option);
///
/// @param model 在 create_ocr_model方法中创建的模型句柄
/// @param max_side_len db detection 模型中图像最长边长度，对于分辨率较大的图像建议设置稍大一点，但会耗部分资源，默认值为960
/// @return 成功时返回MDStatusCode::Success，其他为失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_ocr_det_set_max_side_len(const MDModel* model, int max_side_len = 960);
///
/// @param model 在 create_ocr_model方法中创建的模型句柄
/// @param db_thresh db detection 二值化阈值 【太低 → 背景噪声进来，容易把不相干的区域粘连；太高 → 检测框容易断裂，文字被切碎。】默认值为0.3
/// @return 成功时返回MDStatusCode::Success，其他为失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_ocr_det_set_db_thresh(const MDModel* model, double db_thresh = 0.3);
///
/// @param model 在 create_ocr_model方法中创建的模型句柄
/// @param db_box_thresh 检测框得分阈值，默认值为0.5，太低 → 会保留很多假阳性框； 太高 → 只保留很高质量的框，容易漏检边缘模糊的文字。
/// @return 成功时返回MDStatusCode::Success，其他为失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_ocr_det_set_db_box_thresh(const MDModel* model, double db_box_thresh = 0.6);
///
/// @param model 在 create_ocr_model方法中创建的模型句柄
/// @param db_unclip_ratio 把检测框稍微膨胀，DB 网络预测的是文字区域的收缩边界（shrinked polygon），
/// 收缩是训练时为了避免不同文字粘连。 后处理时需要 反收缩（unclip），把框放大回到原始大小。常见 1.5 ~ 2.0，太小 → 框紧贴文字，容易裁掉部分笔画；
/// 太大 → 框太松，多个相邻文字容易粘连。
/// @return 成功时返回MDStatusCode::Success，其他为失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_ocr_det_db_unclip_ratio(const MDModel* model, double db_unclip_ratio = 1.5);
///
/// @param model 在 create_ocr_model方法中创建的模型句柄
/// @param use_dilation 是否开启形态学膨胀，让文字区域更加连贯，避免笔画间出现断裂的小洞。 尤其对细字体 低分辨率文字有帮助。
/// 影响： 开启后，容易把断开的文字粘起来（利于形成完整连通域）； 但过度会让本来距离较远的两个文字也粘连成一个框。默认不开启0.
/// @return 成功时返回MDStatusCode::Success，其他为失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_ocr_det_db_set_use_dilation(const MDModel* model, int use_dilation = 0);

/// 文本识别
/// \param model 在 create_ocr_model方法中创建的模型句柄
/// \param image 图像数据
/// \param c_results MDOCRResults结果，包含了OCR识别的坐标文本和置信度，可以通过print_ocr_result方法打印出来
/// \return 成功时返回MDStatusCode::Success，其他为失败
MODELDEPLOY_CAPI_EXPORT MDStatusCode
md_ocr_model_predict(const MDModel* model, MDImage* image, MDOCRResults* c_results);

/// 获取文本所在位置
/// \param model 模型句柄，由create_ocr_model生成
/// \param image 原始图像
/// \param text 文本
/// \return 返回文本所在位置[x,y,w,h]
MODELDEPLOY_CAPI_EXPORT MDRect md_get_text_position(const MDModel* model, MDImage* image, const char* text);

/// 打印文本识别结果
MODELDEPLOY_CAPI_EXPORT void md_print_ocr_result(const MDOCRResults* c_results);

/// 绘制文本识别结果
/// \param image 原始图像
/// \param c_results ocr识别结果
/// \param font_path 字体路径
/// \param font_size 字体大小
/// \param alpha 不透明度
/// \param save_result 是否保存结果
MODELDEPLOY_CAPI_EXPORT void md_draw_ocr_result(const MDImage* image, const MDOCRResults* c_results,
                                                const char* font_path,
                                                int font_size, double alpha, int save_result);

/// 释放文本识别结果
/// @param c_results 模型识别结果，通过md_ocr_model_predict方法手动开辟了内存空间，需要手动释放
MODELDEPLOY_CAPI_EXPORT void md_free_ocr_result(MDOCRResults* c_results);

/// 释放模型
/// @param model 由create_ocr_model生成
MODELDEPLOY_CAPI_EXPORT void md_free_ocr_model(MDModel* model);

#ifdef __cplusplus
}
#endif
