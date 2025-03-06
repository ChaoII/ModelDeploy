//
// Created by aichao on 2025/2/26.
//

#pragma once
#include "capi/common/md_decl.h"
#include "capi/common/md_types.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 创建Kokoro模型
 *
 * 该函数用于初始化并创建一个Kokoro模型，Kokoro模型是一种特定的模型类型。
 * 创建模型时，需要提供模型的参数，这些参数将影响模型的行为和性能。
 *
 * @param model 模型的指针，用于接收创建的模型实例
 * @param parameters 模型参数的指针，包含模型初始化所需的各种参数
 * @return MDStatusCode 表示模型创建操作的状态，0表示成功，非零值表示失败
 */
MODELDEPLOY_CAPI_EXPORT
MDStatusCode md_create_kokoro_model(MDModel* model, const MDKokoroParameters* parameters);

/**
 * @brief 使用Kokoro模型进行预测
 *
 * 该函数使用Kokoro模型对给定的文本进行预测，预测结果以音频文件的形式输出。
 * 预测时可以调整语音的速度，以适应不同的需求。
 *
 * @param model 模型的指针，表示要使用的Kokoro模型实例
 * @param text 需要进行预测的文本
 * @param sid 语音的标识符，用于选择特定的语音风格或说话人
 * @param speed 语音速度的调整值，允许对输出语音的速度进行微调
 * @param wav_path 预测结果的文件路径，预测完成后将生成一个音频文件
 * @return MDStatusCode 表示预测操作的状态，0表示成功，非零值表示失败
 */
MODELDEPLOY_CAPI_EXPORT
MDStatusCode md_kokoro_model_predict(const MDModel* model, const char* text, int sid, float speed,
                                     const char* wav_path);

/**
 * @brief 释放Kokoro模型资源
 *
 * 该函数用于释放Kokoro模型实例所占用的资源，包括内存和其他可能的资源。
 * 在不再需要模型实例时，调用该函数可以有效地管理资源。
 *
 * @param model 模型的指针，表示需要释放的Kokoro模型实例
 */
MODELDEPLOY_CAPI_EXPORT
void md_free_kokoro_model(MDModel* model);

#ifdef __cplusplus
}
#endif
