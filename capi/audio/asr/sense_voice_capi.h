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
 * 创建SenseVoice模型
 *
 * @param model 模型指针的引用，用于存储创建的模型对象
 * @param parameters SenseVoice模型的参数，配置模型的行为
 * @return 返回状态码，表示模型创建是否成功
 *
 * 此函数用于初始化并创建一个SenseVoice模型，根据提供的参数进行配置
 * 它是模型部署流程中的关键步骤，负责生成一个可用的模型实例
 */
MODELDEPLOY_CAPI_EXPORT
MDStatusCode md_create_sense_voice_model(MDModel* model, MDSenseVoiceParameters* parameters);


/**
 * 使用SenseVoice模型进行预测的函数。
 * 
 * 该函数接受一个模型实例、一个wav格式音频文件的路径，并返回一个自动语音识别(ASR)结果。
 * 它主要用于通过预训练的语音模型来识别音频内容。
 * 
 * @param model 指向MDModel类型的指针，代表了要使用的模型实例。
 * @param wav_path 指向const char类型的指针，包含了wav文件的路径。
 * @param asr_result 指向MDASRResult类型的指针，用于存储函数执行后的ASR结果。
 * 
 * @return 返回MDStatusCode类型，表示函数执行的状态码，用于指示函数是否成功执行。
 */
MODELDEPLOY_CAPI_EXPORT
MDStatusCode md_sense_voice_model_predict(const MDModel* model, const char* wav_path, MDASRResult* asr_result);


/**
 * 释放语音识别结果内存
 * 
 * 当使用md_sense_voice函数处理语音数据并获得识别结果后，通过本函数释放相关内存资源
 * 这是必要的资源管理措施，特别是在长时间运行或频繁调用识别功能的应用中
 * 
 * @param asr_result 指向MDASRResult结构体的指针，代表语音识别结果
 */
MODELDEPLOY_CAPI_EXPORT
void md_free_sense_voice_result(MDASRResult* asr_result);

/**
* 释放SenseVoice模型的内存
* 该函数用于释放之前分配给SenseVoice模型的资源
*
* @param model: 指向要释放的模型的指针
*/
MODELDEPLOY_CAPI_EXPORT
void md_free_sense_voice_model(MDModel* model);

#ifdef __cplusplus
}
#endif
