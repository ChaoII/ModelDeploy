//
// Created by aichao on 2026/8/10.
//
#pragma once

#include "capi/common/md_decl.h"
#include "capi/common/md_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// SenseVoice ASR：predict 输入为 wav 文件路径，输出识别文本（MDASRResult.msg）
MODELDEPLOY_CAPI_EXPORT MDStatusCode
md_create_sense_voice_model(MDModel* model, const MDSenseVoiceParameters* asr_parameters,
                            const MDRuntimeOption* option);

MODELDEPLOY_CAPI_EXPORT MDStatusCode md_sense_voice_model_predict(const MDModel* model,
                                                                  const char* wav_path,
                                                                  MDASRResult* c_result,
                                                                  int audio_fs);

MODELDEPLOY_CAPI_EXPORT void md_free_sense_voice_result(MDASRResult* c_result);

MODELDEPLOY_CAPI_EXPORT void md_free_sense_voice_model(MDModel* model);

#ifdef __cplusplus
}
#endif
