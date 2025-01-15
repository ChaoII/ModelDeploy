//
// Created by AC on 2025-01-09.
//

#pragma once

#include "../decl.h"
#include "../types.h"

#ifdef __cplusplus
extern "C" {
#endif


EXPORT_DECL MDStatusCode md_create_two_pass_model(MDModel* model,
                                                  const char* offline_model_dir,
                                                  const char* online_model_dir,
                                                  const char* vad_dir,
                                                  const char* punct_dir,
                                                  const char* itn_dir,
                                                  const char* lm_dir,
                                                  const char* hot_word_path,
                                                  ASRMode asr_mode,
                                                  float global_beam = 3.0f,
                                                  float lattice_beam = 3.0f,
                                                  float am_scale = 10.0f,
                                                  int fst_inc_wts = 20,
                                                  int thread_num = 1);

EXPORT_DECL MDStatusCode md_two_pass_model_predict(MDModel* model, const char* wav_path,
                                                   MDASRResult* asr_result, int audio_fs = 16000);


EXPORT_DECL MDStatusCode md_two_pass_model_predict_buffer(const MDModel* model, const char* speech_buff,
                                                          int step,
                                                          bool is_final,
                                                          const char* wav_format,
                                                          int audio_fs,
                                                          ASRCallBack asr_callback);


EXPORT_DECL void md_free_two_pass_result(MDASRResult* asr_result);

EXPORT_DECL void md_free_two_pass_model(MDModel* model);


#ifdef __cplusplus
}
#endif
