//
// Created by AC on 2025-01-07.
//

#pragma once

#include "csrc/common/md_decl.h"
#include "csrc/common/md_types.h"

#ifdef __cplusplus
extern "C" {
#endif


EXPORT_DECL MDStatusCode md_create_asr_offline_model(MDModel* model,
                                                     const char* model_dir,
                                                     const char* vad_dir,
                                                     const char* punct_dir,
                                                     const char* itn_dir = "",
                                                     const char* lm_dir = "",
                                                     const char* hot_word_path = "",
                                                     bool blade_disc = true,
                                                     float global_beam = 3.0f,
                                                     float lattice_beam = 3.0f,
                                                     float am_scale = 10.0f,
                                                     int fst_inc_wts = 20,
                                                     int thread_num = 1,
                                                     int batch_size = 4,
                                                     bool use_gpu = false);

EXPORT_DECL MDStatusCode md_asr_offline_model_predict(MDModel* model, const char* wav_path,
                                                      MDASRResult* asr_result, int audio_fs = 16000);

EXPORT_DECL void md_free_asr_result(MDASRResult* asr_result);

EXPORT_DECL void md_free_asr_offline_model(MDModel* model);


#ifdef __cplusplus
}
#endif
