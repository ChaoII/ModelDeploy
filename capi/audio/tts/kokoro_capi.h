//
// Created by AC on 2024/12/16.
//
#pragma once

#include "capi/common/md_decl.h"
#include "capi/common/md_types.h"

#ifdef __cplusplus
extern "C" {
#endif


MODELDEPLOY_CAPI_EXPORT MDStatusCode
md_create_kokoro_model(MDModel* model, const MDKokoroParameters* kokoro_parameters);


MODELDEPLOY_CAPI_EXPORT MDStatusCode md_kokoro_model_predict(const MDModel* model, const char* input_text,
                                                             const char* voice,
                                                             float speed, MDTTSResult* c_results);


MODELDEPLOY_CAPI_EXPORT MDStatusCode md_write_wav(const MDTTSResult* c_results, const char* output_path);

MODELDEPLOY_CAPI_EXPORT void md_free_kokoro_result(MDTTSResult* c_results);


MODELDEPLOY_CAPI_EXPORT void md_free_kokoro_model(MDModel* model);

#ifdef __cplusplus
}


#endif
