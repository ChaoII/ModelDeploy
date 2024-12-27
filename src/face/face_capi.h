//
// Created by AC on 2024-12-25.
//



#pragma once

#include "../decl.h"
#include "../types.h"


#ifdef __cplusplus
extern "C" {
#endif


EXPORT_DECL MDStatusCode
md_create_face_model(MDModel *model, const char *model_dir, int flags = MD_MASK, int thread_num = 1);


EXPORT_DECL MDStatusCode md_extract_feature(MDModel *model, MDImage *image, MDFaceFeature *feature);


EXPORT_DECL MDStatusCode md_face_anti_spoofing(MDModel *model, MDImage *image, MDFaceAntiSpoofingResult *result);


EXPORT_DECL MDStatusCode
md_quality_evaluate(MDModel *model, MDImage *image, MDFaceQualityEvaluateType type, MDFACEQualityEvaluateRule *result);

EXPORT_DECL MDStatusCode md_age_predict(MDModel *model, MDImage *image, int *age);

EXPORT_DECL MDStatusCode md_gender_predict(MDModel *model, MDImage *image, MDGenderResult *gender);

EXPORT_DECL MDStatusCode md_eye_state_predict(MDModel *model, MDImage *image, MDEyeStateResult *eye_state);

#ifdef __cplusplus
}
#endif




