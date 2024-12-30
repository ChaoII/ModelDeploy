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

EXPORT_DECL MDStatusCode md_face_detection(const MDModel *model, MDImage *image, MDDetectionResults *result);

EXPORT_DECL MDStatusCode
md_face_marker(const MDModel *model, MDImage *image, const MDRect *rect, MDLandMarkResult *result);

EXPORT_DECL MDStatusCode md_face_feature(const MDModel *model, MDImage *image,
                                         const MDLandMarkResult *points, MDFaceFeature *feature);

EXPORT_DECL MDStatusCode md_face_feature_e2e(const MDModel *model, MDImage *image, MDFaceFeature *feature);

EXPORT_DECL MDStatusCode md_face_feature_compare(const MDModel *model, const MDFaceFeature *feature1,
                                                 const MDFaceFeature *feature2, float *similarity);

EXPORT_DECL MDStatusCode md_face_anti_spoofing(const MDModel *model, MDImage *image, MDFaceAntiSpoofingResult *result);


EXPORT_DECL MDStatusCode md_face_quality_evaluate(const MDModel *model, MDImage *image,
                                                  MDFaceQualityEvaluateType type, MDFaceQualityEvaluateResult *result);

EXPORT_DECL MDStatusCode md_face_age_predict(const MDModel *model, MDImage *image, int *age);

EXPORT_DECL MDStatusCode md_face_gender_predict(const MDModel *model, MDImage *image, MDGenderResult *gender);

EXPORT_DECL MDStatusCode md_face_eye_state_predict(const MDModel *model, MDImage *image, MDEyeStateResult *eye_state);

EXPORT_DECL void md_free_face_landmark(MDLandMarkResult *result);

EXPORT_DECL void md_free_face_feature(MDFaceFeature *feature);

EXPORT_DECL void md_print_face_anti_spoofing_result(MDFaceAntiSpoofingResult result);


#ifdef __cplusplus
}
#endif




