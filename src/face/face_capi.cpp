//
// Created by AC on 2024-12-25.
//

#include "face_capi.h"
#include "face_model.h"
#include "../utils_internal.h"


MDStatusCode md_create_face_model(MDModel *model, const char *model_dir, int flag, int thread_num) {
    auto face_model = new FaceModel(model_dir, flag, thread_num);
    model->type = MDModelType::FACE;
    model->format = MDModelFormat::Tennis;
    model->model_content = face_model;
    std::string model_name = "FaceNet";
    model->model_name = (char *) malloc((model_name.size() + 1) * sizeof(char));
    memcpy(model->model_name, model_name.c_str(), model_name.size() + 1);
    return MDStatusCode::Success;
}


MDStatusCode md_extract_feature(MDModel *model, MDImage *image, MDFaceFeature *md_feature) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    auto face_model = static_cast<FaceModel *>(model->model_content);
    auto seeta_image = md_image_to_seeta_image(image);
    auto faces = face_model->face_detection(seeta_image);
    if (faces.empty()) {
        return MDStatusCode::NotFoundFace;
    }
    auto points = face_model->face_marker(seeta_image, faces[0].pos);
    auto features = face_model->extract_feature(seeta_image, points);
    md_feature->size = features.size();
    md_feature->data = (float *) malloc(features.size() * sizeof(float));
    memcpy(md_feature->data, features.data(), features.size() * sizeof(float));
    return MDStatusCode::Success;
}

MDStatusCode md_face_anti_spoofing(MDModel *model, MDImage *image, MDFaceAntiSpoofingResult *result) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    auto face_model = static_cast<FaceModel *>(model->model_content);
    auto seeta_image = md_image_to_seeta_image(image);
    auto faces = face_model->face_detection(seeta_image);
    if (faces.empty()) {
        return MDStatusCode::NotFoundFace;
    }
    auto points = face_model->face_marker(seeta_image, faces[0].pos);
    *result = (MDFaceAntiSpoofingResult) face_model->face_anti_spoofing(seeta_image, faces[0].pos, points);
    return MDStatusCode::Success;
}


MDStatusCode
md_quality_evaluate(MDModel *model, MDImage *image, MDFaceQualityEvaluateType type, MDFACEQualityEvaluateRule *result) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    auto face_model = static_cast<FaceModel *>(model->model_content);
    auto seeta_image = md_image_to_seeta_image(image);
    auto faces = face_model->face_detection(seeta_image);
    if (faces.empty()) {
        return MDStatusCode::NotFoundFace;
    }
    auto points = face_model->face_marker(seeta_image, faces[0].pos);
    auto r = face_model->quality_evaluate(seeta_image, faces[0].pos, points, FaceModel::QualityEvaluateType(type));
    *result = (MDFACEQualityEvaluateRule) r.level;
    return MDStatusCode::Success;
}

MDStatusCode md_age_predict(MDModel *model, MDImage *image, int *age) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    auto face_model = static_cast<FaceModel *>(model->model_content);
    auto seeta_image = md_image_to_seeta_image(image);
    auto faces = face_model->face_detection(seeta_image);
    if (faces.empty()) {
        return MDStatusCode::NotFoundFace;
    }
    auto points = face_model->face_marker(seeta_image, faces[0].pos);
    *age = face_model->age_predict(seeta_image, points);
    return Success;
}

MDStatusCode md_gender_predict(MDModel *model, MDImage *image, MDGenderResult *gender) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    auto face_model = static_cast<FaceModel *>(model->model_content);
    auto seeta_image = md_image_to_seeta_image(image);
    auto faces = face_model->face_detection(seeta_image);
    if (faces.empty()) {
        return MDStatusCode::NotFoundFace;
    }
    auto points = face_model->face_marker(seeta_image, faces[0].pos);
    *gender = (MDGenderResult) face_model->gender_predict(seeta_image, points);
    return Success;
}

MDStatusCode md_eye_state_predict(MDModel *model, MDImage *image, MDEyeStateResult *eye_state) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    auto face_model = static_cast<FaceModel *>(model->model_content);
    auto seeta_image = md_image_to_seeta_image(image);
    auto faces = face_model->face_detection(seeta_image);
    if (faces.empty()) {
        return MDStatusCode::NotFoundFace;
    }
    auto points = face_model->face_marker(seeta_image, faces[0].pos);
    auto eye_state_result = face_model->eye_state_predict(seeta_image, points);
    eye_state->left_eye = (MDEyeState) eye_state_result.first;
    eye_state->right_eye = (MDEyeState) eye_state_result.second;
    return Success;
}






