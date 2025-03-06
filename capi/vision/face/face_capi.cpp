//
// Created by AC on 2024-12-25.
//

#include "capi/vision/face/face_capi.h"
#include "capi/utils/internal/utils.h"
#include "csrc/vision/face/face_model.h"


MDStatusCode md_create_face_model(MDModel* model, const char* model_dir, const int flags, const int thread_num) {
    const auto face_model = new FaceModel(model_dir, flags, thread_num);
    model->type = MDModelType::FACE;
    model->format = MDModelFormat::Tennis;
    model->model_content = face_model;
    model->model_name = strdup("SeetaFace");
    return MDStatusCode::Success;
}

MDStatusCode md_face_detection(const MDModel* model, MDImage* image, MDDetectionResults* result) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    const auto face_model = static_cast<FaceModel*>(model->model_content);
    const auto seeta_image = md_image_to_seeta_image(image);
    if (!face_model->check_flag(MD_FACE_DETECT)) {
        return MDStatusCode::FaceDetectionFlagNotSetError;
    }
    const auto faces = face_model->face_detection(seeta_image);
    if (faces.empty()) {
        return MDStatusCode::NotFoundFace;
    }
    result->size = static_cast<int>(faces.size());
    result->data = static_cast<MDDetectionResult*>(malloc(faces.size() * sizeof(MDDetectionResult)));
    for (int i = 0; i < faces.size(); ++i) {
        result->data[i].box.x = faces[i].pos.x;
        result->data[i].box.y = faces[i].pos.y;
        result->data[i].box.width = faces[i].pos.width;
        result->data[i].box.height = faces[i].pos.height;
        result->data[i].score = faces[i].score;
        result->data[i].label_id = -1;
    }
    return MDStatusCode::Success;
}

MDStatusCode md_face_marker(const MDModel* model, MDImage* image, const MDRect* rect, MDLandMarkResult* result) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    const auto face_model = static_cast<FaceModel*>(model->model_content);
    const auto seeta_image = md_image_to_seeta_image(image);
    if (!face_model->check_flag(MD_FACE_LANDMARK)) {
        return MDStatusCode::FaceLandmarkFlagNotSetError;
    }
    const auto points = face_model->face_marker(seeta_image, {rect->x, rect->y, rect->width, rect->height});
    if (points.empty()) {
        return MDStatusCode::NotFoundLandmark;
    }
    result->size = static_cast<int>(points.size());
    result->data = static_cast<MDPointF*>(malloc(points.size() * sizeof(MDPointF)));
    for (int i = 0; i < points.size(); ++i) {
        result->data[i].x = points[i].x;
        result->data[i].y = points[i].y;
    }
    return MDStatusCode::Success;
}

MDStatusCode
md_face_feature(const MDModel* model, MDImage* image, const MDLandMarkResult* points, MDFaceFeature* feature) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    const auto face_model = static_cast<FaceModel*>(model->model_content);
    if (!face_model->check_flag(MD_FACE_RECOGNITION)) {
        return MDStatusCode::FaceRecognitionFlagNotSetError;
    }
    const auto seeta_image = md_image_to_seeta_image(image);
    std::vector<SeetaPointF> seeta_points;
    seeta_points.reserve(points->size);
    for (int i = 0; i < points->size; ++i) {
        seeta_points.push_back({points->data[i].x, points->data[i].y});
    }
    const auto features = face_model->extract_feature(seeta_image, seeta_points);
    if (features.empty()) {
        return MDStatusCode::FaceFeatureExtractError;
    }
    feature->size = static_cast<int>(features.size());
    feature->data = static_cast<float*>(malloc(features.size() * sizeof(float)));
    memcpy(feature->data, features.data(), features.size() * sizeof(float));
    return MDStatusCode::Success;
}


MDStatusCode md_face_feature_e2e(const MDModel* model, MDImage* image, MDFaceFeature* feature) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    const auto face_model = static_cast<FaceModel*>(model->model_content);
    if (!face_model->check_flag(MD_FACE_DETECT)) {
        return MDStatusCode::FaceDetectionFlagNotSetError;
    }
    const auto seeta_image = md_image_to_seeta_image(image);
    const auto faces = face_model->face_detection(seeta_image);
    if (faces.empty()) {
        return MDStatusCode::NotFoundFace;
    }
    if (!face_model->check_flag(MD_FACE_LANDMARK)) {
        return MDStatusCode::FaceLandmarkFlagNotSetError;
    }
    const auto points = face_model->face_marker(seeta_image, faces[0].pos);
    if (!face_model->check_flag(MD_FACE_RECOGNITION)) {
        return MDStatusCode::FaceRecognitionFlagNotSetError;
    }
    const auto features = face_model->extract_feature(seeta_image, points);
    feature->size = static_cast<int>(features.size());
    feature->data = static_cast<float*>(malloc(features.size() * sizeof(float)));
    memcpy(feature->data, features.data(), features.size() * sizeof(float));
    return MDStatusCode::Success;
}

MDStatusCode md_face_feature_compare(const MDModel* model, const MDFaceFeature* feature1,
                                     const MDFaceFeature* feature2, float* similarity) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    const auto face_model = static_cast<FaceModel*>(model->model_content);
    if (!face_model->check_flag(MD_FACE_RECOGNITION)) {
        return MDStatusCode::FaceRecognitionFlagNotSetError;
    }
    const int size = face_model->get_feature_size();
    const auto feature1_vec = std::vector(feature1->data, feature1->data + size);
    const auto feature2_vec = std::vector(feature2->data, feature2->data + size);
    *similarity = face_model->face_feature_compare(feature1_vec, feature2_vec);
    return MDStatusCode::Success;
}

MDStatusCode md_face_anti_spoofing(const MDModel* model, MDImage* image, MDFaceAntiSpoofingResult* result) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    const auto face_model = static_cast<FaceModel*>(model->model_content);
    const auto seeta_image = md_image_to_seeta_image(image);
    if (!face_model->check_flag(MD_FACE_DETECT)) {
        return MDStatusCode::FaceDetectionFlagNotSetError;
    }
    const auto faces = face_model->face_detection(seeta_image);
    if (faces.empty()) {
        return MDStatusCode::NotFoundFace;
    }
    if (!face_model->check_flag(MD_FACE_LANDMARK)) {
        return MDStatusCode::FaceLandmarkFlagNotSetError;
    }
    const auto points = face_model->face_marker(seeta_image, faces[0].pos);
    if (!face_model->check_flag(MD_FACE_ANTI_SPOOfING)) {
        return MDStatusCode::FaceAntiSpoofingFlagNotSetError;
    }
    *result = static_cast<MDFaceAntiSpoofingResult>(face_model->face_anti_spoofing(seeta_image, faces[0].pos, points));
    return MDStatusCode::Success;
}

MDStatusCode md_face_quality_evaluate(const MDModel* model, MDImage* image,
                                      MDFaceQualityEvaluateType type, MDFaceQualityEvaluateResult* result) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    const auto face_model = static_cast<FaceModel*>(model->model_content);
    const auto seeta_image = md_image_to_seeta_image(image);
    if (!face_model->check_flag(MD_FACE_DETECT)) {
        return MDStatusCode::FaceDetectionFlagNotSetError;
    }
    const auto faces = face_model->face_detection(seeta_image);
    if (faces.empty()) {
        return MDStatusCode::NotFoundFace;
    }
    if (!face_model->check_flag(MD_FACE_LANDMARK)) {
        return MDStatusCode::FaceLandmarkFlagNotSetError;
    }
    const auto points = face_model->face_marker(seeta_image, faces[0].pos);
    if (points.empty()) {
        return MDStatusCode::NotFoundLandmark;
    }
    if (!face_model->check_flag(MD_FACE_QUALITY_EVALUATE)) {
        return MDStatusCode::FaceQualityEvaluateFlagNotSetError;
    }
    auto r = face_model->quality_evaluate(seeta_image, faces[0].pos, points,
                                          static_cast<FaceModel::QualityEvaluateType>(type));
    *result = static_cast<MDFaceQualityEvaluateResult>(r.level);
    return MDStatusCode::Success;
}

MDStatusCode md_face_age_predict(const MDModel* model, MDImage* image, int* age) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    const auto face_model = static_cast<FaceModel*>(model->model_content);
    const auto seeta_image = md_image_to_seeta_image(image);
    if (!face_model->check_flag(MD_FACE_DETECT)) {
        return MDStatusCode::FaceDetectionFlagNotSetError;
    }
    const auto faces = face_model->face_detection(seeta_image);
    if (faces.empty()) {
        return MDStatusCode::NotFoundFace;
    }
    if (!face_model->check_flag(MD_FACE_LANDMARK)) {
        return MDStatusCode::FaceLandmarkFlagNotSetError;
    }
    const auto points = face_model->face_marker(seeta_image, faces[0].pos);
    if (points.empty()) {
        return MDStatusCode::NotFoundLandmark;
    }
    if (!face_model->check_flag(MD_FACE_AGE_ATTRIBUTE)) {
        return MDStatusCode::FaceAgeAttributeFlagNotSetError;
    }
    *age = face_model->age_predict(seeta_image, points);
    return Success;
}

MDStatusCode md_face_gender_predict(const MDModel* model, MDImage* image, MDGenderResult* gender) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    const auto face_model = static_cast<FaceModel*>(model->model_content);
    const auto seeta_image = md_image_to_seeta_image(image);
    if (!face_model->check_flag(MD_FACE_DETECT)) {
        return MDStatusCode::FaceDetectionFlagNotSetError;
    }
    const auto faces = face_model->face_detection(seeta_image);
    if (faces.empty()) {
        return MDStatusCode::NotFoundFace;
    }
    if (!face_model->check_flag(MD_FACE_LANDMARK)) {
        return MDStatusCode::FaceLandmarkFlagNotSetError;
    }
    const auto points = face_model->face_marker(seeta_image, faces[0].pos);
    if (points.empty()) {
        return MDStatusCode::NotFoundLandmark;
    }
    if (!face_model->check_flag(MD_FACE_GENDER_ATTRIBUTE)) {
        return MDStatusCode::FaceGenderAttributeFlagNotSetError;
    }
    *gender = static_cast<MDGenderResult>(face_model->gender_predict(seeta_image, points));
    return Success;
}

MDStatusCode md_face_eye_state_predict(const MDModel* model, MDImage* image, MDEyeStateResult* eye_state) {
    if (model->type != MDModelType::FACE) return MDStatusCode::ModelTypeError;
    const auto face_model = static_cast<FaceModel*>(model->model_content);
    const auto seeta_image = md_image_to_seeta_image(image);
    if (!face_model->check_flag(MD_FACE_DETECT)) {
        return MDStatusCode::FaceDetectionFlagNotSetError;
    }
    const auto faces = face_model->face_detection(seeta_image);
    if (faces.empty()) {
        return MDStatusCode::NotFoundFace;
    }
    if (!face_model->check_flag(MD_FACE_LANDMARK)) {
        return MDStatusCode::FaceLandmarkFlagNotSetError;
    }
    const auto points = face_model->face_marker(seeta_image, faces[0].pos);
    if (points.empty()) {
        return MDStatusCode::NotFoundLandmark;
    }
    if (!face_model->check_flag(MD_FACE_EYE_STATE)) {
        return MDStatusCode::FaceEyeStateFlagNotSetError;
    }
    auto eye_state_result = face_model->eye_state_predict(seeta_image, points);
    eye_state->left_eye = static_cast<MDEyeState>(eye_state_result.first);
    eye_state->right_eye = static_cast<MDEyeState>(eye_state_result.second);
    return Success;
}

void md_print_face_anti_spoofing_result(const MDFaceAntiSpoofingResult result) {
    switch (result) {
    case MDFaceAntiSpoofingResult::REAL:
        std::cout << "real" << std::endl;
        break;
    case MDFaceAntiSpoofingResult::SPOOF:
        std::cout << "spoof" << std::endl;
        break;
    case MDFaceAntiSpoofingResult::FUZZY:
        std::cout << "fuzzy" << std::endl;
    case MDFaceAntiSpoofingResult::DETECTING:
        std::cout << "detecting" << std::endl;
        break;
    default:
        std::cout << "unknown" << std::endl;
    }
}

void md_free_face_landmark(MDLandMarkResult* result) {
    if (result != nullptr) {
        free(result->data);
        result->data = nullptr;
        result->size = 0;
    }
}

void md_free_face_feature(MDFaceFeature* feature) {
    free(feature->data);
    feature->data = nullptr;
    feature->size = 0;
}


void md_free_face_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<FaceModel*>(model->model_content);
        model->model_content = nullptr;
    }
    free(model->model_name);
    model->model_name = nullptr;
}
