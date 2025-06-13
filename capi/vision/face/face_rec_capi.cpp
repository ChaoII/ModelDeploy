//
// Created by aichao on 2025-4-7.
//


#include "csrc/vision.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/face/face_rec_capi.h"

#include <csrc/vision/common/display/display.h>


MDStatusCode md_create_face_rec_model(MDModel* model, const char* model_path,
                                      const MDRuntimeOption* option) {
    modeldeploy::RuntimeOption _option;
    c_runtime_option_2_runtime_option(option, &_option);
    const auto face_rec_model = new modeldeploy::vision::face::SeetaFaceID(model_path, _option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(face_rec_model->name().c_str());
    model->model_content = face_rec_model;
    model->type = MDModelType::FACE;
    if (!face_rec_model->is_initialized()) {
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}


MDStatusCode md_face_rec_predict(const MDModel* model, MDImage* image, MDFaceRecognizerResult* c_result) {
    if (model->type != MDModelType::FACE) {
        return MDStatusCode::ModelTypeError;
    }
    const auto cv_image = md_image_to_mat(image);
    modeldeploy::vision::FaceRecognitionResult result;
    const auto face_rec_model = static_cast<modeldeploy::vision::face::SeetaFaceID*>(model->model_content);
    if (const bool res_status = face_rec_model->predict(cv_image, &result); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    face_recognizer_result_2_c_result(result, c_result);
    return MDStatusCode::Success;
}


void md_print_face_rec_result(const MDFaceRecognizerResult* c_result) {
    std::vector<modeldeploy::vision::FaceRecognitionResult> result(1);
    c_result_2_face_recognizer_result(c_result, &result[0]);
    dis_face_rec(result);
}


void md_free_face_rec_result(MDFaceRecognizerResult* c_result) {
    if (c_result->size > 0 && c_result->embedding != nullptr) {
        delete [] c_result->embedding;
        c_result->embedding = nullptr;
    }
}

void md_free_face_rec_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::face::SeetaFaceID*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
