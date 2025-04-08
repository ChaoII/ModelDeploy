//
// Created by aichao on 2025-4-8.
//


#include <map>
#include "csrc/vision.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/face/face_as_first_capi.h"


MDStatusCode md_create_face_as_first_model(MDModel* model, const char* model_path,
                                           const int thread_num) {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(thread_num);
    const auto face_as_first_model = new modeldeploy::vision::face::SeetaFaceAsFirst(model_path, option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(face_as_first_model->name().c_str());
    model->model_content = face_as_first_model;
    model->type = MDModelType::FACE;
    if (!face_as_first_model->is_initialized()) {
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}


MDStatusCode md_face_as_first_predict(const MDModel* model, MDImage* image, float* c_result) {
    if (model->type != MDModelType::FACE) {
        return MDStatusCode::ModelTypeError;
    }
    auto cv_image = md_image_to_mat(image);
    const auto face_as_first_model = static_cast<modeldeploy::vision::face::SeetaFaceAsFirst*>(model->
        model_content);
    if (const bool res_status = face_as_first_model->predict(cv_image, c_result); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    return MDStatusCode::Success;
}


void md_free_face_as_first_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::face::SeetaFaceAsFirst*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
