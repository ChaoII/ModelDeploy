//
// Created by AC on 2024-12-17.
//


#include <map>
#include "csrc/vision.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/face/face_age_capi.h"


MDStatusCode md_create_face_age_model(MDModel* model, const char* model_path,
                                      const int thread_num) {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(thread_num);
    const auto face_age_model = new modeldeploy::vision::face::SeetaFaceAge(model_path, option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(face_age_model->name().c_str());
    model->model_content = face_age_model;
    model->type = MDModelType::FACE;
    if (!face_age_model->is_initialized()) {
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}


MDStatusCode md_face_age_predict(const MDModel* model, MDImage* image, MDFaceAgeResult* c_result) {
    if (model->type != MDModelType::Detection) {
        return MDStatusCode::ModelTypeError;
    }
    auto cv_image = md_image_to_mat(image);
    const auto face_age_model = static_cast<modeldeploy::vision::face::SeetaFaceAge*>(model->model_content);
    if (const bool res_status = face_age_model->predict(cv_image, c_result); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    return MDStatusCode::Success;
}


void md_free_face_age_result(MDFaceAgeResult* c_result) {

}

void md_free_face_age_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::face::SeetaFaceAge*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
