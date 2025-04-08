//
// Created by aichao on 2025-4-7.
//


#include "csrc/vision.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/face/face_gender_capi.h"


MDStatusCode md_create_face_gender_model(MDModel* model, const char* model_path,
                                         const int thread_num) {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(thread_num);
    const auto face_gender_model = new modeldeploy::vision::face::SeetaFaceGender(model_path, option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(face_gender_model->name().c_str());
    model->model_content = face_gender_model;
    model->type = MDModelType::FACE;
    if (!face_gender_model->is_initialized()) {
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}


MDStatusCode md_face_gender_predict(const MDModel* model, MDImage* image, MDFaceGenderResult* c_result) {
    if (model->type != MDModelType::FACE) {
        return MDStatusCode::ModelTypeError;
    }
    const auto cv_image = md_image_to_mat(image);
    const auto face_gender_model = static_cast<modeldeploy::vision::face::SeetaFaceGender*>(model->model_content);
    int gender_id;
    if (const bool res_status = face_gender_model->predict(cv_image, &gender_id); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    *c_result = static_cast<MDFaceGenderResult>(gender_id);
    return MDStatusCode::Success;
}


void md_free_face_gender_result(MDFaceGenderResult* c_result) {
}

void md_free_face_gender_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::face::SeetaFaceGender*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
