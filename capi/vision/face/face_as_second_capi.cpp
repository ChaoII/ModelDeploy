//
// Created by aichao on 2025-4-8.
//


#include "csrc/vision.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/face/face_as_second_capi.h"


MDStatusCode md_create_face_as_second_model(MDModel* model, const char* model_path,
                                            const MDRuntimeOption* option) {
    modeldeploy::RuntimeOption _option;
    c_runtime_option_2_runtime_option(option, &_option);
    const auto face_as_second_model = new modeldeploy::vision::face::SeetaFaceAsSecond(model_path, _option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(face_as_second_model->name().c_str());
    model->model_content = face_as_second_model;
    model->type = MDModelType::FACE;
    if (!face_as_second_model->is_initialized()) {
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}


MDStatusCode md_face_as_second_predict(const MDModel* model, MDImage* image, MDFaceAsSecondResults* c_results) {
    if (model->type != MDModelType::FACE) {
        return MDStatusCode::ModelTypeError;
    }
    auto cv_image = md_image_to_mat(image);
    const auto face_as_second_model = static_cast<
        modeldeploy::vision::face::SeetaFaceAsSecond*>(model->model_content);

    std::vector<std::tuple<int, float>> result;
    if (const bool res_status = face_as_second_model->predict(cv_image, &result); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }

    c_results->size = static_cast<int>(result.size());
    c_results->data = new MDFaceAsSecondResult[result.size()];
    for (int i = 0; i < result.size(); ++i) {
        c_results->data[i] = {std::get<0>(result[i]), std::get<1>(result[i])};
    }
    return MDStatusCode::Success;
}

void md_free_face_as_second_result(MDFaceAsSecondResults* c_results) {
    if (c_results->size != 0 && c_results->data != nullptr) {
        delete[] c_results->data;
        c_results->data = nullptr;
        c_results->size = 0;
    }
}

void md_free_face_as_second_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::face::SeetaFaceAsSecond*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
