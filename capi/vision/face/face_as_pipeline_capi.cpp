//
// Created by aichao on 2025-4-8.
//


#include "csrc/vision.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/face/face_as_pipeline_capi.h"


MDStatusCode md_create_face_as_pipeline_model(MDModel* model,
                                              const char* face_det_model_file,
                                              const char* first_model_file,
                                              const char* second_model_file,
                                              const MDRuntimeOption* option) {
    modeldeploy::RuntimeOption _option;
    c_runtime_option_2_runtime_option(option, &_option);
    const auto face_as_pipeline_model = new modeldeploy::vision::face::SeetaFaceAsPipeline(
        face_det_model_file, first_model_file, second_model_file, _option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(face_as_pipeline_model->name().c_str());
    model->model_content = face_as_pipeline_model;
    model->type = MDModelType::FACE;
    if (!face_as_pipeline_model->is_initialized()) {
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}


MDStatusCode md_face_as_pipeline_predict(const MDModel* model, MDImage* image, MDFaceAsResults* c_results,
                                         const float fuse_threshold, const float clarity_threshold) {
    if (model->type != MDModelType::FACE) {
        return MDStatusCode::ModelTypeError;
    }
    auto image_data = md_image_to_image_data(image);
    const auto face_as_pipeline_model = static_cast<
        modeldeploy::vision::face::SeetaFaceAsPipeline*>(model->model_content);

    std::vector<FaceAntiSpoofResult> results;
    if (const bool res_status = face_as_pipeline_model->predict(
        image_data, &results, fuse_threshold, clarity_threshold); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    const size_t face_size = results.size();
    c_results->size = static_cast<int>(face_size);
    c_results->data = new MDFaceAsResult[face_size];
    for (int i = 0; i < face_size; ++i) {
        c_results->data[i] = static_cast<MDFaceAsResult>(results[i]);
    }
    return MDStatusCode::Success;
}

void md_free_face_as_pipeline_result(MDFaceAsResults* c_results) {
    if (c_results->size != 0 && c_results->data != nullptr) {
        delete[] c_results->data;
        c_results->data = nullptr;
        c_results->size = 0;
    }
}

void md_free_face_as_pipeline_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::face::SeetaFaceAsPipeline*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
