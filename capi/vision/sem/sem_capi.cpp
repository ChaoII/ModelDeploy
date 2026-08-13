//
// Created by aichao on 2026/8/13.
//

#include "csrc/vision.h"
#include "csrc/core/md_log.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/sem/sem_capi.h"

#include <cstring>

MDStatusCode md_create_sem_model(MDModel* model, const char* model_path,
                                 const MDRuntimeOption* option) {
    modeldeploy::RuntimeOption _option;
    c_runtime_option_2_runtime_option(option, &_option);
    const auto sem_model = new modeldeploy::vision::detection::UltralyticsSem(model_path, _option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(sem_model->name().c_str());
    model->model_content = sem_model;
    model->type = MDModelType::SemSeg;
    if (!sem_model->is_initialized()) {
        MD_LOG_ERROR << "Semantic segmentation model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDStatusCode md_set_sem_input_size(const MDModel* model, const MDSize size) {
    if (model->type != MDModelType::SemSeg) {
        MD_LOG_ERROR << "Model type is not sem!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto sem_model =
        static_cast<modeldeploy::vision::detection::UltralyticsSem*>(model->model_content);
    sem_model->get_preprocessor().set_size({size.width, size.height});
    return MDStatusCode::Success;
}

MDStatusCode md_sem_predict(const MDModel* model, MDImage* image, MDSemSegResult* c_result) {
    if (model->type != MDModelType::SemSeg) {
        MD_LOG_ERROR << "Model type is not sem!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto image_data = md_image_to_image_data(image);
    modeldeploy::vision::SemSegResult result;
    const auto sem_model =
        static_cast<modeldeploy::vision::detection::UltralyticsSem*>(model->model_content);
    if (!sem_model->predict(image_data, &result)) {
        return MDStatusCode::ModelPredictFailed;
    }
    c_result->num_classes = result.num_classes;
    c_result->shape_size = static_cast<int>(result.shape.size());
    c_result->shape = new int[c_result->shape_size];
    for (int i = 0; i < c_result->shape_size; ++i) {
        c_result->shape[i] = static_cast<int>(result.shape[i]);
    }
    const size_t num = result.labels.size();
    c_result->labels = new unsigned char[num];
    std::memcpy(c_result->labels, result.labels.data(), num);
    return MDStatusCode::Success;
}

void md_free_sem_result(MDSemSegResult* c_result) {
    if (c_result->labels != nullptr) {
        delete[] c_result->labels;
        c_result->labels = nullptr;
    }
    if (c_result->shape != nullptr) {
        delete[] c_result->shape;
        c_result->shape = nullptr;
    }
    c_result->shape_size = 0;
    c_result->num_classes = 0;
}

void md_free_sem_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::detection::UltralyticsSem*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}

MDStatusCode md_sem_predict_nv12(
    const MDModel* model,
    const unsigned char* src_y, const unsigned char* src_uv,
    int width, int height, int step_y, int step_uv,
    MDDevice src_device,
    MDSemSegResult* c_result) {
    if (model->type != MDModelType::SemSeg) {
        MD_LOG_ERROR << "Model type is not SemSeg!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto model_ptr = static_cast<modeldeploy::vision::detection::UltralyticsSem*>(model->model_content);
    modeldeploy::Device dev = modeldeploy::Device::CPU;
    switch (src_device) {
    case MD_DEVICE_GPU: dev = modeldeploy::Device::GPU; break;
    case MD_DEVICE_TPU: dev = modeldeploy::Device::TPU; break;
    default: dev = modeldeploy::Device::CPU; break;
    }
    modeldeploy::vision::LetterBoxRecord lbr{};
        modeldeploy::vision::SemSegResult result;
    if (!model_ptr->predict_nv12(src_y, src_uv, width, height, step_y, step_uv,
                                 &result, &lbr, dev, nullptr)) {
        return MDStatusCode::ModelPredictFailed;
    }
    c_result->num_classes = result.num_classes;
    c_result->shape_size = static_cast<int>(result.shape.size());
    c_result->shape = new int[c_result->shape_size];
    for (int i = 0; i < c_result->shape_size; ++i) {
        c_result->shape[i] = static_cast<int>(result.shape[i]);
    }
    const size_t num = result.labels.size();
    c_result->labels = new unsigned char[num];
    std::memcpy(c_result->labels, result.labels.data(), num);
    return MDStatusCode::Success;
}
