//
// Created by aichao on 2026/8/13.
//

#include "csrc/vision.h"
#include "csrc/core/md_log.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/depth/depth_capi.h"

MDStatusCode md_create_depth_model(MDModel* model, const char* model_path,
                                   const MDRuntimeOption* option) {
    modeldeploy::RuntimeOption _option;
    c_runtime_option_2_runtime_option(option, &_option);
    const auto depth_model = new modeldeploy::vision::detection::UltralyticsDepth(model_path, _option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(depth_model->name().c_str());
    model->model_content = depth_model;
    model->type = MDModelType::Depth;
    if (!depth_model->is_initialized()) {
        MD_LOG_ERROR << "Depth estimation model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDStatusCode md_set_depth_input_size(const MDModel* model, const MDSize size) {
    if (model->type != MDModelType::Depth) {
        MD_LOG_ERROR << "Model type is not depth!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto depth_model =
        static_cast<modeldeploy::vision::detection::UltralyticsDepth*>(model->model_content);
    depth_model->get_preprocessor().set_size({size.width, size.height});
    return MDStatusCode::Success;
}

MDStatusCode md_depth_predict(const MDModel* model, MDImage* image, MDDepthResult* c_result) {
    if (model->type != MDModelType::Depth) {
        MD_LOG_ERROR << "Model type is not depth!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto image_data = md_image_to_image_data(image);
    modeldeploy::vision::DepthResult result;
    const auto depth_model =
        static_cast<modeldeploy::vision::detection::UltralyticsDepth*>(model->model_content);
    if (!depth_model->predict(image_data, &result)) {
        return MDStatusCode::ModelPredictFailed;
    }
    c_result->shape_size = static_cast<int>(result.shape.size());
    c_result->shape = new int[c_result->shape_size];
    for (int i = 0; i < c_result->shape_size; ++i) {
        c_result->shape[i] = static_cast<int>(result.shape[i]);
    }
    const size_t num = result.depth.size();
    c_result->depth = new float[num];
    std::memcpy(c_result->depth, result.depth.data(), num * sizeof(float));
    return MDStatusCode::Success;
}

void md_free_depth_result(MDDepthResult* c_result) {
    if (c_result->depth != nullptr) {
        delete[] c_result->depth;
        c_result->depth = nullptr;
    }
    if (c_result->shape != nullptr) {
        delete[] c_result->shape;
        c_result->shape = nullptr;
    }
    c_result->shape_size = 0;
}

void md_free_depth_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::detection::UltralyticsDepth*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
