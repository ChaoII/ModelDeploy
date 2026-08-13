//
// Created by AC on 2024-12-17.
//

#include "csrc/vision.h"
#include "csrc/core/md_log.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/iseg/instance_seg_capi.h"

#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"

MDStatusCode md_create_instance_seg_model(MDModel* model, const char* model_path, const MDRuntimeOption* option) {
    modeldeploy::RuntimeOption _option;
    c_runtime_option_2_runtime_option(option, &_option);
    const auto instance_seg_model = new modeldeploy::vision::detection::UltralyticsSeg(model_path, _option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(instance_seg_model->name().c_str());
    model->model_content = instance_seg_model;
    model->type = MDModelType::InstanceSeg;
    if (!instance_seg_model->is_initialized()) {
        MD_LOG_ERROR << "Instance segmentation model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDStatusCode md_set_instance_seg_input_size(const MDModel* model, const MDSize size) {
    if (model->type != MDModelType::InstanceSeg) {
        MD_LOG_ERROR << "Model type is not instance_seg!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto instance_seg_model = static_cast<modeldeploy::vision::detection::UltralyticsSeg*>(model->model_content);
    instance_seg_model->get_preprocessor().set_size({size.width, size.height});
    return MDStatusCode::Success;
}

MDStatusCode md_instance_seg_predict(const MDModel* model, MDImage* image, MDIsegResults* c_results) {
    if (model->type != MDModelType::InstanceSeg) {
        MD_LOG_ERROR << "Model type is not instance_seg!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto image_data = md_image_to_image_data(image);
    std::vector<modeldeploy::vision::InstanceSegResult> results;
    const auto instance_seg_model = static_cast<modeldeploy::vision::detection::UltralyticsSeg*>(model->model_content);
    if (const bool res_status = instance_seg_model->predict(image_data, &results); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    iseg_results_2_c_results(results, c_results);
    return MDStatusCode::Success;
}


void md_print_instance_seg_result(const MDIsegResults* c_results) {
    std::vector<modeldeploy::vision::InstanceSegResult> results;
    c_results_2_iseg_results(c_results, &results);
    dis_iseg(results);
}


void md_draw_instance_seg_result(const MDImage* image, const MDIsegResults* c_results,
                                 const double threshold, const char* font_path, const int font_size,
                                 const double alpha, const int save_result) {
    auto image_data = md_image_to_image_data(image);
    std::vector<modeldeploy::vision::InstanceSegResult> results;
    c_results_2_iseg_results(c_results, &results);
    modeldeploy::vision::vis_iseg(image_data, results, threshold, font_path, font_size, alpha, save_result);
}

void md_free_instance_seg_result(MDIsegResults* c_results) {
    if (c_results->size > 0 && c_results->data != nullptr) {
        for (int i = 0; i < c_results->size; i++) {
            delete[] c_results->data[i].mask.buffer;
            delete[] c_results->data[i].mask.shape;
            c_results->data[i].mask.buffer = nullptr;
            c_results->data[i].mask.shape = nullptr;
            c_results->data[i].mask.buffer_size = 0;
            c_results->data[i].mask.num_dims = 0;
        }
        c_results->size = 0;
        delete[] c_results->data;
        c_results->data = nullptr;
    }
}

void md_free_instance_seg_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::detection::UltralyticsSeg*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}

MDStatusCode md_iseg_predict_nv12(
    const MDModel* model,
    const unsigned char* src_y, const unsigned char* src_uv,
    int width, int height, int step_y, int step_uv,
    MDDevice src_device,
    MDIsegResults* c_results) {
    if (model->type != MDModelType::InstanceSeg) {
        MD_LOG_ERROR << "Model type is not InstanceSeg!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto model_ptr = static_cast<modeldeploy::vision::detection::UltralyticsSeg*>(model->model_content);
    modeldeploy::Device dev = modeldeploy::Device::CPU;
    switch (src_device) {
    case MD_DEVICE_GPU: dev = modeldeploy::Device::GPU; break;
    case MD_DEVICE_TPU: dev = modeldeploy::Device::TPU; break;
    default: dev = modeldeploy::Device::CPU; break;
    }
    modeldeploy::vision::LetterBoxRecord lbr{};
        std::vector<modeldeploy::vision::InstanceSegResult> results;
    if (!model_ptr->predict_nv12(src_y, src_uv, width, height, step_y, step_uv,
                                 &results, &lbr, dev, nullptr)) {
        return MDStatusCode::ModelPredictFailed;
    }
    iseg_results_2_c_results(results, c_results);
    return MDStatusCode::Success;
}
