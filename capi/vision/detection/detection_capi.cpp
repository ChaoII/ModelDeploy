//
// Created by AC on 2024-12-17.
//

#include "csrc/vision.h"
#include "csrc/core/md_log.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/detection/detection_capi.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"


MDStatusCode md_create_detection_model(MDModel* model, const char* model_path,
                                       const MDRuntimeOption* option) {
    modeldeploy::RuntimeOption _option;
    c_runtime_option_2_runtime_option(option, &_option);
    const auto detection_model = new modeldeploy::vision::detection::UltralyticsDet(model_path, _option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(detection_model->name().c_str());
    model->model_content = detection_model;
    model->type = MDModelType::Detection;
    if (!detection_model->is_initialized()) {
        MD_LOG_ERROR << "Detection model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDStatusCode md_set_detection_input_size(const MDModel* model, const MDSize size) {
    if (model->type != MDModelType::Detection) {
        MD_LOG_ERROR << "Model type is not detection!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto detection_model = static_cast<modeldeploy::vision::detection::UltralyticsDet*>(model->model_content);
    detection_model->get_preprocessor().set_size({size.width, size.height});
    return MDStatusCode::Success;
}

MDStatusCode md_detection_predict(const MDModel* model, MDImage* image, MDDetectionResults* c_results) {
    if (model->type != MDModelType::Detection) {
        MD_LOG_ERROR << "Model type is not detection!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto cv_image = md_image_to_mat(image);
    std::vector<modeldeploy::vision::DetectionResult> results;
    const auto detection_model = static_cast<modeldeploy::vision::detection::UltralyticsDet*>(model->model_content);
    if (const bool res_status = detection_model->predict(cv_image, &results); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    detection_results_2_c_results(results, c_results);
    return MDStatusCode::Success;
}


void md_print_detection_result(const MDDetectionResults* c_results) {
    std::vector<modeldeploy::vision::DetectionResult> results;
    c_results_2_detection_results(c_results, &results);
    dis_det(results);
}


void md_draw_detection_result(const MDImage* image, const MDDetectionResults* c_results,
                              const double threshold, const char* font_path, const int font_size,
                              const double alpha, const int save_result) {
    auto cv_image = md_image_to_mat(image);
    std::vector<modeldeploy::vision::DetectionResult> results;
    c_results_2_detection_results(c_results, &results);
    modeldeploy::vision::vis_det(cv_image, results, threshold, font_path, font_size, alpha, save_result);
}

void md_free_detection_result(MDDetectionResults* c_results) {
    if (c_results->size > 0 && c_results->data != nullptr) {
        c_results->size = 0;
        delete [] c_results->data;
        c_results->data = nullptr;
    }
}

void md_free_detection_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::detection::UltralyticsDet*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
