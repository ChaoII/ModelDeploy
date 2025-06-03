//
// Created by AC on 2025-05-31.
//


#include <map>
#include "csrc/vision.h"
#include "csrc/core/md_log.h"
#include "csrc/vision/common/visualize/visualize.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/obb/obb_capi.h"


MDStatusCode md_create_obb_model(MDModel* model, const char* model_path,
                                 const int thread_num) {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(thread_num);
    const auto obb_model = new modeldeploy::vision::detection::UltralyticsObb(model_path, option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(obb_model->name().c_str());
    model->model_content = obb_model;
    model->type = MDModelType::Detection;
    if (!obb_model->is_initialized()) {
        MD_LOG_ERROR << "Detection model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDStatusCode md_set_obb_input_size(const MDModel* model, const MDSize size) {
    if (model->type != MDModelType::Detection) {
        MD_LOG_ERROR << "Model type is not obb!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto obb_model = static_cast<modeldeploy::vision::detection::UltralyticsObb*>(model->model_content);
    obb_model->get_preprocessor().set_size({size.width, size.height});
    return MDStatusCode::Success;
}

MDStatusCode md_obb_predict(const MDModel* model, MDImage* image, MDObbResults* c_results) {
    if (model->type != MDModelType::Detection) {
        MD_LOG_ERROR << "Model type is not obb!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto cv_image = md_image_to_mat(image);
    std::vector<modeldeploy::vision::ObbResult> results;
    const auto obb_model = static_cast<modeldeploy::vision::detection::UltralyticsObb*>(model->model_content);
    if (const bool res_status = obb_model->predict(cv_image, &results); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    obb_results_2_c_results(results, c_results);
    return MDStatusCode::Success;
}


void md_print_obb_result(const MDObbResults* c_results) {
    std::vector<modeldeploy::vision::ObbResult> results;
    c_results_2_obb_results(c_results, &results);
    // result.display();
}


void md_draw_obb_result(const MDImage* image, const MDObbResults* c_results,
                        const double threshold, const char* font_path, const int font_size,
                        const double alpha, const int save_result) {
    auto cv_image = md_image_to_mat(image);
    std::vector<modeldeploy::vision::ObbResult> results;
    c_results_2_obb_results(c_results, &results);
    modeldeploy::vision::vis_obb(cv_image, results, threshold, font_path, font_size, alpha, save_result);
}

void md_free_obb_result(MDObbResults* c_results) {
    if (c_results->size > 0 && c_results->data != nullptr) {
        c_results->size = 0;
        delete [] c_results->data;
        c_results->data = nullptr;
    }
}

void md_free_obb_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::detection::UltralyticsObb*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
