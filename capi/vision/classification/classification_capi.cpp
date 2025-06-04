//
// Created by AC on 2025-5-26.
//


#include <map>
#include "csrc/vision.h"
#include "csrc/core/md_log.h"
#include "csrc/vision/common/visualize/visualize.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/classification/classification_capi.h"

#include <csrc/vision/common/display/display.h>


MDStatusCode md_create_classification_model(MDModel* model, const char* model_path,
                                            const int thread_num) {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(thread_num);
    const auto classification_model = new modeldeploy::vision::classification::UltralyticsCls(model_path, option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(classification_model->name().c_str());
    model->model_content = classification_model;
    model->type = MDModelType::Classification;
    if (!classification_model->is_initialized()) {
        MD_LOG_ERROR << "Classification model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDStatusCode md_set_classification_input_size(const MDModel* model, const MDSize size) {
    if (model->type != MDModelType::Classification) {
        MD_LOG_ERROR << "Model type is not classification!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto classification_model = static_cast<modeldeploy::vision::classification::UltralyticsCls*>(model->
        model_content);
    classification_model->get_preprocessor().set_size({size.height, size.height});
    return MDStatusCode::Success;
}

MDStatusCode md_classification_predict(const MDModel* model, MDImage* image, MDClassificationResults* c_results) {
    if (model->type != MDModelType::Classification) {
        MD_LOG_ERROR << "Model type is not classification!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto cv_image = md_image_to_mat(image);
    modeldeploy::vision::ClassifyResult result;
    const auto detection_model = static_cast<modeldeploy::vision::classification::UltralyticsCls*>(model->
        model_content);
    if (const bool res_status = detection_model->predict(cv_image, &result); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    classification_result_2_c_results(result, c_results);
    return MDStatusCode::Success;
}


void md_print_classification_result(const MDClassificationResults* c_results) {
    modeldeploy::vision::ClassifyResult result;
    c_results_2_classification_result(c_results, &result);
    dis_cls(result);
}


void md_draw_classification_result(const MDImage* image, const MDClassificationResults* c_results,
                                   const int top_k,
                                   const float score_threshold,
                                   const char* font_path, const int font_size,
                                   const double alpha, const int save_result) {
    auto cv_image = md_image_to_mat(image);
    modeldeploy::vision::ClassifyResult result;
    c_results_2_classification_result(c_results, &result);
    modeldeploy::vision::vis_classification(cv_image, result, top_k, score_threshold,
                                            font_path, font_size, alpha, save_result);
}

void md_free_classification_result(MDClassificationResults* c_results) {
    if (c_results->size > 0 && c_results->data != nullptr) {
        c_results->size = 0;
        delete [] c_results->data;
        c_results->data = nullptr;
    }
}

void md_free_classification_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::classification::UltralyticsCls*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
