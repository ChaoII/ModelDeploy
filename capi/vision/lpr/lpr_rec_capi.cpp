//
// Created by aichao on 2025-5-26.
//


#include "csrc/vision.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/lpr/lpr_rec_capi.h"


MDStatusCode md_create_lpr_rec_model(MDModel* model, const char* model_path,
                                     const int thread_num) {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(thread_num);
    const auto lpr_rec_model = new modeldeploy::vision::lpr::LprRecognizer(model_path, option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(lpr_rec_model->name().c_str());
    model->model_content = lpr_rec_model;
    model->type = MDModelType::LPR;
    if (!lpr_rec_model->is_initialized()) {
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}


MDStatusCode md_lpr_rec_predict(const MDModel* model, MDImage* image, MDLPRResults* c_result) {
    if (model->type != MDModelType::LPR) {
        return MDStatusCode::ModelTypeError;
    }
    auto cv_image = md_image_to_mat(image);
    std::vector<modeldeploy::vision::LprResult> results;
    results.resize(1);
    const auto lpr_rec_model = static_cast<modeldeploy::vision::lpr::LprRecognizer*>(model->model_content);
    if (const bool res_status = lpr_rec_model->predict(cv_image, &results[0]); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    lpr_results_2_c_results(results, c_result);
    return MDStatusCode::Success;
}


void md_print_lpr_rec_result(const MDLPRResults* c_result) {
    std::vector<modeldeploy::vision::LprResult> results;
    c_results_2_lpr_results(c_result, &results);
    // results.display();
}


void md_free_lpr_rec_result(MDLPRResults* c_results) {
    if (c_results->size > 0 && c_results->data != nullptr) {
        for (int i = 0; i < c_results->size; i++) {
            if (c_results->data[i].landmarks) {
                delete[] c_results->data[i].landmarks;
                c_results->data[i].landmarks = nullptr;
            }
            free(c_results->data[i].car_plate_str);
            free(c_results->data[i].car_plate_color);
        }
        delete[] c_results->data;
        c_results->data = nullptr;
        c_results->size = 0;
    }
}

void md_free_lpr_rec_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::lpr::LprRecognizer*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
