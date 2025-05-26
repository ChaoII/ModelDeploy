//
// Created by aichao on 2025-5-26.
//


#include "csrc/vision.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/lpr/lpr_pipeline_capi.h"

#include <csrc/vision/common/visualize/visualize.h>


MDStatusCode md_create_lpr_pipeline_model(MDModel* model,
                                          const char* lpr_det_model_file,
                                          const char* lpr_rec_model_file,
                                          const int thread_num) {
    const auto lpr_pipeline_model = new modeldeploy::vision::lpr::LprPipeline(
        lpr_det_model_file, lpr_rec_model_file, thread_num);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(lpr_pipeline_model->name().c_str());
    model->model_content = lpr_pipeline_model;
    model->type = MDModelType::LPR;
    if (!lpr_pipeline_model->is_initialized()) {
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}


MDStatusCode md_lpr_pipeline_predict(const MDModel* model, MDImage* image,
                                     MDLPRResults* c_results) {
    if (model->type != MDModelType::LPR) {
        return MDStatusCode::ModelTypeError;
    }
    const auto cv_image = md_image_to_mat(image);
    const auto lpr_pipeline_model = static_cast<
        modeldeploy::vision::lpr::LprPipeline*>(model->model_content);

    LprResult results;
    if (const bool res_status = lpr_pipeline_model->predict(
        cv_image, &results); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }

    lpr_result_2_c_results(results, c_results);
    return MDStatusCode::Success;
}

void md_print_lpr_pipeline_result(const MDLPRResults* c_results) {
    LprResult results;
    c_results_2_lpr_result(c_results, &results);
    results.display();
}

void md_draw_lpr_pipeline_result(
    const MDImage* image, const MDLPRResults* c_results,
    const char* font_path, const int font_size,
    const int landmark_radius, const double alpha,
    const int save_result) {
    auto cv_image = md_image_to_mat(image);
    modeldeploy::vision::LprResult result;
    c_results_2_lpr_result(c_results, &result);
    modeldeploy::vision::vis_lpr(cv_image, result, font_path, font_size,
                                 landmark_radius, alpha, save_result);
}


void md_free_lpr_pipeline_result(MDLPRResults* c_results) {
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


void md_free_lpr_pipeline_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::lpr::LprPipeline*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
