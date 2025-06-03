//
// Created by aichao on 2025-5-26.
//


#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/lpr/lpr_det_capi.h"


MDStatusCode md_create_lpr_det_model(MDModel* model, const char* model_path,
                                     const int thread_num) {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(thread_num);
    const auto lpr_det_model = new modeldeploy::vision::lpr::LprDetection(model_path, option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(lpr_det_model->name().c_str());
    model->model_content = lpr_det_model;
    model->type = MDModelType::LPR;
    if (!lpr_det_model->is_initialized()) {
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}


MDStatusCode md_lpr_det_predict(const MDModel* model, MDImage* image, MDDetectionLandmarkResults* c_results) {
    if (model->type != MDModelType::LPR) {
        return MDStatusCode::ModelTypeError;
    }
    const auto cv_image = md_image_to_mat(image);
    std::vector<modeldeploy::vision::DetectionLandmarkResult> results;
    const auto lpr_det_model = static_cast<modeldeploy::vision::lpr::LprDetection*>(model->model_content);
    if (const bool res_status = lpr_det_model->predict(cv_image, &results); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    detection_landmark_result_2_c_results(results, c_results);
    return MDStatusCode::Success;
}


void md_print_lpr_det_result(const MDDetectionLandmarkResults* c_results) {
    std::vector<modeldeploy::vision::DetectionLandmarkResult> results;
    c_results_2_detection_landmark_result(c_results, &results);
    // results.display();
}


void md_draw_lpr_det_result(const MDImage* image, const MDDetectionLandmarkResults* c_results,
                            const char* font_path, const int font_size, const int landmark_radius,
                            const double alpha, const int save_result) {
    const auto cv_image = md_image_to_mat(image);
    std::vector<modeldeploy::vision::DetectionLandmarkResult> results;
    c_results_2_detection_landmark_result(c_results, &results);
    modeldeploy::vision::vis_det_landmarks(cv_image, results, font_path, font_size,
                                           landmark_radius, alpha, save_result);
}


void md_free_lpr_det_result(MDDetectionLandmarkResults* c_results) {
    if (c_results->size > 0 && c_results->data != nullptr) {
        for (int i = 0; i < c_results->size; i++) {
            if (c_results->data[i].landmarks) {
                delete[] c_results->data[i].landmarks;
                c_results->data[i].landmarks = nullptr;
            }
        }
        delete[] c_results->data;
        c_results->data = nullptr;
        c_results->size = 0;
    }
}

void md_free_lpr_det_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::lpr::LprDetection*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
