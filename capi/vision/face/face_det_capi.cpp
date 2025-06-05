//
// Created by aichao on 2025-4-7.
//


#include "csrc/vision.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/face/face_det_capi.h"

#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"


MDStatusCode md_create_face_det_model(MDModel* model, const char* model_path,
                                      const int thread_num) {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(thread_num);
    const auto face_det_model = new modeldeploy::vision::face::Scrfd(model_path, option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(face_det_model->name().c_str());
    model->model_content = face_det_model;
    model->type = MDModelType::FACE;
    if (!face_det_model->is_initialized()) {
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}


MDStatusCode md_face_det_predict(const MDModel* model, MDImage* image, MDDetectionLandmarkResults* c_results) {
    if (model->type != MDModelType::FACE) {
        return MDStatusCode::ModelTypeError;
    }
    auto cv_image = md_image_to_mat(image);
    std::vector<modeldeploy::vision::DetectionLandmarkResult> results;
    const auto face_det_model = static_cast<modeldeploy::vision::face::Scrfd*>(model->model_content);
    if (const bool res_status = face_det_model->predict(cv_image, &results); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    detection_landmark_result_2_c_results(results, c_results);
    return MDStatusCode::Success;
}


void md_print_face_det_result(const MDDetectionLandmarkResults* c_results) {
    std::vector<modeldeploy::vision::DetectionLandmarkResult> results;
    c_results_2_detection_landmark_result(c_results, &results);
    dis_lmk(results);
}


void md_draw_face_det_result(const MDImage* image, const MDDetectionLandmarkResults* c_results,
                             const char* font_path, const int font_size, const int landmark_radius,
                             const double alpha, const int save_result) {
    auto cv_image = md_image_to_mat(image);
    std::vector<modeldeploy::vision::DetectionLandmarkResult> results;
    c_results_2_detection_landmark_result(c_results, &results);
    modeldeploy::vision::vis_det_landmarks(cv_image, results, font_path, font_size,
                                           landmark_radius, alpha, save_result);
}


void md_free_face_det_result(MDDetectionLandmarkResults* c_results) {
    if (c_results->size > 0 && c_results->data != nullptr) {
        for (int i = 0; i < c_results->size; i++) {
            if (c_results->data[i].landmarks) {
                delete[] c_results->data[i].landmarks;
                c_results->data[i].landmarks = nullptr;
                c_results->data[i].landmarks_size = 0;
            }
        }
        delete[] c_results->data;
        c_results->data = nullptr;
        c_results->size = 0;
    }
}

void md_free_face_det_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::face::Scrfd*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
