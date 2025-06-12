//
// Created by AC on 2024-12-17.
//


#include <map>
#include "csrc/vision.h"
#include "csrc/core/md_log.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/pose/pose_capi.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"


MDStatusCode md_create_pose_model(MDModel* model, const char* model_path,
                                  const MDRuntimeOption* option) {
    modeldeploy::RuntimeOption _option;
    c_runtime_option_2_runtime_option(option, &_option);
    const auto pose_model = new modeldeploy::vision::detection::UltralyticsPose(model_path, _option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(pose_model->name().c_str());
    model->model_content = pose_model;
    model->type = MDModelType::Detection;
    if (!pose_model->is_initialized()) {
        MD_LOG_ERROR << "Detection model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDStatusCode md_set_pose_input_size(const MDModel* model, const MDSize size) {
    if (model->type != MDModelType::Detection) {
        MD_LOG_ERROR << "Model type is not pose!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto pose_model = static_cast<modeldeploy::vision::detection::UltralyticsPose*>(model->model_content);
    pose_model->get_preprocessor().set_size({size.width, size.height});
    return MDStatusCode::Success;
}

MDStatusCode md_pose_predict(const MDModel* model, MDImage* image, MDPoseResults* c_results) {
    if (model->type != MDModelType::Detection) {
        MD_LOG_ERROR << "Model type is not pose!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto cv_image = md_image_to_mat(image);
    std::vector<modeldeploy::vision::PoseResult> results;
    const auto pose_model = static_cast<modeldeploy::vision::detection::UltralyticsPose*>(model->model_content);
    if (const bool res_status = pose_model->predict(cv_image, &results); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    pose_results_2_c_results(results, c_results);
    return MDStatusCode::Success;
}


void md_print_pose_result(const MDPoseResults* c_results) {
    std::vector<modeldeploy::vision::PoseResult> results;
    c_results_2_pose_results(c_results, &results);
    dis_pose(results);
}


void md_draw_pose_result(const MDImage* image, const MDPoseResults* c_results,
                         const char* font_path, const int font_size,
                         const int keypoint_radius,
                         const double alpha, const int save_result) {
    auto cv_image = md_image_to_mat(image);
    std::vector<modeldeploy::vision::PoseResult> results;
    c_results_2_pose_results(c_results, &results);
    modeldeploy::vision::vis_pose(cv_image, results, font_path, font_size, keypoint_radius, alpha, save_result);
}

void md_free_pose_result(MDPoseResults* c_results) {
    if (c_results->size > 0 && c_results->data != nullptr) {
        for (int i = 0; i < c_results->size; i++) {
            delete[] c_results->data[i].keypoints;
            c_results->data[i].keypoints = nullptr;
            c_results->data[i].keypoints_size = 0;
        }
        c_results->size = 0;
        delete [] c_results->data;
        c_results->data = nullptr;
    }
}

void md_free_pose_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::detection::UltralyticsPose*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
