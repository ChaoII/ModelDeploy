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


MDStatusCode md_create_keypoint_model(MDModel* model, const char* model_path,
                                      const MDRuntimeOption* option) {
    modeldeploy::RuntimeOption _option;
    c_runtime_option_2_runtime_option(option, &_option);
    const auto keypoint_model = new modeldeploy::vision::detection::UltralyticsPose(model_path, _option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(keypoint_model->name().c_str());
    model->model_content = keypoint_model;
    model->type = MDModelType::Keypoint;
    if (!keypoint_model->is_initialized()) {
        MD_LOG_ERROR << "Detection model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDStatusCode md_set_keypoint_input_size(const MDModel* model, const MDSize size) {
    if (model->type != MDModelType::Keypoint) {
        MD_LOG_ERROR << "Model type is not keypoint!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto keypoint_model = static_cast<modeldeploy::vision::detection::UltralyticsPose*>(model->model_content);
    keypoint_model->get_preprocessor().set_size({size.width, size.height});
    return MDStatusCode::Success;
}

MODELDEPLOY_CAPI_EXPORT MDStatusCode md_set_keypoint_num(
    const MDModel* model, const int num) {
    if (model->type != MDModelType::Keypoint) {
        MD_LOG_ERROR << "Model type is not keypoint!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto keypoint_model = static_cast<modeldeploy::vision::detection::UltralyticsPose*>(model->model_content);
    keypoint_model->get_postprocessor().set_keypoints_num(num);
    return MDStatusCode::Success;
}


MDStatusCode md_keypoint_predict(const MDModel* model, MDImage* image, MDKeyPointResults* c_results) {
    if (model->type != MDModelType::Keypoint) {
        MD_LOG_ERROR << "Model type is not keypoint!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto image_data = md_image_to_image_data(image);
    std::vector<modeldeploy::vision::KeyPointsResult> results;
    const auto keypoint_model = static_cast<modeldeploy::vision::detection::UltralyticsPose*>(model->model_content);
    if (const bool res_status = keypoint_model->predict(image_data, &results); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    keypoint_results_2_c_results(results, c_results);
    return MDStatusCode::Success;
}


void md_print_keypoint_result(const MDKeyPointResults* c_results) {
    std::vector<modeldeploy::vision::KeyPointsResult> results;
    c_results_2_keypoint_results(c_results, &results);
    dis_pose(results);
}


void md_draw_keypoint_result(const MDImage* image, const MDKeyPointResults* c_results,
                             const char* font_path, const int font_size,
                             const int keypoint_radius,
                             const double alpha, const int save_result, const int draw_lines) {
    auto image_data = md_image_to_image_data(image);
    std::vector<modeldeploy::vision::KeyPointsResult> results;
    c_results_2_keypoint_results(c_results, &results);
    // 人体关键点识别
    if (!results.empty() && results[0].keypoints.size() == 17) {
        modeldeploy::vision::vis_pose(image_data,
                                      results,
                                      font_path,
                                      font_size,
                                      keypoint_radius,
                                      alpha, save_result);
    }
    else {
        // 其它关键点识别
        modeldeploy::vision::vis_keypoints(image_data,
                                           results,
                                           font_path,
                                           font_size,
                                           keypoint_radius,
                                           alpha,
                                           save_result,
                                           draw_lines);
    }
}

void md_free_keypoint_result(MDKeyPointResults* c_results) {
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

void md_free_keypoint_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::detection::UltralyticsPose*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
