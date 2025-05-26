//
// Created by AC on 2024-12-17.
//


#include <map>
#include "csrc/vision.h"
#include "csrc/core/md_log.h"
#include "csrc/vision/common/visualize/visualize.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/instance_seg/instance_seg_capi.h"


MDStatusCode md_create_instance_seg_model(MDModel* model, const char* model_path,
                                          const int thread_num) {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(thread_num);
    const auto instance_seg_model = new modeldeploy::vision::detection::YOLOv5Seg(model_path, option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(instance_seg_model->name().c_str());
    model->model_content = instance_seg_model;
    model->type = MDModelType::Detection;
    if (!instance_seg_model->is_initialized()) {
        MD_LOG_ERROR << "Instance segmentation model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDStatusCode md_set_instance_seg_input_size(const MDModel* model, const MDSize size) {
    if (model->type != MDModelType::Detection) {
        MD_LOG_ERROR << "Model type is not instance_seg!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto instance_seg_model = static_cast<modeldeploy::vision::detection::YOLOv5Seg*>(model->model_content);
    instance_seg_model->get_preprocessor().set_size({size.width, size.height});
    return MDStatusCode::Success;
}

MDStatusCode md_instance_seg_predict(const MDModel* model, MDImage* image, MDDetectionResults* c_results) {
    if (model->type != MDModelType::Detection) {
        MD_LOG_ERROR << "Model type is not instance_seg!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto cv_image = md_image_to_mat(image);
    modeldeploy::vision::DetectionResult result;
    const auto instance_seg_model = static_cast<modeldeploy::vision::detection::YOLOv5Seg*>(model->model_content);
    if (const bool res_status = instance_seg_model->predict(cv_image, &result); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    detection_result_2_c_results(result, c_results);
    return MDStatusCode::Success;
}


void md_print_instance_seg_result(const MDDetectionResults* c_results) {
    modeldeploy::vision::DetectionResult result;
    c_results_2_detection_result(c_results, &result);
    result.display();
}


void md_draw_instance_seg_result(const MDImage* image, const MDDetectionResults* c_results,
                                 const char* font_path, const int font_size,
                                 const double alpha, const int save_result) {
    auto cv_image = md_image_to_mat(image);
    modeldeploy::vision::DetectionResult result;
    c_results_2_detection_result(c_results, &result);
    modeldeploy::vision::vis_detection(cv_image, result, font_path, font_size, alpha, save_result);
}

void md_free_instance_seg_result(MDDetectionResults* c_results) {
    if (c_results->size > 0 && c_results->data != nullptr) {
        c_results->size = 0;
        delete [] c_results->data;
        c_results->data = nullptr;
    }
}

void md_free_instance_seg_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::detection::YOLOv5Seg*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
