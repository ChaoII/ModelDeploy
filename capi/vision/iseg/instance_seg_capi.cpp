//
// Created by AC on 2024-12-17.
//


#include <map>
#include "csrc/vision.h"
#include "csrc/core/md_log.h"
#include "csrc/vision/common/visualize/visualize.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/iseg/instance_seg_capi.h"


MDStatusCode md_create_instance_seg_model(MDModel* model, const char* model_path,
                                          const int thread_num) {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(thread_num);
    const auto instance_seg_model = new modeldeploy::vision::detection::UltralyticsSeg(model_path, option);
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
    const auto instance_seg_model = static_cast<modeldeploy::vision::detection::UltralyticsSeg*>(model->model_content);
    instance_seg_model->get_preprocessor().set_size({size.width, size.height});
    return MDStatusCode::Success;
}

MDStatusCode md_instance_seg_predict(const MDModel* model, MDImage* image, MDIsegResults* c_results) {
    if (model->type != MDModelType::Detection) {
        MD_LOG_ERROR << "Model type is not instance_seg!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto cv_image = md_image_to_mat(image);
    std::vector<modeldeploy::vision::InstanceSegResult> results;
    const auto instance_seg_model = static_cast<modeldeploy::vision::detection::UltralyticsSeg*>(model->model_content);
    if (const bool res_status = instance_seg_model->predict(cv_image, &results); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    iseg_results_2_c_results(results, c_results);
    return MDStatusCode::Success;
}


void md_print_instance_seg_result(const MDIsegResults* c_results) {
    std::vector<modeldeploy::vision::InstanceSegResult> results;
    c_results_2_iseg_results(c_results, &results);
    // result.display();
}


void md_draw_instance_seg_result(const MDImage* image, const MDIsegResults* c_results,
                                 const double threshold, const char* font_path, const int font_size,
                                 const double alpha, const int save_result) {
    auto cv_image = md_image_to_mat(image);
    std::vector<modeldeploy::vision::InstanceSegResult> results;
    c_results_2_iseg_results(c_results, &results);
    modeldeploy::vision::vis_iseg(cv_image, results, threshold, font_path, font_size, alpha, save_result);
}

void md_free_instance_seg_result(MDIsegResults* c_results) {
    if (c_results->size > 0 && c_results->data != nullptr) {
        for (int i = 0; i < c_results->size; i++) {
            delete[] c_results->data[i].mask.buffer;
            delete[] c_results->data[i].mask.shape;
            c_results->data[i].mask.buffer = nullptr;
            c_results->data[i].mask.shape = nullptr;
            c_results->data[i].mask.buffer_size = 0;
            c_results->data[i].mask.num_dims = 0;
        }
        c_results->size = 0;
        delete[] c_results->data;
        c_results->data = nullptr;
    }
}

void md_free_instance_seg_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::detection::UltralyticsSeg*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
