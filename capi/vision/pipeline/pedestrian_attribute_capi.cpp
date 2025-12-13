//
// Created by AC on 2024/12/16.
//

#include <string>
#include <filesystem>

#include "csrc/vision.h"
#include "csrc/core/md_log.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"
#include "capi/vision/pipeline/pedestrian_attribute_capi.h"


namespace fs = std::filesystem;

MDStatusCode md_create_attr_model(MDModel* model,
                                  const char* det_model_path,
                                  const char* cls_model_path,
                                  const MDRuntimeOption* option) {
    const auto det_model_file_path = fs::path(det_model_path);
    const auto cls_model_file_path = fs::path(cls_model_path);

    modeldeploy::RuntimeOption _option;
    c_runtime_option_2_runtime_option(option, &_option);
    const auto ocr_model = new modeldeploy::vision::pipeline::PedestrianAttribute(
        det_model_file_path.string(),
        cls_model_file_path.string(),
        _option);
    model->type = MDModelType::PIPELINE;
    model->format = MDModelFormat::ONNX;
    model->model_content = ocr_model;
    model->model_name = strdup(ocr_model->name().c_str());
    if (!ocr_model->is_initialized()) {
        MD_LOG_ERROR << "PedestrianAttribute model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDStatusCode md_set_attr_det_input_size(const MDModel* model, MDSize size) {
    if (model->type != MDModelType::PIPELINE) {
        MD_LOG_ERROR << "Model type is not pipeline!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto attr_model = static_cast<modeldeploy::vision::pipeline::PedestrianAttribute*>(model->model_content);
    attr_model->set_det_input_size({size.width, size.height});
    return MDStatusCode::Success;
}

MDStatusCode md_set_attr_cls_input_size(const MDModel* model, MDSize size) {
    if (model->type != MDModelType::PIPELINE) {
        MD_LOG_ERROR << "Model type is not pipeline!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto attr_model = static_cast<modeldeploy::vision::pipeline::PedestrianAttribute*>(model->model_content);
    attr_model->set_cls_input_size({size.width, size.height});
    return MDStatusCode::Success;
}

MDStatusCode md_set_attr_cls_batch_size(const MDModel* model, int batch_size) {
    if (model->type != MDModelType::PIPELINE) {
        MD_LOG_ERROR << "Model type is not pipeline!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto attr_model = static_cast<modeldeploy::vision::pipeline::PedestrianAttribute*>(model->model_content);
    attr_model->set_cls_batch_size(batch_size);
    return MDStatusCode::Success;
}

MDStatusCode md_set_attr_det_threshold(const MDModel* model, const float threshold) {
    if (model->type != MDModelType::PIPELINE) {
        MD_LOG_ERROR << "Model type is not pipeline!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto attr_model = static_cast<modeldeploy::vision::pipeline::PedestrianAttribute*>(model->model_content);
    attr_model->set_det_threshold(threshold);
    return MDStatusCode::Success;
}


MDStatusCode md_attr_model_predict(const MDModel* model, MDImage* image, MDAttributeResults* c_results) {
    const auto image_data = md_image_to_image_data(image);
    std::vector<modeldeploy::vision::AttributeResult> results;
    const auto attr_model = static_cast<modeldeploy::vision::pipeline::PedestrianAttribute*>(model->model_content);
    if (const bool res_status = attr_model->predict(image_data, &results); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    attr_results_2_c_results(results, c_results);
    return MDStatusCode::Success;
}

void md_print_attr_result(const MDAttributeResults* c_results) {
    std::vector<modeldeploy::vision::AttributeResult> results;
    c_results_2_attr_results(c_results, &results);
    dis_attr(results);
}

void md_draw_attr_result(const MDImage* image,
                         const MDAttributeResults* c_results,
                         const double threshold,
                         const char* font_path,
                         const int font_size,
                         const double alpha,
                         const int save_result) {
    auto image_data = md_image_to_image_data(image);
    std::vector<modeldeploy::vision::AttributeResult> results;
    c_results_2_attr_results(c_results, &results);
    // todo add label_map
    modeldeploy::vision::vis_attr(image_data, results, threshold, {}, font_path, font_size, alpha, save_result);
}

void md_free_attr_result(MDAttributeResults* c_results) {
    if (c_results->size > 0 && c_results->data != nullptr) {
        for (int i = 0; i < c_results->size; i++) {
            delete [] c_results->data[i].attr_scores;
            c_results->data[i].attr_scores = nullptr;
            c_results->data[i].attr_scores_size = 0;
        }
        c_results->size = 0;
        delete [] c_results->data;
        c_results->data = nullptr;
    }
}

void md_free_attr_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::pipeline::PedestrianAttribute*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
