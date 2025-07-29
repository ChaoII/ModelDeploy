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
#include "capi/vision/ocr/structure_pipeline_capi.h"

namespace fs = std::filesystem;

MDStatusCode md_create_structure_table_model(MDModel* model, const MDStructureTableModelParameters* parameters,
                                             const MDRuntimeOption* option) {
    const auto det_model_file_path = fs::path(parameters->det_model_file);
    const auto rec_model_file_path = fs::path(parameters->rec_model_file);
    const auto tab_model_file_path = fs::path(parameters->table_model_file);
    const auto rec_label_file_path = fs::path(parameters->rec_label_file);
    const auto tab_label_file_path = fs::path(parameters->table_char_dict_path);
    modeldeploy::RuntimeOption _option;
    c_runtime_option_2_runtime_option(option, &_option);
    const auto structure_table_model = new modeldeploy::vision::ocr::PPStructureV2Table(
        det_model_file_path.string(),
        rec_model_file_path.string(),
        tab_model_file_path.string(),
        rec_label_file_path.string(),
        tab_label_file_path.string(),
        parameters->max_side_len,
        parameters->det_db_thresh,
        parameters->det_db_box_thresh,
        parameters->det_db_unclip_ratio,
        parameters->det_db_score_mode,
        parameters->use_dilation,
        parameters->rec_batch_size, _option);

    model->type = MDModelType::OCR;
    model->format = MDModelFormat::ONNX;
    model->model_content = structure_table_model;
    model->model_name = strdup(structure_table_model->name().c_str());
    if (!structure_table_model->is_initialized()) {
        MD_LOG_ERROR << "Detection model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}


MDStatusCode md_structure_table_model_predict(const MDModel* model, MDImage* image, MDOCRResults* c_results) {
    const auto image_data = md_image_to_image_data(image);
    modeldeploy::vision::OCRResult result;
    const auto structure_table_model = static_cast<modeldeploy::vision::ocr::PPStructureV2Table*>(model->
        model_content);
    if (const bool res_status = structure_table_model->predict(image_data, &result); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    ocr_result_2_c_results(result, c_results);
    return MDStatusCode::Success;
}

void md_print_structure_table_result(const MDOCRResults* results) {
    OCRResult result;
    c_results_2_ocr_result(results, &result);
    dis_ocr(result);
}

void md_draw_structure_table_result(const MDImage* image, const MDOCRResults* c_results, const char* font_path,
                                    const int font_size, const double alpha, const int save_result) {
    auto image_data = md_image_to_image_data(image);
    modeldeploy::vision::OCRResult result;
    c_results_2_ocr_result(c_results, &result);
    modeldeploy::vision::vis_ocr(image_data, result, font_path, font_size, alpha, save_result);
}

void md_free_structure_table_result(MDOCRResults* c_results) {
    for (int i = 0; i < c_results->size; ++i) {
        free(c_results->data[i].text);
        delete[] c_results->data[i].box.data;
        c_results->data[i].box.data = nullptr;
        c_results->data[i].box.size = 0;
        free(c_results->data[i].table_structure);
        delete[] c_results->data[i].table_boxes.data;
        c_results->data[i].table_boxes.data = nullptr;
        c_results->data[i].table_boxes.size = 0;
    }
    delete[] c_results->data;
    free(c_results->table_html);
    c_results->data = nullptr;
    c_results->table_html = nullptr;
    c_results->size = 0;
}

void md_free_structure_table_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::ocr::PaddleOCR*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
