//
// Created by AC on 2024/12/16.
//

#include <string>
#include <filesystem>


#include "csrc/vision.h"
#include "csrc/core/md_log.h"
#include "capi/vision/ocr/ocr_capi.h"

#include <csrc/vision/common/visualize/visualize.h>

#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"

namespace fs = std::filesystem;

MDStatusCode md_create_ocr_model(MDModel* model, const MDOCRModelParameters* parameters) {
    if (!fs::exists(parameters->model_dir)) {
        MD_LOG_ERROR << "Model directory " << parameters->model_dir << " is not existed!";
        return MDStatusCode::PathNotFound;
    }
    const auto det_model_file_path = fs::path(parameters->model_dir) / "det_infer.onnx";
    const auto cls_model_file_path = fs::path(parameters->model_dir) / "cls_infer.onnx";
    const auto rec_model_file_path = fs::path(parameters->model_dir) / "rec_infer.onnx";
    const auto ocr_model = new modeldeploy::vision::ocr::PaddleOCR(det_model_file_path.string(),
                                                                   cls_model_file_path.string(),
                                                                   rec_model_file_path.string(),
                                                                   parameters->dict_path,
                                                                   parameters->thread_num,
                                                                   parameters->max_side_len,
                                                                   parameters->det_db_thresh,
                                                                   parameters->det_db_box_thresh,
                                                                   parameters->det_db_unclip_ratio,
                                                                   parameters->det_db_score_mode,
                                                                   parameters->use_dilation,
                                                                   parameters->rec_batch_size);

    model->type = MDModelType::OCR;
    model->format = parameters->format;
    model->model_content = ocr_model;
    model->model_name = strdup(ocr_model->name().c_str());
    if (!ocr_model->is_initialized()) {
        MD_LOG_ERROR << "Detection model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDRect md_get_text_position(const MDModel* model, MDImage* image, const char* text) {
    const cv::Mat cv_image = md_image_to_mat(image);
    modeldeploy::vision::OCRResult res;
    const auto ocr_model = static_cast<modeldeploy::vision::ocr::PaddleOCR*>(model->model_content);
    if (const bool res_status = ocr_model->predict(cv_image, &res); !res_status) {
        return MDRect{0, 0, 0, 0};
    }
    for (int i = 0; i < res.boxes.size(); ++i) {
        if (contains_substring(res.text[i], text)) {
            std::vector<cv::Point> polygon;
            polygon.reserve(4);
            for (int j = 0; j < 4; ++j) {
                polygon.emplace_back(res.boxes[i][j * 2], res.boxes[i][j * 2 + 1]);
            }
            const cv::Rect boundingRect = cv::boundingRect(polygon);
            return MDRect{boundingRect.x, boundingRect.y, boundingRect.width, boundingRect.height};
        }
    }
    return MDRect{0, 0, 0, 0};
}


MDStatusCode md_ocr_model_predict(const MDModel* model, MDImage* image, MDOCRResults* c_results) {
    const auto cv_image = md_image_to_mat(image);
    modeldeploy::vision::OCRResult result;
    const auto ocr_model = static_cast<modeldeploy::vision::ocr::PaddleOCR*>(model->model_content);
    if (const bool res_status = ocr_model->predict(cv_image, &result); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    ocr_result_2_c_results(result, c_results);
    return MDStatusCode::Success;
}

void md_print_ocr_result(const MDOCRResults* results) {
    for (int i = 0; i < results->size; ++i) {
        std::cout
            << "box: " << format_polygon(results->data[i].box)
            << " text: " << results->data[i].text
            << " score: " << results->data[i].score
            << std::endl;
    }
}

void md_draw_ocr_result(const MDImage* image, const MDOCRResults* c_results, const char* font_path,
                        const int font_size, const double alpha, const int save_result) {
    cv::Mat cv_image = md_image_to_mat(image);
    modeldeploy::vision::OCRResult result;
    c_results_2_ocr_result(c_results, &result);
    modeldeploy::vision::vis_ocr(cv_image, result, font_path, font_size, alpha, save_result);
}

void md_free_ocr_result(MDOCRResults* c_results) {
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

void md_free_ocr_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::ocr::PaddleOCR*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
