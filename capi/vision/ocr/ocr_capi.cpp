//
// Created by AC on 2024/12/16.
//

#include <string>
#include <filesystem>


#include "csrc/vision.h"
#include "csrc/core/md_log.h"
#include "capi/vision/ocr/ocr_capi.h"
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
    const auto rec_model_file_path = fs::path(parameters->model_dir) / "rec_infer1.onnx";
    const auto ocr_model = new modeldeploy::vision::ocr::PPOCRv4(det_model_file_path.string(),
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
    const auto ocr_model = static_cast<modeldeploy::vision::ocr::PPOCRv4*>(model->model_content);
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


MDStatusCode md_ocr_model_predict(const MDModel* model, MDImage* image, MDOCRResults* results) {
    const auto cv_image = md_image_to_mat(image);
    modeldeploy::vision::OCRResult res;
    const auto ocr_model = static_cast<modeldeploy::vision::ocr::PPOCRv4*>(model->model_content);
    if (const bool res_status = ocr_model->predict(cv_image, &res); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    const auto r_size = res.boxes.size();
    results->size = static_cast<int>(r_size);
    if (r_size == 0) {
        results->data = nullptr;
        return MDStatusCode::Success;
    }
    results->data = static_cast<MDOCRResult*>(malloc(sizeof(MDOCRResult) * r_size));
    for (int i = 0; i < r_size; ++i) {
        auto text = res.text[i];
        results->data[i].text = static_cast<char*>(malloc(text.size() + 1));
        memcpy(results->data[i].text, text.c_str(), text.size() + 1);
        results->data[i].score = res.rec_scores[i];
        // const 保证 data和size成员本身不被修改，但是不会限制data指向的内容不被修改
        const MDPolygon polygon{static_cast<MDPoint*>(malloc(sizeof(MDPoint) * 4)), 4};
        for (int j = 0; j < 4; ++j) {
            polygon.data[j] = {res.boxes[i][j * 2], res.boxes[i][j * 2 + 1]};
        }
        results->data[i].box = polygon;
    }
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

void md_draw_ocr_result(const MDImage* image, const MDOCRResults* results, const char* font_path, const int font_size,
                        const MDColor* color, const double alpha, const int save_result) {
    cv::Mat overlay;
    cv::Mat cv_image = md_image_to_mat(image);
    cv_image.copyTo(overlay);
    cv::FontFace font(font_path);
    cv::Scalar cv_color(color->b, color->g, color->r);
    // 绘制半透明部分（填充多边形）和文字背景色
    for (int i = 0; i < results->size; ++i) {
        const auto polygon = results->data[i].box;
        std::vector<cv::Point> points;
        points.reserve(polygon.size);
        for (int j = 0; j < polygon.size; ++j) {
            points.emplace_back(polygon.data[j].x, polygon.data[j].y);
        }
        cv::fillPoly(overlay, points, cv_color, cv::LINE_AA, 0);
        const auto size = cv::getTextSize(cv::Size(0, 0), results->data[i].text,
                                          {points[0].x, points[0].y}, font, font_size);
        cv::rectangle(cv_image, size, cv_color, -1, cv::LINE_AA, 0);
    }
    cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
    // 绘制多边形边框，文字背景边框，文字
    for (int i = 0; i < results->size; ++i) {
        const auto polygon = results->data[i].box;
        std::vector<cv::Point> points;
        points.reserve(polygon.size);
        for (int j = 0; j < polygon.size; ++j) {
            points.emplace_back(polygon.data[j].x, polygon.data[j].y);
        }
        cv::polylines(cv_image, points, true, cv_color, 1, cv::LINE_AA, 0);
        const auto size = cv::getTextSize(cv::Size(0, 0), results->data[i].text,
                                          {points[0].x, points[0].y}, font, font_size);
        cv::rectangle(cv_image, size, cv_color, 1, cv::LINE_AA, 0);
        const auto inv_color = cv::Scalar(255 - cv_color[0], 255 - cv_color[1], 255 - cv_color[2]);
        cv::putText(cv_image, results->data[i].text, {points[0].x, points[0].y - 2},
                    inv_color, font, font_size);
    }
    if (save_result) {
        cv::imwrite("vis_result.jpg", cv_image);
    }
}

void md_free_ocr_result(MDOCRResults* results) {
    for (int i = 0; i < results->size; ++i) {
        free(results->data[i].text);
        free(results->data[i].box.data);
    }
    free(results->data);
    results->data = nullptr;
    results->size = 0;
}

void md_free_ocr_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::ocr::PPOCRv4*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
