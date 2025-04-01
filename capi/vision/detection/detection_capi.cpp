//
// Created by AC on 2024-12-17.
//


#include <map>
#include <tabulate/tabulate.hpp>
#include "csrc/vision.h"
#include "csrc/core/md_log.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/detection/detection_capi.h"


MDStatusCode md_create_detection_model(MDModel* model, const char* model_path,
                                       const int thread_num) {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(thread_num);
    const auto detection_model = new modeldeploy::vision::detection::YOLOv8(model_path, option);
    if (!detection_model->initialized()) {
        MD_LOG_ERROR << "Detection model initial failed!" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(detection_model->name().c_str());
    model->model_content = detection_model;
    model->type = MDModelType::Detection;
    return MDStatusCode::Success;
}

MDStatusCode md_set_detection_input_size(const MDModel* model, const MDSize size) {
    if (model->type != MDModelType::Detection) {
        MD_LOG_ERROR << "Model type is not detection!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto detection_model = static_cast<modeldeploy::vision::detection::YOLOv8*>(model->model_content);
    detection_model->get_preprocessor().set_size({size.height, size.height});
    return MDStatusCode::Success;
}

MDStatusCode md_detection_predict(const MDModel* model, MDImage* image, MDDetectionResults* results) {
    if (model->type != MDModelType::Detection) {
        MD_LOG_ERROR << "Model type is not detection!" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto cv_image = md_image_to_mat(image);
    modeldeploy::vision::DetectionResult res;
    const auto detection_model = static_cast<modeldeploy::vision::detection::YOLOv8*>(model->model_content);
    if (const bool res_status = detection_model->predict(cv_image, &res); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    const auto r_size = res.boxes.size();
    results->size = static_cast<int>(r_size);
    if (r_size == 0) {
        MD_LOG_WARN << "Not found detect object!" << std::endl;
        results->data = nullptr;
        return MDStatusCode::Success;
    }
    results->data = static_cast<MDDetectionResult*>(malloc(sizeof(MDDetectionResult) * r_size));
    for (int i = 0; i < r_size; ++i) {
        auto box = res.boxes[i];
        results->data[i].box = {
            static_cast<int>(box[0]), static_cast<int>(box[1]),
            static_cast<int>(box[2] - box[0]), static_cast<int>(box[3] - box[1])
        };
        results->data[i].score = res.scores[i];
        results->data[i].label_id = res.label_ids[i];
    }
    return MDStatusCode::Success;
}


void md_print_detection_result(const MDDetectionResults* result) {
    tabulate::Table detection_results_table;
    detection_results_table.format()
                           .border_color(tabulate::Color::magenta)
                           .font_color(tabulate::Color::green)
                           .corner_color(tabulate::Color::magenta);
    detection_results_table.add_row({"box([x,y,w,h])", "label_id", "score"});
    for (int i = 0; i < result->size; ++i) {
        detection_results_table.add_row({
            format_rect(result->data[i].box),
            std::to_string(result->data[i].label_id),
            std::to_string(result->data[i].score)
        });
    }
    detection_results_table.column(1).format().font_align(tabulate::FontAlign::center);
    std::cout << detection_results_table << std::endl;
}


void md_draw_detection_result(const MDImage* image, const MDDetectionResults* result,
                              const char* font_path, const int font_size,
                              const double alpha, const int save_result) {
    cv::Mat overlay;
    cv::Mat cv_image = md_image_to_mat(image);
    cv_image.copyTo(overlay);
    cv::FontFace font(font_path);
    // 根据label_id获取颜色
    std::map<int, cv::Scalar> color_map;
    // 绘制半透明部分（填充矩形）
    for (int i = 0; i < result->size; ++i) {
        auto class_id = result->data[i].label_id;
        if (color_map.find(class_id) == color_map.end()) {
            color_map[class_id] = get_random_color();
        }
        auto [x, y, width, height] = result->data[i].box;
        auto cv_color = color_map[class_id];
        // 绘制对象矩形框
        cv::rectangle(overlay, {x, y, width, height}, cv_color, -1);
        const auto size = cv::getTextSize(cv::Size(0, 0), std::to_string(class_id), cv::Point(x, y), font, font_size);
        // 绘制标签背景
        cv::rectangle(overlay, size, cv_color, -1);
    }
    cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
    // 绘制对象矩形矩形边框、文字背景边框、文字
    for (int i = 0; i < result->size; ++i) {
        auto class_id = result->data[i].label_id;
        const auto [x, y, width, height] = result->data[i].box;
        auto cv_color = color_map[class_id];
        cv::rectangle(cv_image, cv::Rect(x, y, width, height), cv_color, 1, cv::LINE_AA, 0);
        const auto size = cv::getTextSize(cv::Size(0, 0), std::to_string(class_id),
                                          cv::Point(x, y), font, font_size);
        cv::rectangle(cv_image, size, cv_color, 1, cv::LINE_AA, 0);
        cv::putText(cv_image, std::to_string(class_id), cv::Point(x, y - 2),
                    cv::Scalar(255 - cv_color[0], 255 - cv_color[1], 255 - cv_color[2]), font, font_size);
    }
    if (save_result) {
        MD_LOG_INFO << "Save detection result to [vis_result.jpg]" << std::endl;
        cv::imwrite("vis_result.jpg", cv_image);
    }
}

void md_free_detection_result(MDDetectionResults* result) {
    if (result->size > 0 && result->data != nullptr) {
        result->size = 0;
        free(result->data);
        result->data = nullptr;
    }
}

void md_free_detection_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::detection::YOLOv8*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
