//
// Created by AC on 2024-12-17.
//
#include "detection_capi.h"
#include "../utils/internal/utils.h"
#include "src/utils/utils_capi.h"
#include "fastdeploy/vision.h"
#include <map>

using YOLOv8 = fastdeploy::vision::detection::YOLOv8;
using PPYOLOE = fastdeploy::vision::detection::PPYOLOE;

MDStatusCode
md_create_detection_model(MDModel *model, const char *model_dir, int thread_num, MDModelFormat model_format) {
    if (!model) {
        return MDStatusCode::MemoryAllocatedFailed;
    }
    fastdeploy::SetLogger(false);
    fastdeploy::RuntimeOption option;
    option.UseOrtBackend();
    option.SetCpuThreadNum(thread_num);
    fastdeploy::FastDeployModel *detection_model;
    if (model_format == MDModelFormat::ONNX) {
        detection_model = new YOLOv8(model_dir, "", option);
    } else {
        std::string model_file = std::string(model_dir) + "/model.pdmodel";
        std::string param_file = std::string(model_dir) + "/model.pdiparams";
        std::string config_file = std::string(model_dir) + "/infer_cfg.yml";
        detection_model = new PPYOLOE(model_file, param_file, config_file, option);
    }
    if (!detection_model->Initialized()) {
        std::cerr << "model initial failed" << std::endl;
        return MDStatusCode::ModelInitializeFailed;
    }
    auto model_name = detection_model->ModelName();
    model->format = model_format;
    model->model_name = strdup(detection_model->ModelName().c_str());
    model->model_content = detection_model;
    model->type = MDModelType::Detection;
    return MDStatusCode::Success;
}

MDStatusCode md_set_detection_input_size(const MDModel *model, const MDSize size) {


    if (!model) return MDStatusCode::MemoryAllocatedFailed;
    if (model->format == MDModelFormat::PaddlePaddle) return MDStatusCode::CallError;
    auto detection_model = static_cast<YOLOv8 *> (model->model_content);
    detection_model->GetPreprocessor().SetSize({size.height, size.height});
    return MDStatusCode::Success;
}

MDStatusCode md_detection_predict(const MDModel *model, MDImage *image, MDDetectionResults *results) {
    auto cv_image = md_image_to_mat(image);
    fastdeploy::vision::DetectionResult res;
    auto detection_model = static_cast<YOLOv8 *> (model->model_content);
    bool res_status = detection_model->Predict(cv_image, &res);
    if (!res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    auto r_size = res.boxes.size();
    results->size = (int) r_size;
    if (r_size == 0) {
        results->data = nullptr;
        return MDStatusCode::Success;
    }
    results->data = (MDDetectionResult *) malloc(sizeof(MDDetectionResult) * r_size);
    for (int i = 0; i < r_size; ++i) {
        auto box = res.boxes[i];
        results->data[i].box = {int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])};
        results->data[i].score = res.scores[i];
        results->data[i].label_id = res.label_ids[i];
    }
    return MDStatusCode::Success;
}


void md_print_detection_result(const MDDetectionResults *result) {

    for (int i = 0; i < result->size; ++i) {
        std::cout << "box: " << format_rect(result->data[i].box) << " label_id: "
                  << result->data[i].label_id << " score: " << result->data[i].score << std::endl;
    }
}


void md_draw_detection_result(MDImage *image, const MDDetectionResults *result, const char *font_path, int font_size,
                              double alpha, int save_result) {
    cv::Mat cv_image, overlay;
    cv_image = md_image_to_mat(image);
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
        auto box = result->data[i].box;
        auto cv_color = color_map[class_id];
        cv::rectangle(overlay, {box.x, box.y, box.width, box.height}, cv_color, -1);
        auto size = cv::getTextSize(cv::Size(0, 0), std::to_string(class_id), cv::Point(box.x, box.y), font, font_size);
        cv::rectangle(overlay, size, cv_color, -1);
    }
    cv::addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image);
    // 绘制非半透明部分（矩形边框、文字等）
    for (int i = 0; i < result->size; ++i) {
        auto class_id = result->data[i].label_id;
        auto box = result->data[i].box;
        auto cv_color = color_map[class_id];
        cv::rectangle(cv_image, cv::Rect(box.x, box.y, box.width, box.height), cv_color, 1, cv::LINE_AA, 0);
        auto size = cv::getTextSize(cv::Size(0, 0), std::to_string(class_id),
                                    cv::Point(box.x, box.y), font, font_size);
        cv::rectangle(cv_image, size, cv_color, 1, cv::LINE_AA, 0);
        cv::putText(cv_image, std::to_string(class_id), cv::Point(box.x, box.y - 2),
                    cv::Scalar(255 - cv_color[0], 255 - cv_color[1], 255 - cv_color[2]), font, font_size);
    }
    if (save_result) {
        cv::imwrite("vis_result.jpg", cv_image);
    }
}


void md_free_detection_result(MDDetectionResults *result) {
    if (result->size > 0 && result->data != nullptr) {
        result->size = 0;
        free(result->data);
        result->data = nullptr;
    }
}


void md_free_detection_model(MDModel *model) {
    if (model->model_content != nullptr) {
        delete static_cast<YOLOv8 *>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}



