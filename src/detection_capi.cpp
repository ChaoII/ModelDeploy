//
// Created by AC on 2024-12-17.
//
#include "detection_capi.h"
#include "utils_internal.h"
#include "utils.h"
#include "fastdeploy/vision.h"
#include <map>

using YOLOv8 = fastdeploy::vision::detection::YOLOv8;
using PPYOLOE = fastdeploy::vision::detection::PPYOLOE;

StatusCode create_detection_model(WModel *model, const char *model_dir, int thread_num, ModelFormat model_format) {
    if (!model) {
        return StatusCode::MemoryAllocatedFailed;
    }
    fastdeploy::SetLogger(false);
    fastdeploy::RuntimeOption option;
    option.UseOrtBackend();
    option.SetCpuThreadNum(thread_num);
    fastdeploy::FastDeployModel *detection_model;
    if (model_format == ModelFormat::ONNX) {
        detection_model = new YOLOv8(model_dir, "", option);
    } else {
        std::string model_file = std::string(model_dir) + "/model.pdmodel";
        std::string param_file = std::string(model_dir) + "/model.pdiparams";
        std::string config_file = std::string(model_dir) + "/infer_cfg.yml";
        detection_model = new PPYOLOE(model_file, param_file, config_file, option);
    }
    if (!detection_model->Initialized()) {
        std::cerr << "model initial failed" << std::endl;
        return StatusCode::ModelInitializeFailed;
    }
    auto model_name = detection_model->ModelName();
    model->format = model_format;
    model->model_name = (char *) malloc((detection_model->ModelName().size() + 1) * sizeof(char));
    memcpy(model->model_name, model_name.c_str(), model_name.size() + 1);
    model->model_content = detection_model;
    model->type = ModelType::Detection;
    return StatusCode::Success;
}

StatusCode set_detection_input_size(WModel *model, WSize size) {
    if (model->format == ModelFormat::PaddlePaddle) return StatusCode::CallError;
    auto detection_model = static_cast<YOLOv8 *> (model->model_content);
    detection_model->GetPreprocessor().SetSize({size.height, size.height});
    return StatusCode::Success;
}

StatusCode detection_predict(WModel *model, WImage *image, WDetectionResults *results) {
    auto cv_image = wimage_to_mat(image);
    fastdeploy::vision::DetectionResult res;
    auto detection_model = static_cast<YOLOv8 *> (model->model_content);
    bool res_status = detection_model->Predict(cv_image, &res);
    if (!res_status) {
        return StatusCode::ModelPredictFailed;
    }
    auto r_size = res.boxes.size();
    results->size = r_size;
    if (r_size == 0) {
        results->data = nullptr;
        return StatusCode::Success;
    }
    results->data = (WDetectionResult *) malloc(sizeof(WDetectionResult) * r_size);
    for (int i = 0; i < r_size; ++i) {
        auto box = res.boxes[i];
        results->data[i].box = {int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])};
        results->data[i].score = res.scores[i];
        results->data[i].label_id = res.label_ids[i];
    }
    return StatusCode::Success;
}


void print_detection_result(WDetectionResults *result) {
    if (!result) return;
    for (int i = 0; i < result->size; ++i) {
        std::cout << "box: " << format_rect(result->data[i].box) << " label_id: "
                  << result->data[i].label_id << " score: " << result->data[i].score << std::endl;
    }
}

void free_detection_result(WDetectionResults *result) {
    if (result != nullptr) {
        if (result->size > 0 && result->data != nullptr) {
            free(result->data);
        }
    }
}


void free_detection_model(WModel *model) {
    if (!model) return;
    free(model->model_name);
    delete static_cast<YOLOv8 *>(model->model_content);
}


void draw_detection_result(WImage *image, WDetectionResults *result, const char *font_path, int font_size,
                           double alpha, int save_result) {
    cv::Mat cv_image, overlay;
    cv_image = wimage_to_mat(image);
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
