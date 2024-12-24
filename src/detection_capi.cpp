//
// Created by AC on 2024-12-17.
//
#include "detection_capi.h"
#include "utils.h"
#include "fastdeploy/vision.h"

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

StatusCode detection_predict(WModel *model, WDetectionResults *results, WImage *image,
                             int draw_result, WColor color, double alpha, int is_save_result) {
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
    auto vis_image = fastdeploy::vision::VisDetection(cv_image, res, 0.3);
    cv::imwrite("asd.jpg", vis_image);
    if (is_save_result > 0) {
        cv::imwrite("vis_result.jpg", cv_image);
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