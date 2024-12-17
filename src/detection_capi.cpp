//
// Created by AC on 2024-12-17.
//
#include "detection_capi.h"
#include "utils.h"
#include "fastdeploy/vision.h"

using YOLOv8 = fastdeploy::vision::detection::YOLOv8;

DetectionModelHandle create_detection_model(const char *model_dir, int thread_num) {
    fastdeploy::SetLogger(false);
    fastdeploy::RuntimeOption option;
    option.UseOrtBackend();
    option.SetCpuThreadNum(thread_num);
    auto detection_model = new YOLOv8(model_dir, "", option);
    if (!(detection_model->Initialized())) {
        std::cerr << "Failed to initialize detection model." << std::endl;
        return nullptr;
    }
    return detection_model;
}

void set_detection_input_size(DetectionModelHandle model, WSize size) {
    auto detection_model = static_cast<YOLOv8 *> (model);
    detection_model->GetPreprocessor().SetSize({size.height, size.height});
}

WDetectionResult *detection_predict(DetectionModelHandle model, WImage *image,
                                    int draw_result, WColor color, double alpha, int is_save_result) {
    auto cv_image = wimage_to_mat(image);
    fastdeploy::vision::DetectionResult res;
    auto detection_model = static_cast<YOLOv8 *> (model);
    bool res_status = detection_model->Predict(cv_image, &res);
    if (!res_status) {
        return nullptr;
    }
    auto r_size = res.boxes.size();
    auto out_data = (WDetectionResult *) malloc(sizeof(WDetectionResult));
    out_data->size = r_size;
    out_data->boxes = (WRect *) calloc(r_size, sizeof(WRect));
    out_data->scores = (float *) calloc(r_size, sizeof(float));
    out_data->label_ids = (int *) calloc(r_size, sizeof(int));
    for (int i = 0; i < r_size; ++i) {
        auto box = res.boxes[i];
        out_data->boxes[i] = {int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])};
        out_data->scores[i] = res.scores[i];
        out_data->label_ids[i] = res.label_ids[i];
    }
    if (is_save_result > 0) {
        cv::imwrite("vis_result.jpg", cv_image);
    }
    return out_data;
}


void print_detection_result(WDetectionResult *result) {
    if (!result) return;
    for (int i = 0; i < result->size; ++i) {
        std::cout << "box: " << format_rect(result->boxes[i]) << " label_id: "
                  << result->label_ids[i] << " score: " << result->scores[i] << std::endl;
    }
}

void free_detection_result(WDetectionResult *result) {
    if (!result) return;
    free(result->boxes);
    free(result->scores);
    free(result->label_ids);
    result->size = 0;
}

void free_ocr_model(DetectionModelHandle model) {
    delete static_cast<YOLOv8 *>(model);
}