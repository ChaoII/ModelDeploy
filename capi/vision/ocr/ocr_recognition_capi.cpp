//
// Created by aichao on 2025/3/3.
//


#include <string>
#include "csrc/vision.h"
#include "csrc/core/md_log.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/ocr/ocr_recognition_capi.h"


MDStatusCode md_create_ocr_recognition_model(MDModel* model, const char* model_path,
                                             const char* dict_path, const int thread_num) {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(thread_num);
    const auto ocr_rec_model = new modeldeploy::vision::ocr::Recognizer(model_path, dict_path, option);
    model->format = MDModelFormat::ONNX;
    model->model_content = ocr_rec_model;
    model->model_name = strdup(ocr_rec_model->name().c_str());
    model->type = MDModelType::OCR;
    return MDStatusCode::Success;
}


MDStatusCode md_ocr_recognition_model_predict(const MDModel* model, const MDImage* image, MDOCRResult* result) {
    if (model->type != MDModelType::OCR) {
        MD_LOG_ERROR << "Model type is not OCR" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto cv_image = md_image_to_mat(image);
    const auto ocr_rec_model = static_cast<modeldeploy::vision::ocr::Recognizer*>(model->model_content);
    std::string text;
    float score;
    if (const bool res_status = ocr_rec_model->predict(cv_image, &text, &score); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }
    result->box.size = 0;
    result->box.data = nullptr;
    result->text = strdup(text.c_str());
    result->score = score;
    return MDStatusCode::Success;
}

MDStatusCode md_ocr_recognition_model_predict_batch(
    const MDModel* model, const MDImage* image, const int batch_size,
    const MDPolygon* polygon, const int size, MDOCRResults* results) {
    if (model->type != MDModelType::OCR) {
        MD_LOG_ERROR << "Model type is not OCR" << std::endl;
        return MDStatusCode::ModelTypeError;
    }
    const auto ocr_rec_model = static_cast<modeldeploy::vision::ocr::Recognizer*>(model->model_content);
    const auto cv_image = md_image_to_mat(image);
    std::vector<std::string> text_ptr;
    std::vector<float> rec_scores_ptr;
    std::vector<int> indices;
    std::vector<cv::Mat> image_list;
    image_list.reserve(size);
    indices.reserve(size);
    for (int i = 0; i < size; i++) {
        indices.push_back(i);
        image_list.push_back(get_rotate_crop_image(cv_image, &polygon[i]));
    }
    int result_index = 0;
    results->size = size;
    results->data = static_cast<MDOCRResult*>(malloc(sizeof(MDOCRResult) * size));
    for (int start_index = 0; start_index < size; start_index += batch_size) {
        const int end_index = std::min(start_index + batch_size, size);
        if (!ocr_rec_model->batch_predict(image_list, &text_ptr, &rec_scores_ptr,
                                          start_index, end_index, indices)) {
            MD_LOG_ERROR << "There's error while recognizing image in OCR recognition model." << std::endl;
        }
        for (int i = 0; i < text_ptr.size(); i++) {
            // 注意在识别模型中不会返回box信息
            results->data[result_index].box.size = 0;
            results->data[result_index].box.data = nullptr;
            results->data[result_index].text = strdup(text_ptr[i].c_str());
            results->data[result_index].score = rec_scores_ptr[i];
            result_index++;
        }
    }
    return MDStatusCode::Success;
}


void md_free_ocr_recognition_result(MDOCRResult* result) {
    // 该方法未给data分配堆内存直接赋nullptr避免野指针，悬垂指针
    if (result->box.data) {
        result->box.data = nullptr;
    }
    if (result->text) {
        free(result->text);
    }
}

void md_free_ocr_recognition_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::ocr::Recognizer*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
