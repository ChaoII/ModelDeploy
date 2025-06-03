//
// Created by aichao on 2025-4-8.
//


#include "csrc/vision.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/face/face_rec_pipeline_capi.h"


MDStatusCode md_create_face_rec_pipeline_model(MDModel* model,
                                               const char* face_det_model_file,
                                               const char* face_rec_model_file,
                                               const int thread_num) {
    const auto face_rec_pipeline_model = new modeldeploy::vision::face::FaceRecognizerPipeline(
        face_det_model_file, face_rec_model_file, thread_num);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(face_rec_pipeline_model->name().c_str());
    model->model_content = face_rec_pipeline_model;
    model->type = MDModelType::FACE;
    if (!face_rec_pipeline_model->is_initialized()) {
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}


MDStatusCode md_face_rec_pipeline_predict(const MDModel* model, MDImage* image, MDFaceRecognizerResults* c_results) {
    if (model->type != MDModelType::FACE) {
        return MDStatusCode::ModelTypeError;
    }
    const auto cv_image = md_image_to_mat(image);
    const auto face_rec_pipeline_model = static_cast<
        modeldeploy::vision::face::FaceRecognizerPipeline*>(model->model_content);

    std::vector<FaceRecognitionResult> results;
    if (const bool res_status = face_rec_pipeline_model->predict(
        cv_image, &results); !res_status) {
        return MDStatusCode::ModelPredictFailed;
    }

    face_recognizer_results_2_c_results(results, c_results);
    return MDStatusCode::Success;
}

void md_print_face_rec_pipeline_result(const MDFaceRecognizerResults* c_results) {
    std::vector<FaceRecognitionResult> results;
    c_results_2_face_recognizer_results(c_results, &results);
    for (auto& result : results) {
        result.display();
    }
}

void md_free_face_rec_pipeline_result(MDFaceRecognizerResults* c_results) {
    if (c_results->size != 0 && c_results->data != nullptr) {
        for (int i = 0; i < c_results->size; i++) {
            delete[] c_results->data[i].embedding;
            c_results->data[i].embedding = nullptr;
            c_results->data[i].size = 0;
        }
        delete[] c_results->data;
        c_results->data = nullptr;
        c_results->size = 0;
    }
}


void md_free_face_rec_pipeline_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::face::FaceRecognizerPipeline*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
