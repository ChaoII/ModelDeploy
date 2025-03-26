//
// Created by aichao on 2025/3/24.
//

#include "csrc/vision/face_id/seetaface.h"
#include <csrc/core/md_log.h>

namespace modeldeploy::vision::face {
    SeetaFaceID::SeetaFaceID(
        const std::string& model_file,
        const modeldeploy::RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = initialize();
    }

    bool SeetaFaceID::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR("Failed to initialize fastdeploy backend.");
            return false;
        }
        return true;
    }

    bool SeetaFaceID::predict(const cv::Mat& image, FaceRecognitionResult* result) {
        std::vector<FaceRecognitionResult> results;
        if (!batch_predict({image}, &results)) {
            return false;
        }
        if (!results.empty()) {
            *result = results[0];
        }
        return true;
    }

    bool SeetaFaceID::batch_predict(const std::vector<cv::Mat>& images,
                                  std::vector<FaceRecognitionResult>* results) {
        std::vector<cv::Mat> fd_images = images;
        if (images.size() != 1) {
            MD_LOG_ERROR("Only support batch = 1 now.");
        }
        if (!preprocessor_.run(&fd_images, &reused_input_tensors_)) {
            MD_LOG_ERROR("Failed to preprocess the input image.");
            return false;
        }
        reused_input_tensors_[0].name = get_input_info(0).name;
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR("Failed to inference by runtime.");
            return false;
        }
        if (!postprocessor_.run(reused_output_tensors_, results)) {
            MD_LOG_ERROR("Failed to postprocess the inference results by runtime.");
            return false;
        }
        return true;
    }
}
