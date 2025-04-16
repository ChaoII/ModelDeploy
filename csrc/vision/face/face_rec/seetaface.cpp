//
// Created by aichao on 2025/3/24.
//


#include "csrc/core/md_log.h"
#include "csrc/vision/face/face_rec/seetaface.h"

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
            MD_LOG_ERROR << "Failed to initialize modeldelpoy backend." << std::endl;
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
            MD_LOG_ERROR << "Only support batch = 1 now." << std::endl;
        }
        if (!preprocessor_.run(&fd_images, &reused_input_tensors_)) {
            MD_LOG_ERROR << "Failed to preprocess the input image." << std::endl;
            return false;
        }
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }
        if (!postprocessor_.run(reused_output_tensors_, results)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        return true;
    }
}
