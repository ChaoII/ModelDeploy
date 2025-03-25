//
// Created by aichao on 2025/3/24.
//

#include "csrc/vision/faceid/seetaface.h"

#include <csrc/core/md_log.h>

namespace modeldeploy::vision::faceid {
    SeetaFace::SeetaFace(
        const std::string& model_file,
        const modeldeploy::RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = initialize();
    }

    bool SeetaFace::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR("Failed to initialize fastdeploy backend.");
            return false;
        }
        return true;
    }

    bool SeetaFace::predict(const cv::Mat& im,
                            FaceRecognitionResult* result) {
        std::vector<FaceRecognitionResult> results;
        if (!batch_predict({im}, &results)) {
            return false;
        }
        if (!results.empty()) {
            *result = results[0];
        }
        return true;
    }

    bool SeetaFace::batch_predict(const std::vector<cv::Mat>& images,
                                  std::vector<FaceRecognitionResult>* results) {
        std::vector<cv::Mat> fd_images = images;
        if (images.size() != 1) {
            std::cerr << "Only support batch = 1 now." << std::endl;
        }
        if (!preprocessor_.Run(&fd_images, &reused_input_tensors_)) {
            std::cerr << "Failed to preprocess the input image." << std::endl;
            return false;
        }

        reused_input_tensors_[0].name = get_input_info(0).name;
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            std::cerr << "Failed to inference by runtime." << std::endl;
            return false;
        }

        if (!postprocessor_.Run(reused_output_tensors_, results)) {
            std::cerr << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        return true;
    }
}
