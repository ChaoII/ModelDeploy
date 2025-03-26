//
// Created by aichao on 2025/3/24.
//

#include "csrc/vision/face_gender/seetaface_gender.h"
#include <csrc/core/md_log.h>

namespace modeldeploy::vision::face {
    SeetaFaceGender::SeetaFaceGender(
        const std::string& model_file,
        const modeldeploy::RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = initialize();
    }

    bool SeetaFaceGender::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR("Failed to initialize fastdeploy backend.");
            return false;
        }
        return true;
    }

    bool SeetaFaceGender::predict(const cv::Mat& image, int* age) {
        std::vector<int> ages;
        if (!batch_predict({image}, &ages)) {
            return false;
        }
        if (!ages.empty()) {
            *age = ages[0];
        }
        return true;
    }

    bool SeetaFaceGender::batch_predict(const std::vector<cv::Mat>& images,
                                        std::vector<int>* ages) {
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
        if (!postprocessor_.run(reused_output_tensors_, ages)) {
            MD_LOG_ERROR("Failed to postprocess the inference results by runtime.");
            return false;
        }
        return true;
    }
}
