//
// Created by aichao on 2025/5/30.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/obb/ultralytics_obb.h"

namespace modeldeploy::vision::detection {
    UltralyticsObb::UltralyticsObb(const std::string& model_file,
                                   const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = initialize();
    }

    bool UltralyticsObb::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy backend." << std::endl;
            return false;
        }
        return true;
    }

    bool UltralyticsObb::predict(const cv::Mat& image, DetectionResult* result) {
        std::vector<DetectionResult> results;
        if (!batch_predict({image}, &results)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool UltralyticsObb::batch_predict(const std::vector<cv::Mat>& images, std::vector<DetectionResult>* results) {
        std::vector<std::map<std::string, std::array<float, 2>>> ims_info;
        std::vector<cv::Mat> images_ = images;
        if (!preprocessor_.run(&images_, &reused_input_tensors_, &ims_info)) {
            MD_LOG_ERROR << "Failed to preprocess the input image." << std::endl;
            return false;
        }
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }
        std::cout<<"reused_output_tensors_[0].shape[0]= " <<reused_output_tensors_[0].shape()[0]<<std::endl;
        std::cout<<"reused_output_tensors_[0].shape[1]= " <<reused_output_tensors_[0].shape()[1]<<std::endl;
        std::cout<<"reused_output_tensors_[0].shape[2]= " <<reused_output_tensors_[0].shape()[2]<<std::endl;
        if (!postprocessor_.run(reused_output_tensors_, results, ims_info)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        return true;
    }
} // namespace modeldeploy::vision::detection
