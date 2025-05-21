//
// Created by aichao on 2025/2/24.
//

#include "yolov5cls.h"

#include <csrc/core/md_log.h>

namespace modeldeploy::vision::classification {
    YOLOv5Cls::YOLOv5Cls(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = initialize();
    }

    bool YOLOv5Cls::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        return true;
    }

    bool YOLOv5Cls::Predict(const cv::Mat& im, ClassifyResult* result) {
        std::vector<ClassifyResult> results;
        if (!BatchPredict({im}, &results)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool YOLOv5Cls::BatchPredict(const std::vector<cv::Mat>& images, std::vector<ClassifyResult>* results) {
        std::vector<std::map<std::string, std::array<float, 2>>> ims_info;
        std::vector<cv::Mat> imgs = images;
        if (!preprocessor_.Run(&imgs, &reused_input_tensors_, &ims_info)) {
            MD_LOG_ERROR << "Failed to preprocess the input image." << std::endl;
            return false;
        }

        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }

        if (!postprocessor_.Run(reused_output_tensors_, results, ims_info)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        return true;
    }
} // namespace classification
