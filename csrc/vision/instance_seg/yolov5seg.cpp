//
// Created by aichao on 2025/4/14.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/instance_seg/yolov5seg.h"

namespace modeldeploy::vision::detection {
    YOLOv5Seg::YOLOv5Seg(const std::string& model_file,
                         const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = initialize();
    }

    bool YOLOv5Seg::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy backend." << std::endl;
            return false;
        }
        return true;
    }

    bool YOLOv5Seg::predict(const cv::Mat& image, DetectionResult* result) {
        std::vector<DetectionResult> results;
        if (!batch_predict({image}, &results)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool YOLOv5Seg::batch_predict(const std::vector<cv::Mat>& images, std::vector<DetectionResult>* results) {
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
        if (!postprocessor_.run(reused_output_tensors_, results, ims_info)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        return true;
    }
} // namespace modeldeploy::vision::detection
