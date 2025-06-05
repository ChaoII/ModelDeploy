//
// Created by aichao on 2025/2/20.
//


#include "csrc/core/md_log.h"
#include "csrc/vision/detection/ultralytics_det.h"

#include <chrono>

namespace modeldeploy::vision::detection {
    UltralyticsDet::UltralyticsDet(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option_ = custom_option;
        runtime_option_.model_filepath = model_file;
        initialized_ = initialize();
    }

    bool UltralyticsDet::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        return true;
    }

    bool UltralyticsDet::predict(const cv::Mat& im, std::vector<DetectionResult>* result) {
        std::vector<std::vector<DetectionResult>> results;
        if (!batch_predict({im}, &results)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool UltralyticsDet::batch_predict(const std::vector<cv::Mat>& images,
                                       std::vector<std::vector<DetectionResult>>* results) {
        std::vector<LetterBoxRecord> letter_box_records;
        std::vector<cv::Mat> imgs = images;
        auto _pre_time = std::chrono::high_resolution_clock::now();
        if (!preprocessor_.run(&imgs, &reused_input_tensors_, &letter_box_records)) {
            MD_LOG_ERROR << "Failed to preprocess the input image." << std::endl;
            return false;
        }
        std::cout << "preprocess time: "
            << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - _pre_time).count()
            << " ms" << std::endl;
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        auto _infer_time = std::chrono::high_resolution_clock::now();
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }
        std::cout << "infer time: "
            << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - _infer_time).count()
            << " ms" << std::endl;
        auto _post_time = std::chrono::high_resolution_clock::now();
        if (!postprocessor_.run(reused_output_tensors_, results, letter_box_records)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        std::cout << "post time: "
            << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - _post_time).count()
            << " ms" << std::endl;
        return true;
    }
}
