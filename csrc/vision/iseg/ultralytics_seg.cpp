//
// Created by aichao on 2025/4/14.
//

#include "core/md_log.h"
#include "vision/iseg/ultralytics_seg.h"

namespace modeldeploy::vision::detection {
    UltralyticsSeg::UltralyticsSeg(const std::string& model_file,
                                   const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.model_file = model_file;
        initialized_ = initialize();
    }

    bool UltralyticsSeg::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy backend." << std::endl;
            return false;
        }
        return true;
    }

    bool UltralyticsSeg::predict(const cv::Mat& image, std::vector<InstanceSegResult>* result,
                                 TimerArray* timers) {
        std::vector<std::vector<InstanceSegResult>> results;
        if (!batch_predict({image}, &results, timers)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool UltralyticsSeg::batch_predict(const std::vector<cv::Mat>& images,
                                       std::vector<std::vector<InstanceSegResult>>* results,
                                       TimerArray* timers) {
        std::vector<LetterBoxRecord> ims_info;
        std::vector<cv::Mat> images_ = images;
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(&images_, &reused_input_tensors_, &ims_info)) {
            MD_LOG_ERROR << "Failed to preprocess the input image." << std::endl;
            return false;
        }
        if (timers) timers->pre_timer.stop();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (timers) timers->infer_timer.start();
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }
        if (timers) timers->infer_timer.stop();
        if (timers) timers->post_timer.start();
        if (!postprocessor_.run(reused_output_tensors_, results, ims_info)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        if (timers) timers->post_timer.stop();
        return true;
    }
} // namespace modeldeploy::vision::detection
