//
// Created by aichao on 2025/6/10.
//


#include "csrc/core/md_log.h"
#include "csrc/vision/lpr/lpr_rec/lpr_rec.h"


namespace modeldeploy::vision::lpr {
    LprRecognizer::LprRecognizer(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.model_file = model_file;
        initialized_ = initialize();
    }

    bool LprRecognizer::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize fastdeploy backend." << std::endl;
            return false;
        }
        return true;
    }

    bool LprRecognizer::predict(const cv::Mat& image, LprResult* result, TimerArray* timers) {
        std::vector<LprResult> results;
        if (!batch_predict({image}, &results, timers)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool LprRecognizer::batch_predict(const std::vector<cv::Mat>& images,
                                      std::vector<LprResult>* results,
                                      TimerArray* timers) {
        std::vector<cv::Mat> _images = images;
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(&_images, &reused_input_tensors_)) {
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
        if (!postprocessor_.run(reused_output_tensors_, results)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        if (timers) timers->post_timer.stop();
        return true;
    }
}
