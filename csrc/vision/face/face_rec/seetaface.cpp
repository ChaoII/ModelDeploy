//
// Created by aichao on 2025/3/24.
//


#include "core/md_log.h"
#include "vision/face/face_rec/seetaface.h"

namespace modeldeploy::vision::face {
    SeetaFaceID::SeetaFaceID(
        const std::string& model_file,
        const modeldeploy::RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool SeetaFaceID::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldelpoy backend." << std::endl;
            return false;
        }
        return true;
    }

    bool SeetaFaceID::predict(const cv::Mat& image, FaceRecognitionResult* result, TimerArray* timers) {
        std::vector<FaceRecognitionResult> results;
        if (!batch_predict({image}, &results, timers)) {
            return false;
        }
        if (!results.empty()) {
            *result = results[0];
        }
        return true;
    }

    bool SeetaFaceID::batch_predict(const std::vector<cv::Mat>& images,
                                    std::vector<FaceRecognitionResult>* results, TimerArray* timers) {
        std::vector<cv::Mat> fd_images = images;
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(&fd_images, &reused_input_tensors_)) {
            MD_LOG_ERROR << "Failed to preprocess the input image." << std::endl;
            return false;
        }
        if (timers) timers->pre_timer.stop();
        if (timers) timers->infer_timer.start();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
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
