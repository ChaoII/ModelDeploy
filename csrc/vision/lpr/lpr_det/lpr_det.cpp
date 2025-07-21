//
// Created by aichao on 2025/6/10.
//


#include "core/md_log.h"
#include "vision/lpr/lpr_det/lpr_det.h"


namespace modeldeploy::vision::lpr {
    LprDetection::LprDetection(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool LprDetection::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        return true;
    }

    bool LprDetection::predict(const ImageData& image, std::vector<DetectionLandmarkResult>* result,
                               TimerArray* timers) {
        std::vector<std::vector<DetectionLandmarkResult>> results;
        if (!batch_predict({image}, &results, timers)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool LprDetection::batch_predict(const std::vector<ImageData>& images,
                                     std::vector<std::vector<DetectionLandmarkResult>>* results,
                                     TimerArray* timers) {
        std::vector<LetterBoxRecord> letter_box_records;
        std::vector<ImageData> _images = images;
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(&_images, &reused_input_tensors_, &letter_box_records)) {
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
        if (!postprocessor_.run(reused_output_tensors_, results, letter_box_records)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        if (timers) timers->post_timer.stop();
        return true;
    }
}
