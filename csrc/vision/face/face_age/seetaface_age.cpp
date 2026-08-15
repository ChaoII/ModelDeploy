//
// Created by aichao on 2025/3/24.
//

#include "core/md_log.h"
#include "vision/face/face_age/seetaface_age.h"

namespace modeldeploy::vision::face {
    SeetaFaceAge::SeetaFaceAge(
        const std::string& model_file,
        const modeldeploy::RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool SeetaFaceAge::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        return true;
    }

    std::unique_ptr<SeetaFaceAge> SeetaFaceAge::clone() const {
        auto clone_model = std::make_unique<SeetaFaceAge>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }
bool SeetaFaceAge::predict(const ImageData& image, int* age, TimerArray* timers) {
        std::vector<int> ages;
        if (!batch_predict({image}, &ages, timers)) {
            return false;
        }
        if (!ages.empty()) {
            *age = ages[0];
        }
        return true;
    }

    bool SeetaFaceAge::batch_predict(const std::vector<ImageData>& images,
                                     std::vector<int>* ages, TimerArray* timers) {
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(images, &reused_input_tensors_)) {
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
        if (!postprocessor_.run(reused_output_tensors_, ages)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        if (timers) timers->post_timer.stop();
        return true;
    }
}

