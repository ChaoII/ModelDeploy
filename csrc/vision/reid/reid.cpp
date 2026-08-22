//
// Created for standalone pedestrian Re-ID (OSNet) model.
//

#include "core/md_log.h"
#include "vision/reid/reid.h"

namespace modeldeploy::vision::reid {
    ReID::ReID(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool ReID::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        setup_processor_backend();
        return true;
    }

    void ReID::setup_processor_backend() {
        // 该处理器使用基于 OpenCV 的 CPU 预处理（直接产出 FP32 NCHW CPU 张量），
        // 无需额外的处理器后端内核分发。
    }

    bool ReID::predict(const ImageData& img,
                       std::vector<ReIdResult>* results,
                       TimerArray* timer) {
        std::vector<ImageData> batch{img};
        std::vector<std::vector<ReIdResult>> tmp;
        if (!batch_predict(batch, &tmp, timer)) {
            return false;
        }
        if (!tmp.empty()) {
            *results = std::move(tmp[0]);
        }
        return true;
    }

    bool ReID::batch_predict(const std::vector<ImageData>& imgs,
                             std::vector<std::vector<ReIdResult>>* results,
                             TimerArray* timer) {
        results->clear();
        results->resize(imgs.size());

        if (timer) timer->pre_timer.start();
        if (!preprocessor_.run(imgs, &reused_input_tensors_)) {
            MD_LOG_ERROR << "Failed to preprocess the input image." << std::endl;
            return false;
        }
        if (timer) timer->pre_timer.stop();

        if (timer) timer->infer_timer.start();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }
        if (timer) timer->infer_timer.stop();

        if (timer) timer->post_timer.start();
        if (!postprocessor_.run(reused_output_tensors_, results)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        if (timer) timer->post_timer.stop();
        return true;
    }

    std::unique_ptr<ReID> ReID::clone() const {
        auto clone_model = std::make_unique<ReID>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }
} // namespace modeldeploy::vision::reid
