#include "core/md_log.h"
#include "vision/sam/fastsam.h"

namespace modeldeploy::vision::seg {
    FastSam::FastSam(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool FastSam::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy backend." << std::endl;
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool FastSam::predict(const ImageData& image, std::vector<InstanceSegResult>* result,
                          TimerArray* timers) {
        std::vector<std::vector<InstanceSegResult>> results;
        if (!batch_predict({image}, &results, timers)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool FastSam::batch_predict(const std::vector<ImageData>& images,
                                std::vector<std::vector<InstanceSegResult>>* results,
                                TimerArray* timers) {
        std::vector<LetterBoxRecord> ims_info;
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(images, &reused_input_tensors_, &ims_info)) {
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

    std::unique_ptr<FastSam> FastSam::clone() const {
        auto clone_model = std::make_unique<FastSam>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }

    bool FastSam::draw_result(ImageData& frame, const std::vector<InstanceSegResult>& result,
                              double threshold) {
        if (frame.empty()) return false;
        auto* backend = preprocessor_.get_processor_backend().get();
        if (!backend) return false;
        for (const auto& r : result) {
            if (r.score < threshold) continue;
            const Rect2f& box = r.box;
            if (!backend->draw_rect_nv12(frame, box.x, box.y, box.width, box.height,
                                         0, 165, 255, 2)) return false;
            const std::string label = std::to_string(r.label_id) + " " + std::to_string(r.score);
            backend->draw_text_nv12(frame, box.x, box.y - 16, label, 255, 255, 255, 1);
        }
        return true;
    }
} // namespace modeldeploy::vision::seg
