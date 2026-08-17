//
// Created by aichao on 2025/4/14.
//

#include "core/md_log.h"
#include "vision/iseg/ultralytics_seg.h"

#include <string>

namespace modeldeploy::vision::detection {
    UltralyticsSeg::UltralyticsSeg(const std::string& model_file,
                                   const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool UltralyticsSeg::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy backend." << std::endl;
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool UltralyticsSeg::predict(const ImageData& image, std::vector<InstanceSegResult>* result,
                                 TimerArray* timers) {
        std::vector<std::vector<InstanceSegResult>> results;
        if (!batch_predict({image}, &results, timers)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool UltralyticsSeg::batch_predict(const std::vector<ImageData>& images,
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

    std::unique_ptr<UltralyticsSeg> UltralyticsSeg::clone() const {
        auto clone_model = std::make_unique<UltralyticsSeg>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }

    bool UltralyticsSeg::predict_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                             int width, int height, int step_y, int step_uv,
                             std::vector<InstanceSegResult>* result, LetterBoxRecord* letter_box_record,
                             ImageData* out_frame,
                             Device src_device, TimerArray* timers) {
        if (!src_y || !src_uv || !result) return false;
        if (out_frame) {
            *out_frame = ImageData::from_device_planes(
                const_cast<uint8_t*>(src_y), const_cast<uint8_t*>(src_uv),
                width, height, step_y, step_uv, src_device);
        }
        std::vector<LetterBoxRecord> lbr(1);
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(src_y, src_uv, {width, height}, step_y, step_uv,
                               &reused_input_tensors_[0], &lbr[0],
                               src_device)) {
            MD_LOG_ERROR << "Failed to preprocess the NV12 input." << std::endl;
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
        std::vector<std::vector<InstanceSegResult>> batch_results;
        if (!postprocessor_.run(reused_output_tensors_, &batch_results, lbr)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        if (timers) timers->post_timer.stop();
        if (!batch_results.empty()) {
            *result = std::move(batch_results[0]);
        }
        if (letter_box_record) {
            *letter_box_record = lbr[0];
        }
        return true;
    }

    bool UltralyticsSeg::draw_result(ImageData& frame, const std::vector<InstanceSegResult>& result,
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
} // namespace modeldeploy::vision::detection