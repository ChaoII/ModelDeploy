//
// Created by aichao on 2025/2/20.
//


#include "core/md_log.h"
#include "vision/detection/ultralytics_det.h"
#include "vision/utils.h"

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

namespace modeldeploy::vision::detection {
    UltralyticsDet::UltralyticsDet(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool UltralyticsDet::initialize() {
        if (!init_runtime()) {
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool UltralyticsDet::predict(const ImageData& image, std::vector<DetectionResult>* result,
                                 TimerArray* timers) {
        std::vector<std::vector<DetectionResult>> results;
        if (!batch_predict({image}, &results, timers)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool UltralyticsDet::batch_predict(const std::vector<ImageData>& images,
                                       std::vector<std::vector<DetectionResult>>* results,
                                       TimerArray* timers) {
        std::vector<LetterBoxRecord> letter_box_records;
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(images, &reused_input_tensors_, &letter_box_records)) {
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

    bool UltralyticsDet::predict_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                                      int width, int height, int step_y, int step_uv,
                                      std::vector<DetectionResult>* result,
                                      LetterBoxRecord* letter_box_record,
                                      ImageData* out_frame,
                                      Device src_device, TimerArray* timers) {
        if (!src_y || !src_uv || !result) return false;
        if (out_frame) {
            *out_frame = ImageData::from_device_planes(
                const_cast<uint8_t*>(src_y), const_cast<uint8_t*>(src_uv),
                width, height, step_y, step_uv, src_device);
        }
        reused_input_tensors_.resize(1);
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(src_y, src_uv, {width, height}, step_y, step_uv,
                               &reused_input_tensors_[0], letter_box_record,
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
        std::vector<std::vector<DetectionResult>> results;
        if (!postprocessor_.run(reused_output_tensors_, &results, {*letter_box_record})) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        if (timers) timers->post_timer.stop();
        if (!results.empty()) *result = std::move(results[0]);
        return true;
    }

    bool UltralyticsDet::draw_result(ImageData& frame, const std::vector<DetectionResult>& result,
                                     double threshold) {
        if (frame.empty()) return false;
        auto* backend = preprocessor_.get_processor_backend().get();
        if (!backend) return false;
        for (const auto& r : result) {
            if (r.score < threshold) continue;
            const Rect2f& box = r.box;
            if (!backend->draw_rect_nv12(frame, box.x, box.y, box.width, box.height,
                                         255, 0, 0, 2)) return false;
            const std::string label = std::to_string(r.label_id) + " " + std::to_string(r.score);
            backend->draw_text_nv12(frame, box.x, box.y - 16, label, 255, 255, 255, 1);
        }
        return true;
    }

    std::unique_ptr<UltralyticsDet> UltralyticsDet::clone() const {
        auto clone_model = std::make_unique<UltralyticsDet>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }
}
