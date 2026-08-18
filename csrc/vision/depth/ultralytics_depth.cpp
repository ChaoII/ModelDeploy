//
// Created by aichao on 2026/8/13.
//

#include "core/md_log.h"
#include "vision/depth/ultralytics_depth.h"

namespace modeldeploy::vision::detection {
    UltralyticsDepth::UltralyticsDepth(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool UltralyticsDepth::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool UltralyticsDepth::predict(const ImageData& image, DepthResult* result, TimerArray* timers) {
        // NV12/NV21 或设备帧 → 零拷贝 NV12 预处路径；否则打包路径
        if (image.format() == MdImageType::NV12 || image.format() == MdImageType::NV21 ||
            image.device() != Device::CPU) {
            return predict_single_nv12(image, result, nullptr, timers);
        }
        std::vector<DepthResult> results;
        if (!batch_predict({image}, &results, timers)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool UltralyticsDepth::predict_single_nv12(const ImageData& frame,
                                               DepthResult* result,
                                               LetterBoxRecord* letter_box_record,
                                               TimerArray* timers) {
        if (frame.plane_count() < 2) {
            MD_LOG_ERROR << "NV12 input requires 2 planes." << std::endl;
            return false;
        }
        const auto& y_plane = frame.plane(0);
        const auto& uv_plane = frame.plane(1);
        reused_input_tensors_.resize(1);
        std::vector<LetterBoxRecord> lbr(1);
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(y_plane.data, uv_plane.data, {frame.width(), frame.height()},
                               y_plane.step, uv_plane.step, &reused_input_tensors_[0], &lbr[0],
                               frame.device())) {
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
        std::vector<DepthResult> depth_results;
        if (!postprocessor_.run(reused_output_tensors_, &depth_results, lbr)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        if (timers) timers->post_timer.stop();
        if (!depth_results.empty()) {
            *result = std::move(depth_results[0]);
        }
        if (letter_box_record) {
            *letter_box_record = lbr[0];
        }
        return true;
    }

    bool UltralyticsDepth::batch_predict(const std::vector<ImageData>& images,
                                         std::vector<DepthResult>* results, TimerArray* timers) {
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

    std::unique_ptr<UltralyticsDepth> UltralyticsDepth::clone() const {
        auto clone_model = std::make_unique<UltralyticsDepth>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }

    bool UltralyticsDepth::draw_result(ImageData& frame, const DepthResult& result,
                                       double threshold) {
        (void) result;
        (void) threshold;
        return !frame.empty();
    }
}