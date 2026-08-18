//
// Created by aichao on 2025/5/30.
//

#include "core/md_log.h"
#include "vision/obb/ultralytics_obb.h"

#include <cmath>
#include <string>
#include <vector>

namespace modeldeploy::vision::detection {
    UltralyticsObb::UltralyticsObb(const std::string& model_file,
                                   const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool UltralyticsObb::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy backend." << std::endl;
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool UltralyticsObb::predict(const ImageData& image, std::vector<ObbResult>* result, TimerArray* timers) {
        // NV12/NV21 或设备帧 → 零拷贝 NV12 预处路径；否则打包路径
        if (image.format() == MdImageType::NV12 || image.format() == MdImageType::NV21 ||
            image.device() != Device::CPU) {
            return predict_single_nv12(image, result, nullptr, timers);
        }
        std::vector<std::vector<ObbResult>> results;
        if (!batch_predict({image}, &results, timers)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool UltralyticsObb::predict_single_nv12(const ImageData& frame,
                                             std::vector<ObbResult>* result,
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
        std::vector<std::vector<ObbResult>> batch_results;
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

    bool UltralyticsObb::batch_predict(const std::vector<ImageData>& images,
                                       std::vector<std::vector<ObbResult>>* results, TimerArray* timers) {
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

    std::unique_ptr<UltralyticsObb> UltralyticsObb::clone() const {
        auto clone_model = std::make_unique<UltralyticsObb>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }

    bool UltralyticsObb::predict_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                             int width, int height, int step_y, int step_uv,
                             std::vector<ObbResult>* result, LetterBoxRecord* letter_box_record,
                             ImageData* out_frame,
                             Device src_device, TimerArray* timers) {
        if (!src_y || !src_uv || !result) return false;
        ImageData frame = ImageData::from_device_planes(
            const_cast<uint8_t*>(src_y), const_cast<uint8_t*>(src_uv),
            width, height, step_y, step_uv, src_device);
        if (out_frame) {
            *out_frame = frame;
        }
        return predict_single_nv12(frame, result, letter_box_record, timers);
    }

    bool UltralyticsObb::draw_result(ImageData& frame, const std::vector<ObbResult>& result,
                                     double threshold) {
        if (frame.empty()) return false;
        auto* backend = preprocessor_.get_processor_backend().get();
        if (!backend) return false;
        for (const auto& r : result) {
            if (r.score < threshold) continue;
            const RotatedRect& rr = r.rotated_box;
            const float rad = rr.angle * 3.14159265f / 180.0f;
            const float cos_a = std::cos(rad), sin_a = std::sin(rad);
            const float hw = rr.width * 0.5f, hh = rr.height * 0.5f;
            std::vector<Point2f> quad(4);
            const float dx[4] = { hw, -hw, -hw, hw };
            const float dy[4] = { hh, hh, -hh, -hh };
            for (int i = 0; i < 4; ++i) {
                quad[i] = Point2f(rr.xc + dx[i] * cos_a - dy[i] * sin_a,
                                  rr.yc + dx[i] * sin_a + dy[i] * cos_a);
            }
            if (!backend->draw_polygon_nv12(frame, quad, 255, 215, 0, 2)) return false;
            const std::string label = std::to_string(r.label_id) + " " + std::to_string(r.score);
            backend->draw_text_nv12(frame, rr.xc, rr.yc, label, 255, 255, 255, 1);
        }
        return true;
    }
} // namespace modeldeploy::vision::detection