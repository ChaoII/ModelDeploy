//
// Created by aichao on 2025/06/2.
//

#include <opencv2/opencv.hpp>

#include <cmath>
#include <string>

#include "core/md_log.h"
#include "vision/pose/ultralytics_pose.h"

namespace modeldeploy::vision::detection {
    UltralyticsPose::UltralyticsPose(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool UltralyticsPose::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool UltralyticsPose::predict(const ImageData& image, std::vector<KeyPointsResult>* result, TimerArray* timers) {
        // NV12/NV21 或设备帧 → 零拷贝 NV12 预处路径；否则打包路径
        if (image.format() == MdImageType::NV12 || image.format() == MdImageType::NV21 ||
            image.device() != Device::CPU) {
            return predict_single_nv12(image, result, nullptr, timers);
        }
        std::vector<std::vector<KeyPointsResult>> results;
        if (!batch_predict({image}, &results, timers)) {
            return false;
        }
        *result = results[0];
        return true;
    }

    bool UltralyticsPose::predict_single_nv12(const ImageData& frame,
                                              std::vector<KeyPointsResult>* result,
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
        std::vector<std::vector<KeyPointsResult>> batch_results;
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

    bool UltralyticsPose::batch_predict(const std::vector<ImageData>& images,
                                        std::vector<std::vector<KeyPointsResult>>* results, TimerArray* timers) {
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

    std::unique_ptr<UltralyticsPose> UltralyticsPose::clone() const {
        auto clone_model = std::make_unique<UltralyticsPose>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }

    bool UltralyticsPose::draw_result(ImageData& frame, const std::vector<KeyPointsResult>& result,
                                      double threshold) {
        if (frame.empty()) return false;
        auto* backend = preprocessor_.get_processor_backend().get();
        if (!backend) return false;
        for (const auto& r : result) {
            if (r.score < threshold) continue;
            const Rect2f& box = r.box;
            if (!backend->draw_rect_nv12(frame, box.x, box.y, box.width, box.height,
                                         148, 0, 211, 2)) return false;
            for (const auto& kp : r.keypoints) {
                const std::vector<Point3f> single{kp};
                backend->draw_points_nv12(frame, single, 0, 255, 255, 3);
            }
        }
        return true;
    }
}