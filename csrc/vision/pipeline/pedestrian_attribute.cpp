//
// Created by aichao on 2025/12/13.
//

#include "core/md_log.h"
#include "vision/pipeline/pedestrian_attribute.h"

namespace modeldeploy::vision::pipeline {
    PedestrianAttribute::PedestrianAttribute(const std::string& det_model_path,
                                             const std::string& mlc_model_path,
                                             const RuntimeOption& option) {
        detector_ = std::make_shared<detection::UltralyticsDet>(det_model_path, option);
        classifier_ = std::make_shared<classification::Classification>(mlc_model_path, option);
        classifier_->get_preprocessor().disable_center_crop();
        classifier_->get_postprocessor().set_multi_label(true);
    }

    PedestrianAttribute::~PedestrianAttribute() = default;

    bool PedestrianAttribute::is_initialized() const {
        if (detector_ != nullptr && !detector_->is_initialized()) {
            return false;
        }
        if (classifier_ != nullptr && !classifier_->is_initialized()) {
            return false;
        }
        return true;
    }

    void PedestrianAttribute::set_det_threshold(const float threshold) const {
        detector_->get_postprocessor().set_conf_threshold(threshold);
    }

    float PedestrianAttribute::get_det_threshold() const {
        return detector_->get_postprocessor().get_conf_threshold();
    }

    void PedestrianAttribute::set_det_input_size(const std::vector<int>& size) const {
        detector_->get_preprocessor().set_size(size);
    }

    [[nodiscard]] std::vector<int> PedestrianAttribute::get_det_input_size() const {
        return detector_->get_preprocessor().get_size();
    }

    void PedestrianAttribute::set_cls_input_size(const std::vector<int>& size) const {
        classifier_->get_preprocessor().set_size(size);
    }

    [[nodiscard]] std::vector<int> PedestrianAttribute::get_cls_input_size() const {
        return classifier_->get_preprocessor().get_size();
    }


    bool PedestrianAttribute::set_cls_batch_size(const int cls_batch_size) {
        if (cls_batch_size < -1 || cls_batch_size == 0) {
            MD_LOG_ERROR << "batch_size > 0 or batch_size == -1." << std::endl;
            return false;
        }
        cls_batch_size_ = cls_batch_size;
        return true;
    }

    int PedestrianAttribute::get_cls_batch_size() const { return cls_batch_size_; }

    std::shared_ptr<detection::UltralyticsDet> PedestrianAttribute::get_detector() {
        return detector_;
    }

    std::shared_ptr<classification::Classification> PedestrianAttribute::get_classifier() {
        return classifier_;
    }

    bool PedestrianAttribute::predict(const ImageData& image, std::vector<AttributeResult>* result,
                                      TimerArray* timers) {
        std::vector<std::vector<AttributeResult>> batch_result(1);
        if (const bool success = batch_predict({image}, &batch_result, timers); !success) {
            return success;
        }
        *result = std::move(batch_result[0]);
        return true;
    }

    bool PedestrianAttribute::batch_predict(
        const std::vector<ImageData>& images,
        std::vector<std::vector<AttributeResult>>* batch_result, TimerArray* timers) {
        batch_result->clear();
        batch_result->resize(images.size());
        std::vector<std::vector<DetectionResult>> batch_detection_results(images.size());
        if (timers) {
            timers->pre_timer.push_back(0);
            timers->post_timer.push_back(0);
            timers->infer_timer.start();
        }
        if (!detector_->batch_predict(images, &batch_detection_results)) {
            MD_LOG_ERROR << "There's error while detecting image in PedestrianAttribute." << std::endl;
            return false;
        }
        for (int i_batch = 0; i_batch < images.size(); ++i_batch) {
            std::vector<AttributeResult>& attr_result = (*batch_result)[i_batch];
            // Get cropped images by detection result
            const std::vector<DetectionResult>& det_result = batch_detection_results[i_batch];
            const ImageData& img = images[i_batch];
            std::vector<ImageData> image_list;
            if (det_result.empty()) {
                MD_LOG_WARN << "There's no detection result in image " << i_batch << "." << std::endl;
                return true;
            }
            image_list.resize(det_result.size());
            attr_result.resize(det_result.size());
            for (size_t i_box = 0; i_box < det_result.size(); ++i_box) {
                image_list[i_box] = img.crop(det_result[i_box].box);
                // attr_result[i_box].box = det_result[i_box].box;
                // attr_result[i_box].box_score = det_result[i_box].score;
                // attr_result[i_box].box_label_id = det_result[i_box].label_id;
            }
            for (size_t start_index = 0; start_index < image_list.size(); start_index += cls_batch_size_) {
                const size_t end_index = std::min(start_index + cls_batch_size_, image_list.size());
                std::vector<ClassifyResult> mlc_results;
                std::vector sub_image_list(image_list.begin() + static_cast<int>(start_index),
                                           image_list.begin() + static_cast<int>(end_index));
                if (!classifier_->batch_predict(sub_image_list, &mlc_results)) {
                    MD_LOG_ERROR << "There's error while recognizing image in PedestrianAttribute." << std::endl;
                    return false;
                }
                for (size_t i_img = start_index; i_img < end_index; ++i_img) {
                    attr_result[i_img].attr_scores = mlc_results[i_img - start_index].scores;
                    attr_result[i_img].box = det_result[i_img].box;
                    attr_result[i_img].box_score = det_result[i_img].score;
                    attr_result[i_img].box_label_id = det_result[i_img].label_id;
                }
            }
        }
        if (timers) timers->infer_timer.stop();
        return true;
    }
}
