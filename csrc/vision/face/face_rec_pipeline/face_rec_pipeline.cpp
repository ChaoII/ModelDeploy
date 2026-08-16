//
// // Created by aichao on 2025/4/7.
//

#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/face/face_rec_pipeline/face_rec_pipeline.h"

namespace modeldeploy::vision::face {
    FaceRecognizerPipeline::FaceRecognizerPipeline(const std::string& det_model_path,
                                                   const std::string& rec_model_path,
                                                   const RuntimeOption& option) {
        detector_ = std::make_unique<Scrfd>(det_model_path, option);
        recognizer_ = std::make_unique<SeetaFaceID>(rec_model_path, option);
    }

    FaceRecognizerPipeline::~FaceRecognizerPipeline() = default;

    bool FaceRecognizerPipeline::is_initialized() const {
        if (detector_ != nullptr && !detector_->is_initialized()) {
            return false;
        }
        if (recognizer_ != nullptr && !recognizer_->is_initialized()) {
            return false;
        }
        return true;
    }


    bool FaceRecognizerPipeline::predict(const ImageData& image, std::vector<FaceRecognitionResult>* results,
                                         TimerArray* timers) {
        if (timers) {
            timers->pre_timer.add_sample(0);
            timers->post_timer.add_sample(0);
            timers->infer_timer.start();
        }
        std::vector<KeyPointsResult> det_result;
        if (!detector_->predict(image, &det_result)) {
            MD_LOG_ERROR << "detector predict failed" << std::endl;
            return false;
        }
        if (det_result.empty()) {
            MD_LOG_WARN << "Cant not find any face!" << std::endl;
            return false;
        }
        utils::sorted_det_land_mark_results(det_result);
        const auto aligned_images =
            modeldeploy::vision::utils::align_face_with_five_points(image, det_result);
        if (!recognizer_->batch_predict(aligned_images, results)) {
            MD_LOG_ERROR << "recognizer predict failed" << std::endl;
            return false;
        }
        if (timers)timers->infer_timer.stop();
        return true;
    }

    bool FaceRecognizerPipeline::predict_max_face(const ImageData& image, FaceRecognitionResult* result,
                                                  int* face_count, TimerArray* timers) {
        if (!result) return false;
        std::vector<KeyPointsResult> det_result;
        if (timers) {
            timers->pre_timer.add_sample(0);
            timers->post_timer.add_sample(0);
            timers->infer_timer.start();
        }
        if (!detector_->predict(image, &det_result)) {
            MD_LOG_ERROR << "detector predict failed" << std::endl;
            return false;
        }
        if (face_count) *face_count = static_cast<int>(det_result.size());
        if (det_result.empty()) {
            MD_LOG_WARN << "Cant not find any face!" << std::endl;
            return false;
        }
        // 取 bbox 面积最大的人脸
        size_t max_i = 0;
        float max_area = -1.0f;
        for (size_t i = 0; i < det_result.size(); ++i) {
            const float area = det_result[i].box.width * det_result[i].box.height;
            if (area > max_area) { max_area = area; max_i = i; }
        }
        std::vector<KeyPointsResult> one{det_result[max_i]};
        const auto aligned_images =
            modeldeploy::vision::utils::align_face_with_five_points(image, one);
        std::vector<FaceRecognitionResult> results;
        if (!recognizer_->batch_predict(aligned_images, &results)) {
            MD_LOG_ERROR << "recognizer predict failed" << std::endl;
            return false;
        }
        if (!results.empty()) *result = std::move(results[0]);
        if (timers)timers->infer_timer.stop();
        return true;
    }

    std::unique_ptr<FaceRecognizerPipeline> FaceRecognizerPipeline::clone() const {
        auto clone_model = std::make_unique<FaceRecognizerPipeline>(*this);
        if (detector_) clone_model->detector_ = detector_->clone();
        if (recognizer_) clone_model->recognizer_ = recognizer_->clone();
        return clone_model;
    }
}

