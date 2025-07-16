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


    bool FaceRecognizerPipeline::predict(const cv::Mat& image, std::vector<FaceRecognitionResult>* results,
                                         TimerArray* timers) {
        if (timers) {
            timers->pre_timer.push_back(0);
            timers->post_timer.push_back(0);
            timers->infer_timer.start();
        }
        std::vector<DetectionLandmarkResult> det_result;
        if (!detector_->predict(image, &det_result)) {
            MD_LOG_ERROR << "detector predict failed" << std::endl;
            return false;
        }
        const auto aligned_images =
            modeldeploy::vision::utils::align_face_with_five_points(image, det_result);
        recognizer_->batch_predict(aligned_images, results);
        if (timers)timers->infer_timer.stop();
        return true;
    }
}
