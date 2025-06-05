//
// // Created by aichao on 2025/4/7.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/face/face_rec_pipeline/face_rec_pipeline.h"

namespace modeldeploy::vision::face
{
    FaceRecognizerPipeline::FaceRecognizerPipeline(const std::string& det_model_path,
                                                   const std::string& rec_model_path,
                                                   int thread_num) {
        RuntimeOption option;
        option.set_cpu_thread_num(thread_num);
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
        std::vector<DetectionLandmarkResult> det_result;
        TimerArray timer_det;
        if (!detector_->predict(image, &det_result, &timer_det)) {
            MD_LOG_ERROR << "detector predict failed" << std::endl;
            return false;
        }
        const auto aligned_images =
            modeldeploy::vision::utils::align_face_with_five_points(image, det_result);
        TimerArray timer_recs;
        const size_t lp_num = aligned_images.size();
        results->reserve(lp_num);
        for (const auto& aligned_image : aligned_images) {
            TimerArray timer_rec;
            FaceRecognitionResult tmp_result;
            recognizer_->predict(aligned_image, &tmp_result, &timer_rec);
            results->emplace_back(tmp_result);
            timer_recs += timer_rec;
        }
        *timers = timer_det + timer_recs;
        return true;
    }
}
