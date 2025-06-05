//
// // Created by aichao on 2025/4/7.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/face/face_rec_pipeline/face_rec_pipeline.h"

namespace modeldeploy::vision::face {
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


    bool FaceRecognizerPipeline::predict(const cv::Mat& image, std::vector<FaceRecognitionResult>* results) {
        cv::Mat image_bak = image;
        std::vector<DetectionLandmarkResult> det_result;
        if (!detector_->predict(image_bak, &det_result)) {
            MD_LOG_ERROR << "detector predict failed" << std::endl;
            return false;
        }
        const auto aligned_images =
            modeldeploy::vision::utils::align_face_with_five_points(image, det_result);
        const size_t lp_num = aligned_images.size();
        results->reserve(lp_num);
        for (const auto& aligned_image : aligned_images) {
            FaceRecognitionResult tmp_result;
            recognizer_->predict(aligned_image, &tmp_result);
            results->emplace_back(tmp_result);
        }
        return true;
    }
}
