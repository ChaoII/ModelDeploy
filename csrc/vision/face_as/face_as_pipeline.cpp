//
// Created by aichao on 2025/3/26.
//

#include "csrc/core/md_log.h"
#include "csrc/vision/utils.h"
#include "csrc/vision/common/processors/resize.h"
#include "csrc/vision/common/processors/hwc2chw.h"
#include "csrc/vision/common/processors/cast.h"
#include "csrc/vision/common/processors/center_crop.h"
#include "csrc/vision/face_as/face_as_pipeline.h"


namespace modeldeploy::vision::face {
    SeetaFaceAsPipeline::SeetaFaceAsPipeline(
        const std::string& face_det_model_file,
        const std::string& first_model_file,
        const std::string& second_model_file,
        const int thread_num) {
        RuntimeOption option;
        option.set_cpu_thread_num(thread_num);
        face_det_ = std::make_unique<SCRFD>(face_det_model_file, option);
        face_as_first_ = std::make_unique<SeetaFaceAntiSpoofFirst>(first_model_file, option);
        face_as_second_ = std::make_unique<SeetaFaceAntiSpoofSecond>(second_model_file, option);
    }


    bool SeetaFaceAsPipeline::is_initialized() const {
        if (face_det_ != nullptr && !face_det_->is_initialized()) {
            return false;
        }
        if (face_as_first_ != nullptr && !face_as_first_->is_initialized()) {
            return false;
        }
        if (face_as_second_ != nullptr && !face_as_second_->is_initialized()) {
            return false;
        }
        return true;
    }


    bool SeetaFaceAsPipeline::predict(cv::Mat& im, FaceAntiSpoofResult* results, const float fuse_threshold) {
        auto im_bak0 = im.clone();
        auto im_bak1 = im.clone();
        std::vector<std::tuple<int, float>> face_as_second_result;
        face_as_second_->predict(im_bak0, &face_as_second_result);
        std::vector<float> passive_results;
        DetectionLandmarkResult face_det_result_;
        std::cout << "size: " << face_as_second_result.size() << std::endl;
        cv::imshow("im", im_bak0);
        cv::waitKey(0);
        if (face_as_second_result.empty()) {
            if (!face_det_->predict(im_bak1, &face_det_result_)) {
                return false;
            }
            face_det_result_.display();
            results->resize(face_det_result_.boxes.size());
            auto align_im_list = utils::AlignFaceWithFivePoints(im, face_det_result_);
            for (auto& align_image : align_im_list) {
                float passive_result;
                face_as_first_->predict(align_image, &passive_result);
                passive_results.push_back(passive_result);
            }
        }
        else {
            passive_results.push_back(0.0);
        }

        for (const auto passive_result : passive_results) {
            if (passive_result > fuse_threshold) {
                results->anti_spoofs.push_back(FaceAntiSpoofType::REAL);
            }
            else {
                results->anti_spoofs.push_back(FaceAntiSpoofType::SPOOF);
            }
        }

        return true;
    }
}
