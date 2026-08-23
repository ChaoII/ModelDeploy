#include "vision/landmark/face_landmark.h"

namespace modeldeploy::vision::landmark {
    FaceLandmark::FaceLandmark(const std::string& model_file, const RuntimeOption& option)
        : landmark_(std::make_unique<face::InsightFaceLandmark>(model_file, option)) {}

    FaceLandmark::FaceLandmark(std::unique_ptr<face::InsightFaceLandmark> lm)
        : landmark_(std::move(lm)) {}

    bool FaceLandmark::is_initialized() const {
        return landmark_ && landmark_->is_initialized();
    }

    bool FaceLandmark::predict(const ImageData& img,
                               std::vector<KeyPointsResult>* results,
                               TimerArray* timer) {
        if (!results || !landmark_) return false;
        // 对"人脸裁剪图"整体估计 106 点：bbox 取整图范围（裁剪图输入约定，spec §3.2）。
        const float W = static_cast<float>(img.width());
        const float H = static_cast<float>(img.height());
        std::array<float, 4> bbox{0.f, 0.f, W, H};
        std::vector<std::array<float, 2>> lms;
        if (!landmark_->predict_2d106(img, bbox, &lms, timer)) return false;

        KeyPointsResult r;
        r.box = Rect2f(0.f, 0.f, W, H);
        r.label_id = 0;
        r.score = 1.0f;
        r.keypoints.reserve(lms.size());
        for (const auto& p : lms)
            r.keypoints.emplace_back(p[0], p[1], 0.f);
        results->clear();
        results->push_back(std::move(r));
        return true;
    }

    std::unique_ptr<FaceLandmark> FaceLandmark::clone() const {
        // InsightFaceLandmark::clone() 会基于存储的 model_file + runtime 重建独立实例
        auto lm = landmark_ ? landmark_->clone() : nullptr;
        return std::unique_ptr<FaceLandmark>(new FaceLandmark(std::move(lm)));
    }
} // namespace modeldeploy::vision::landmark
