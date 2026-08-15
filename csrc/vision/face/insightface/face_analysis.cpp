//
// insightface FaceAnalysis 综合 pipeline 实现。
//
#include "core/md_log.h"
#include "vision/face/insightface/face_analysis.h"
#include "utils/benchmark.h"

namespace modeldeploy::vision::face {

    InsightFaceAnalysis::InsightFaceAnalysis(const std::string& det_model,
                                             const std::string& rec_model,
                                             const std::string& lmk2d_model,
                                             const std::string& lmk3d_model,
                                             const RuntimeOption& option,
                                             const std::string& genderage_model) {
        if (!det_model.empty()) det_ = std::make_unique<InsightFaceDet>(det_model, option);
        if (!rec_model.empty()) rec_ = std::make_unique<InsightFaceRecognition>(rec_model, option);
        if (!lmk2d_model.empty()) lmk_2d_ = std::make_unique<InsightFaceLandmark>(lmk2d_model, option);
        if (!lmk3d_model.empty()) lmk_3d_ = std::make_unique<InsightFaceLandmark>(lmk3d_model, option);
        if (!genderage_model.empty()) genderage_ = std::make_unique<InsightFaceGenderAge>(genderage_model, option);
        initialized_ = true;
        if (det_ && !det_->is_initialized()) { MD_LOG_ERROR << "det model init failed" << std::endl; initialized_ = false; }
        if (rec_ && !rec_->is_initialized()) { MD_LOG_ERROR << "rec model init failed" << std::endl; initialized_ = false; }
        if (lmk_2d_ && !lmk_2d_->is_initialized()) { MD_LOG_ERROR << "2d106 model init failed" << std::endl; initialized_ = false; }
        if (lmk_3d_ && !lmk_3d_->is_initialized()) { MD_LOG_ERROR << "3d68 model init failed" << std::endl; initialized_ = false; }
        if (genderage_ && !genderage_->is_initialized()) { MD_LOG_ERROR << "genderage model init failed" << std::endl; initialized_ = false; }
    }

    bool InsightFaceAnalysis::is_initialized() const { return initialized_; }

    void InsightFaceAnalysis::set_det_thresh(float thresh) {
        if (det_) det_->get_preprocessor().det_thresh = thresh;
    }

    bool InsightFaceAnalysis::detect(const ImageData& image, std::vector<InsightFaceBox>* boxes,
                                     TimerArray* timers) {
        if (!det_) return false;
        return det_->predict(image, boxes, timers);
    }

    bool InsightFaceAnalysis::analyze(const ImageData& image, std::vector<InsightFaceResult>* results,
                                      bool with_2d106, bool with_3d68, bool with_recognition,
                                      bool with_genderage, TimerArray* timers) {
        if (!results) return false;
        results->clear();
        std::vector<InsightFaceBox> boxes;
        if (!detect(image, &boxes, timers)) return false;

        results->reserve(boxes.size());
        for (const auto& b : boxes) {
            InsightFaceResult r;
            r.bbox = b.bbox;
            r.det_score = b.score;
            r.kps = b.kps;
            if (with_2d106 && lmk_2d_) {
                std::vector<std::array<float, 2>> lmk;
                if (lmk_2d_->predict_2d106(image, b.bbox, &lmk, timers)) r.landmark_2d_106 = std::move(lmk);
            }
            if (with_3d68 && lmk_3d_) {
                std::vector<std::array<float, 3>> lmk;
                std::array<float, 3> pose{0, 0, 0};
                if (lmk_3d_->predict_3d68(image, b.bbox, &lmk, &pose, timers)) {
                    r.landmark_3d_68 = std::move(lmk);
                    r.pose = pose;
                }
            }
            if (with_recognition && rec_ && b.kps.size() == 5) {
                std::vector<float> emb;
                if (rec_->predict(image, b.kps, &emb, timers)) r.embedding = std::move(emb);
            }
            if (with_genderage && genderage_) {
                GenderAgeResult ga;
                if (genderage_->predict_gender_age(image, b.bbox, &ga, timers)) {
                    r.gender = ga.gender;
                    r.age = ga.age;
                }
            }
            results->push_back(std::move(r));
        }
        return true;
    }

    std::unique_ptr<InsightFaceAnalysis> InsightFaceAnalysis::create_from_dir(
        const std::string& model_dir, const RuntimeOption& option) {
        return std::make_unique<InsightFaceAnalysis>(
            model_dir + "/det_10g.onnx", model_dir + "/w600k_r50.onnx",
            model_dir + "/2d106det.onnx", model_dir + "/1k3d68.onnx", option,
            model_dir + "/genderage.onnx");
    }

} // namespace modeldeploy::vision::face
