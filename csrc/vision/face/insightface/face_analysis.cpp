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
                                             const RuntimeOption& option) {
        if (!det_model.empty()) det_ = std::make_unique<InsightFaceDet>(det_model, option);
        if (!rec_model.empty()) rec_ = std::make_unique<InsightFaceRecognition>(rec_model, option);
        if (!lmk2d_model.empty()) lmk_2d_ = std::make_unique<InsightFaceLandmark>(lmk2d_model, option);
        if (!lmk3d_model.empty()) lmk_3d_ = std::make_unique<InsightFaceLandmark>(lmk3d_model, option);
        initialized_ = true;
        if (det_ && !det_->is_initialized()) { MD_LOG_ERROR << "det model init failed" << std::endl; initialized_ = false; }
        if (rec_ && !rec_->is_initialized()) { MD_LOG_ERROR << "rec model init failed" << std::endl; initialized_ = false; }
        if (lmk_2d_ && !lmk_2d_->is_initialized()) { MD_LOG_ERROR << "2d106 model init failed" << std::endl; initialized_ = false; }
        if (lmk_3d_ && !lmk_3d_->is_initialized()) { MD_LOG_ERROR << "3d68 model init failed" << std::endl; initialized_ = false; }
    }

    bool InsightFaceAnalysis::is_initialized() const { return initialized_; }

    bool InsightFaceAnalysis::detect(const ImageData& image, std::vector<InsightFaceBox>* boxes,
                                     TimerArray* timers) {
        if (!det_) return false;
        det_->det_thresh_ = det_thresh;
        return det_->predict(image, boxes, timers);
    }

    bool InsightFaceAnalysis::analyze(const ImageData& image, std::vector<InsightFaceResult>* results,
                                      bool with_2d106, bool with_3d68, bool with_recognition,
                                      TimerArray* timers) {
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
                if (lmk_2d_->predict_2d106(image, b.bbox, &lmk)) r.landmark_2d_106 = std::move(lmk);
            }
            if (with_3d68 && lmk_3d_) {
                std::vector<std::array<float, 3>> lmk;
                std::array<float, 3> pose{0, 0, 0};
                if (lmk_3d_->predict_3d68(image, b.bbox, &lmk, &pose)) {
                    r.landmark_3d_68 = std::move(lmk);
                    r.pose = pose;
                }
            }
            if (with_recognition && rec_ && b.kps.size() == 5) {
                std::vector<float> emb;
                if (rec_->predict(image, b.kps, &emb, timers)) r.embedding = std::move(emb);
            }
            results->push_back(std::move(r));
        }
        return true;
    }

    std::unique_ptr<InsightFaceAnalysis> InsightFaceAnalysis::create_from_dir(
        const std::string& model_dir, const RuntimeOption& option) {
        const auto det_path = model_dir + "/det_10g.onnx";
        const auto rec_path = model_dir + "/w600k_r50.onnx";
        const auto lmk2d_path = model_dir + "/2d106det.onnx";
        const auto lmk3d_path = model_dir + "/1k3d68.onnx";
        return std::make_unique<InsightFaceAnalysis>(det_path, rec_path, lmk2d_path,
                                                     lmk3d_path, option);
    }

} // namespace modeldeploy::vision::face
