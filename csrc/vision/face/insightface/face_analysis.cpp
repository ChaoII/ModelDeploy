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

    InsightFaceAnalysis::InsightFaceAnalysis(std::unique_ptr<InsightFaceDet> det,
                                             std::unique_ptr<InsightFaceRecognition> rec,
                                             std::unique_ptr<InsightFaceLandmark> lmk2d,
                                             std::unique_ptr<InsightFaceLandmark> lmk3d,
                                             std::unique_ptr<InsightFaceGenderAge> genderage)
        : det_(std::move(det)), rec_(std::move(rec)),
          lmk_2d_(std::move(lmk2d)), lmk_3d_(std::move(lmk3d)),
          genderage_(std::move(genderage)) {
        initialized_ = true;
    }

    std::unique_ptr<InsightFaceAnalysis> InsightFaceAnalysis::clone() const {
        std::unique_ptr<InsightFaceDet> det;
        std::unique_ptr<InsightFaceRecognition> rec;
        std::unique_ptr<InsightFaceLandmark> lmk2d;
        std::unique_ptr<InsightFaceLandmark> lmk3d;
        std::unique_ptr<InsightFaceGenderAge> genderage;
        // 逐个深拷贝子模型：复用各自已加载的 backend session，不重新加载模型
        if (det_) det = det_->clone();
        if (rec_) rec = rec_->clone();
        if (lmk_2d_) lmk2d = lmk_2d_->clone();
        if (lmk_3d_) lmk3d = lmk_3d_->clone();
        if (genderage_) genderage = genderage_->clone();
        return std::unique_ptr<InsightFaceAnalysis>(new InsightFaceAnalysis(
            std::move(det), std::move(rec), std::move(lmk2d), std::move(lmk3d), std::move(genderage)));
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
                                      bool with_genderage, bool max_face_only, TimerArray* timers) {
        if (!results) return false;
        results->clear();
        std::vector<InsightFaceBox> boxes;
        if (!detect(image, &boxes, timers)) return false;
        if (boxes.empty()) return true;

        // 仅最大人脸模式：按 bbox 面积取最大
        if (max_face_only) {
            size_t max_i = 0;
            float max_area = -1.0f;
            for (size_t i = 0; i < boxes.size(); ++i) {
                const float w = boxes[i].bbox[2] - boxes[i].bbox[0];
                const float h = boxes[i].bbox[3] - boxes[i].bbox[1];
                const float area = w * h;
                if (area > max_area) { max_area = area; max_i = i; }
            }
            std::vector<InsightFaceBox> one{boxes[max_i]};
            return analyze_impl(image, one, results, with_2d106, with_3d68,
                                with_recognition, with_genderage, timers);
        }
        return analyze_impl(image, boxes, results, with_2d106, with_3d68,
                            with_recognition, with_genderage, timers);
    }

    bool InsightFaceAnalysis::analyze_max_face(const ImageData& image, InsightFaceResult* result,
                                               bool with_2d106, bool with_3d68,
                                               bool with_recognition, bool with_genderage,
                                               int* face_count, TimerArray* timers) {
        if (!result) return false;
        std::vector<InsightFaceResult> results;
        if (!analyze(image, &results, with_2d106, with_3d68, with_recognition, with_genderage,
                     true, timers)) return false;
        if (face_count) *face_count = static_cast<int>(results.size());
        if (results.empty()) return false;
        *result = std::move(results[0]);
        return true;
    }

    bool InsightFaceAnalysis::analyze_impl(const ImageData& image,
                                           const std::vector<InsightFaceBox>& boxes,
                                           std::vector<InsightFaceResult>* results,
                                           bool with_2d106, bool with_3d68,
                                           bool with_recognition, bool with_genderage,
                                           TimerArray* timers) {
        const size_t n = boxes.size();
        results->resize(n);
        // bbox 列表（供 batch 推理）
        std::vector<std::array<float, 4>> bbox_list(n);
        for (size_t i = 0; i < n; ++i) {
            bbox_list[i] = boxes[i].bbox;
            (*results)[i].bbox = boxes[i].bbox;
            (*results)[i].det_score = boxes[i].score;
            (*results)[i].kps = boxes[i].kps;
        }

        // 2D 106 关键点：一次 batch 推理
        if (with_2d106 && lmk_2d_ && n > 0) {
            std::vector<std::vector<std::array<float, 2>>> lmk_list;
            if (lmk_2d_->batch_predict_2d106(image, bbox_list, &lmk_list, timers)) {
                for (size_t i = 0; i < n; ++i) (*results)[i].landmark_2d_106 = std::move(lmk_list[i]);
            }
        }
        // 3D 68 关键点 + pose：一次 batch 推理
        if (with_3d68 && lmk_3d_ && n > 0) {
            std::vector<std::vector<std::array<float, 3>>> lmk_list;
            std::vector<std::array<float, 3>> pose_list;
            if (lmk_3d_->batch_predict_3d68(image, bbox_list, &lmk_list, &pose_list, timers)) {
                for (size_t i = 0; i < n; ++i) {
                    (*results)[i].landmark_3d_68 = std::move(lmk_list[i]);
                    (*results)[i].pose = pose_list[i];
                }
            }
        }
        // 识别 embedding：一次 batch 推理
        if (with_recognition && rec_ && n > 0) {
            std::vector<std::vector<std::array<float, 2>>> kps_list;
            bool all_kps = true;
            for (size_t i = 0; i < n; ++i) {
                if (boxes[i].kps.size() == 5) kps_list.push_back(boxes[i].kps);
                else { all_kps = false; kps_list.emplace_back(); }
            }
            if (all_kps) {
                std::vector<std::vector<float>> emb_list;
                if (rec_->batch_predict(image, kps_list, &emb_list, timers)) {
                    for (size_t i = 0; i < n; ++i) (*results)[i].embedding = std::move(emb_list[i]);
                }
            } else {
                // 个别脸缺 kps：逐脸 fallback
                for (size_t i = 0; i < n; ++i) {
                    if (boxes[i].kps.size() == 5) {
                        std::vector<float> emb;
                        if (rec_->predict(image, boxes[i].kps, &emb, timers)) (*results)[i].embedding = std::move(emb);
                    }
                }
            }
        }
        // genderage：一次 batch 推理
        if (with_genderage && genderage_ && n > 0) {
            std::vector<GenderAgeResult> ga_list;
            if (genderage_->batch_predict_gender_age(image, bbox_list, &ga_list, timers)) {
                for (size_t i = 0; i < n; ++i) {
                    (*results)[i].gender = ga_list[i].gender;
                    (*results)[i].age = ga_list[i].age;
                }
            }
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
