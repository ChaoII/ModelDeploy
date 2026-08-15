//
// insightface buffalo_l 人脸分析 C API 实现。
//

#include "csrc/vision.h"
#include "capi/common/md_micro.h"
#include "capi/utils/internal/utils.h"
#include "capi/vision/face/insightface_capi.h"

#include <cstring>
#include <vector>
#include <array>

namespace {
    // InsightFaceResult -> MDInsightFaceResult
    void insightface_result_2_c(const modeldeploy::vision::face::InsightFaceResult& r,
                                MDInsightFaceResult* c) {
        c->box.x = static_cast<int>(r.bbox[0]);
        c->box.y = static_cast<int>(r.bbox[1]);
        c->box.width = static_cast<int>(r.bbox[2] - r.bbox[0]);
        c->box.height = static_cast<int>(r.bbox[3] - r.bbox[1]);
        c->score = r.det_score;
        // kps
        c->kps_size = static_cast<int>(r.kps.size());
        c->kps = c->kps_size > 0 ? new MDPoint3f[c->kps_size] : nullptr;
        for (int i = 0; i < c->kps_size; ++i) {
            c->kps[i].x = r.kps[i][0];
            c->kps[i].y = r.kps[i][1];
            c->kps[i].z = 0.0f;
        }
        // 2d106
        c->landmark_2d_106_size = static_cast<int>(r.landmark_2d_106.size());
        c->landmark_2d_106 = c->landmark_2d_106_size > 0 ? new MDPoint3f[c->landmark_2d_106_size] : nullptr;
        for (int i = 0; i < c->landmark_2d_106_size; ++i) {
            c->landmark_2d_106[i].x = r.landmark_2d_106[i][0];
            c->landmark_2d_106[i].y = r.landmark_2d_106[i][1];
            c->landmark_2d_106[i].z = 0.0f;
        }
        // 3d68
        c->landmark_3d_68_size = static_cast<int>(r.landmark_3d_68.size());
        c->landmark_3d_68 = c->landmark_3d_68_size > 0 ? new MDPoint3f[c->landmark_3d_68_size] : nullptr;
        for (int i = 0; i < c->landmark_3d_68_size; ++i) {
            c->landmark_3d_68[i].x = r.landmark_3d_68[i][0];
            c->landmark_3d_68[i].y = r.landmark_3d_68[i][1];
            c->landmark_3d_68[i].z = r.landmark_3d_68[i][2];
        }
        // pose
        c->pose[0] = r.pose[0];
        c->pose[1] = r.pose[1];
        c->pose[2] = r.pose[2];
        // embedding
        c->embedding_size = static_cast<int>(r.embedding.size());
        c->embedding = c->embedding_size > 0 ? new float[c->embedding_size] : nullptr;
        if (c->embedding) std::memcpy(c->embedding, r.embedding.data(), c->embedding_size * sizeof(float));
    }
} // namespace

MDStatusCode md_create_insightface_model(MDModel* model,
                                         const char* det_model_path,
                                         const char* rec_model_path,
                                         const char* lmk2d_model_path,
                                         const char* lmk3d_model_path,
                                         const MDRuntimeOption* option) {
    modeldeploy::RuntimeOption _option;
    c_runtime_option_2_runtime_option(option, &_option);
    auto analysis = std::make_unique<modeldeploy::vision::face::InsightFaceAnalysis>(
        det_model_path ? det_model_path : "", rec_model_path ? rec_model_path : "",
        lmk2d_model_path ? lmk2d_model_path : "", lmk3d_model_path ? lmk3d_model_path : "",
        _option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup("InsightFaceAnalysis");
    model->model_content = analysis.release();
    model->type = MDModelType::InsightFace;
    const auto* p = static_cast<modeldeploy::vision::face::InsightFaceAnalysis*>(model->model_content);
    if (!p->is_initialized()) {
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDStatusCode md_create_insightface_det_model(MDModel* model, const char* model_path,
                                             const MDRuntimeOption* option) {
    modeldeploy::RuntimeOption _option;
    c_runtime_option_2_runtime_option(option, &_option);
    auto det = std::make_unique<modeldeploy::vision::face::InsightFaceDet>(model_path, _option);
    model->format = MDModelFormat::ONNX;
    model->model_name = strdup(det->name().c_str());
    model->model_content = det.release();
    model->type = MDModelType::InsightFace;
    if (!static_cast<modeldeploy::vision::face::InsightFaceDet*>(model->model_content)->is_initialized()) {
        return MDStatusCode::ModelInitializeFailed;
    }
    return MDStatusCode::Success;
}

MDStatusCode md_insightface_analyze(const MDModel* model, MDImage* image,
                                    MDInsightFaceResults* c_results) {
    if (model->type != MDModelType::InsightFace) {
        return MDStatusCode::ModelTypeError;
    }
    auto image_data = md_image_to_image_data(image);
    std::vector<modeldeploy::vision::face::InsightFaceResult> results;
    const auto analysis = static_cast<modeldeploy::vision::face::InsightFaceAnalysis*>(model->model_content);
    if (!analysis->analyze(image_data, &results, true, true, true)) {
        return MDStatusCode::ModelPredictFailed;
    }
    c_results->size = static_cast<int>(results.size());
    c_results->data = results.empty() ? nullptr : new MDInsightFaceResult[results.size()];
    for (size_t i = 0; i < results.size(); ++i) {
        insightface_result_2_c(results[i], &c_results->data[i]);
    }
    return MDStatusCode::Success;
}

MDStatusCode md_insightface_detect(const MDModel* model, MDImage* image,
                                   MDKeyPointResults* c_results) {
    if (model->type != MDModelType::InsightFace) {
        return MDStatusCode::ModelTypeError;
    }
    auto image_data = md_image_to_image_data(image);
    std::vector<modeldeploy::vision::face::InsightFaceBox> boxes;
    const auto det = static_cast<modeldeploy::vision::face::InsightFaceDet*>(model->model_content);
    if (!det->predict(image_data, &boxes)) {
        return MDStatusCode::ModelPredictFailed;
    }
    std::vector<modeldeploy::vision::KeyPointsResult> kp_results;
    kp_results.reserve(boxes.size());
    for (const auto& b : boxes) {
        modeldeploy::vision::KeyPointsResult k;
        k.box.x = b.bbox[0];
        k.box.y = b.bbox[1];
        k.box.width = b.bbox[2] - b.bbox[0];
        k.box.height = b.bbox[3] - b.bbox[1];
        k.score = b.score;
        for (const auto& p : b.kps) k.keypoints.emplace_back(p[0], p[1], 0.0f);
        kp_results.push_back(std::move(k));
    }
    keypoint_results_2_c_results(kp_results, c_results);
    return MDStatusCode::Success;
}

void md_insightface_set_det_thresh(MDModel* model, float thresh) {
    auto analysis = static_cast<modeldeploy::vision::face::InsightFaceAnalysis*>(model->model_content);
    analysis->set_det_thresh(thresh);
}

void md_free_insightface_result(MDInsightFaceResults* c_results) {
    if (c_results->size > 0 && c_results->data != nullptr) {
        for (int i = 0; i < c_results->size; ++i) {
            delete[] c_results->data[i].kps;
            delete[] c_results->data[i].landmark_2d_106;
            delete[] c_results->data[i].landmark_3d_68;
            delete[] c_results->data[i].embedding;
        }
        delete[] c_results->data;
        c_results->data = nullptr;
        c_results->size = 0;
    }
}

void md_free_insightface_model(MDModel* model) {
    if (model->model_content != nullptr) {
        delete static_cast<modeldeploy::vision::face::InsightFaceAnalysis*>(model->model_content);
        model->model_content = nullptr;
    }
    if (model->model_name != nullptr) {
        free(model->model_name);
        model->model_name = nullptr;
    }
}
