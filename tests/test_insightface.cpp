//
// insightface buffalo_l 精度对齐测试：C++ 实现 vs python insightface 基准。
// 检测/关键点/姿态逐值对比，embedding 用宽松容差（OpenCV 版本插值差异）。
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "csrc/vision/face/insightface/face_analysis.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;
using json = nlohmann::json;

namespace {

    std::string test_data_dir() {
        const char* d = std::getenv("TEST_DATA_DIR");
        return d ? d : ".";
    }

    // 读取基准 JSON
    std::vector<face::InsightFaceBox> load_ref_boxes(const std::string& path) {
        std::ifstream f(path);
        json j;
        f >> j;
        std::vector<face::InsightFaceBox> boxes;
        for (const auto& fj : j["faces"]) {
            face::InsightFaceBox b;
            b.bbox = {fj["bbox"][0].get<float>(), fj["bbox"][1].get<float>(),
                      fj["bbox"][2].get<float>(), fj["bbox"][3].get<float>()};
            b.score = fj["det_score"].get<float>();
            for (const auto& k : fj["kps"]) {
                b.kps.push_back({k[0].get<float>(), k[1].get<float>()});
            }
            boxes.push_back(b);
        }
        return boxes;
    }
} // namespace

// ==================== 检测对齐 ====================
TEST_CASE("InsightFace det_10g detection aligns with python", "[insightface][model]") {
    const std::string model_dir = test_data_dir() + "/test_data/test_models/onnx/insightface/buffalo_l";
    const std::string img_path = test_data_dir() + "/test_data/test_images/test_person.jpg";
    const std::string ref_path = test_data_dir() + "/tests/data/insightface_buffalo_l_ref.json";
    if (!std::filesystem::exists(model_dir + "/det_10g.onnx")) return;
    if (!std::filesystem::exists(img_path)) return;
    if (!std::filesystem::exists(ref_path)) return;

    auto det = std::make_unique<face::InsightFaceDet>(model_dir + "/det_10g.onnx");
    REQUIRE(det->is_initialized());
    auto img = ImageData::imread(img_path);

    std::vector<face::InsightFaceBox> boxes;
    REQUIRE(det->predict(img, &boxes));

    const auto ref = load_ref_boxes(ref_path);
    REQUIRE(!ref.empty());
    REQUIRE(!boxes.empty());

    // 按 score 排序对比 top N（用索引排序避免结构体拷贝问题）
    auto top_indices = [](const auto& v) {
        std::vector<int> idx(v.size());
        for (int i = 0; i < (int)v.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a].score > v[b].score; });
        return idx;
    };
    auto cpp_idx = top_indices(boxes);
    auto ref_idx = top_indices(ref);

    const size_t n = std::min<size_t>(5, std::min(boxes.size(), ref.size()));
    REQUIRE(n >= 3); // 至少 3 张对齐
    for (size_t i = 0; i < n; ++i) {
        INFO("face " << i);
        const auto& b = boxes[cpp_idx[i]];
        const auto& r = ref[ref_idx[i]];
        for (int j = 0; j < 4; ++j) {
            REQUIRE(std::fabs(b.bbox[j] - r.bbox[j]) < 0.1f);
        }
        REQUIRE(std::fabs(b.score - r.score) < 1e-3f);
        REQUIRE(b.kps.size() == 5);
        REQUIRE(r.kps.size() == 5);
        for (int k = 0; k < 5; ++k) {
            REQUIRE(std::fabs(b.kps[k][0] - r.kps[k][0]) < 0.1f);
            REQUIRE(std::fabs(b.kps[k][1] - r.kps[k][1]) < 0.1f);
        }
    }
}

// ==================== 全流程（landmark + recognition） ====================
TEST_CASE("InsightFace full pipeline aligns with python", "[insightface][model]") {
    const std::string model_dir = test_data_dir() + "/test_data/test_models/onnx/insightface/buffalo_l";
    const std::string img_path = test_data_dir() + "/test_data/test_images/test_person.jpg";
    const std::string ref_path = test_data_dir() + "/tests/data/insightface_buffalo_l_ref.json";
    if (!std::filesystem::exists(model_dir + "/2d106det.onnx")) return;

    auto analysis = face::InsightFaceAnalysis::create_from_dir(model_dir);
    REQUIRE(analysis->is_initialized());
    auto img = ImageData::imread(img_path);

    std::vector<face::InsightFaceResult> results;
    REQUIRE(analysis->analyze(img, &results, true, true, true));
    REQUIRE(!results.empty());

    // 对比 top 人脸
    const auto ref = load_ref_boxes(ref_path);

    // 找分数最高的参考脸
    float best_score = -1;
    size_t best_idx = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i].score > best_score) { best_score = ref[i].score; best_idx = i; }
    }
    // 用 C++ 结果里 bbox 最接近参考的脸
    size_t match = 0;
    float min_center_dist = 1e9;
    for (size_t i = 0; i < results.size(); ++i) {
        const float cx = (results[i].bbox[0] + results[i].bbox[2]) / 2.0f;
        const float cy = (results[i].bbox[1] + results[i].bbox[3]) / 2.0f;
        const float rx = (ref[best_idx].bbox[0] + ref[best_idx].bbox[2]) / 2.0f;
        const float ry = (ref[best_idx].bbox[1] + ref[best_idx].bbox[3]) / 2.0f;
        const float dist = std::hypot(cx - rx, cy - ry);
        if (dist < min_center_dist) { min_center_dist = dist; match = i; }
    }

    // 2D 106 关键点对齐（容差 1px）
    std::ifstream f(ref_path);
    json j;
    f >> j;
    const auto& ref2d = j["faces"][best_idx]["landmark_2d_106"];
    const auto& ref3d = j["faces"][best_idx]["landmark_3d_68"];
    const auto& refpose = j["faces"][best_idx]["pose"];

    if (!results[match].landmark_2d_106.empty() && ref2d.size() == 106) {
        for (int i = 0; i < 106; ++i) {
            REQUIRE(std::fabs(results[match].landmark_2d_106[i][0] - ref2d[i][0].get<float>()) < 1.0f);
            REQUIRE(std::fabs(results[match].landmark_2d_106[i][1] - ref2d[i][1].get<float>()) < 1.0f);
        }
    }
    if (!results[match].landmark_3d_68.empty() && ref3d.size() == 68) {
        for (int i = 0; i < 68; ++i) {
            REQUIRE(std::fabs(results[match].landmark_3d_68[i][0] - ref3d[i][0].get<float>()) < 1.0f);
            REQUIRE(std::fabs(results[match].landmark_3d_68[i][1] - ref3d[i][1].get<float>()) < 1.0f);
            REQUIRE(std::fabs(results[match].landmark_3d_68[i][2] - ref3d[i][2].get<float>()) < 2.0f);
        }
        // pose（角度，容差 1 度）
        REQUIRE(std::fabs(results[match].pose[0] - refpose[0].get<float>()) < 1.0f);
        REQUIRE(std::fabs(results[match].pose[1] - refpose[1].get<float>()) < 1.0f);
        REQUIRE(std::fabs(results[match].pose[2] - refpose[2].get<float>()) < 1.0f);
    }

    // embedding：相对误差（L2 归一化后 cosine 相似度 > 0.99）
    if (!results[match].embedding.empty()) {
        const auto& refemb = j["faces"][best_idx]["embedding"];
        REQUIRE(refemb.size() == results[match].embedding.size());
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < refemb.size(); ++i) {
            const double a = results[match].embedding[i];
            const double b = refemb[i].get<double>();
            dot += a * b; na += a * a; nb += b * b;
        }
        const double sim = dot / (std::sqrt(na) * std::sqrt(nb));
        INFO("embedding cosine sim=" << sim);
        REQUIRE(sim > 0.99);
    }
}
