//
// insightface buffalo_l 性能基准：单模型（pre/infer/post 分解）+ 完整 pipeline。
// 用 TimerArray 分解各阶段耗时；结果打印到 stdout（INFO），供横向对比后端。
// 运行: ./test_modeldeploy "[insightface][benchmark]"（数据缺失时自动跳过）
//
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "csrc/utils/benchmark.h"
#include "csrc/vision/face/insightface/face_analysis.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;

namespace {
    std::string bm_data_dir() {
        const char* d = std::getenv("TEST_DATA_DIR");
        return d ? d : ".";
    }

    bool has_file(const std::string& p) { return std::filesystem::exists(p); }

    // 运行 N 次并汇总各阶段平均耗时
    void print_timers(const std::string& tag, const std::vector<TimerArray>& runs) {
        double pre = 0, infer = 0, post = 0, total = 0;
        for (const auto& t : runs) {
            pre += t.pre_timer.total_ms();
            infer += t.infer_timer.total_ms();
            post += t.post_timer.total_ms();
            total += t.pre_timer.total_ms() + t.infer_timer.total_ms() + t.post_timer.total_ms();
        }
        const size_t n = runs.size();
        pre /= n; infer /= n; post /= n; total /= n;
        std::cout << "== " << tag << ": avg over " << n << " runs -> pre="
                  << pre << "ms infer=" << infer << "ms post=" << post
                  << "ms total=" << total << "ms" << std::endl;
    }

    std::array<float, 4> pick_face_bbox() {
        // 用基准 JSON 里第一张脸的 bbox，保证所有单模型用同一张脸
        const std::string ref_path = bm_data_dir() + "/tests/data/insightface_buffalo_l_ref.json";
        if (has_file(ref_path)) {
            std::ifstream f(ref_path);
            nlohmann::json j;
            f >> j;
            if (!j["faces"].empty()) {
                const auto& b = j["faces"][0]["bbox"];
                return {b[0].get<float>(), b[1].get<float>(), b[2].get<float>(), b[3].get<float>()};
            }
        }
        return {640.0f, 124.0f, 672.0f, 168.0f};
    }
} // namespace

// ==================== 单模型：det_10g ====================
TEST_CASE("InsightFace benchmark det_10g", "[insightface][benchmark]") {
    const std::string model_dir = bm_data_dir() + "/test_data/test_models/onnx/insightface/buffalo_l";
    const std::string img_path = bm_data_dir() + "/test_data/test_images/test_person.jpg";
    if (!has_file(model_dir + "/det_10g.onnx")) return;
    if (!has_file(img_path)) return;

    auto det = std::make_unique<face::InsightFaceDet>(model_dir + "/det_10g.onnx");
    REQUIRE(det->is_initialized());
    auto img = ImageData::imread(img_path);

    constexpr int kRuns = 10;
    std::vector<TimerArray> runs;
    runs.reserve(kRuns);
    for (int i = 0; i < kRuns; ++i) {
        std::vector<face::InsightFaceBox> boxes;
        TimerArray t;
        REQUIRE(det->predict(img, &boxes, &t));
        REQUIRE(!boxes.empty());
        runs.push_back(t);
    }
    print_timers("det_10g", runs);
}

// ==================== 单模型：2d106det ====================
TEST_CASE("InsightFace benchmark 2d106det", "[insightface][benchmark]") {
    const std::string model_dir = bm_data_dir() + "/test_data/test_models/onnx/insightface/buffalo_l";
    const std::string img_path = bm_data_dir() + "/test_data/test_images/test_person.jpg";
    if (!has_file(model_dir + "/2d106det.onnx")) return;
    if (!has_file(img_path)) return;

    auto lmk = std::make_unique<face::InsightFaceLandmark>(model_dir + "/2d106det.onnx");
    REQUIRE(lmk->is_initialized());
    auto img = ImageData::imread(img_path);
    const auto bbox = pick_face_bbox();

    constexpr int kRuns = 10;
    std::vector<TimerArray> runs;
    for (int i = 0; i < kRuns; ++i) {
        std::vector<std::array<float, 2>> pts;
        TimerArray t;
        REQUIRE(lmk->predict_2d106(img, bbox, &pts, &t));
        REQUIRE(pts.size() == 106);
        runs.push_back(t);
    }
    print_timers("2d106det", runs);
}

// ==================== 单模型：1k3d68 ====================
TEST_CASE("InsightFace benchmark 1k3d68", "[insightface][benchmark]") {
    const std::string model_dir = bm_data_dir() + "/test_data/test_models/onnx/insightface/buffalo_l";
    const std::string img_path = bm_data_dir() + "/test_data/test_images/test_person.jpg";
    if (!has_file(model_dir + "/1k3d68.onnx")) return;
    if (!has_file(img_path)) return;

    auto lmk = std::make_unique<face::InsightFaceLandmark>(model_dir + "/1k3d68.onnx");
    REQUIRE(lmk->is_initialized());
    auto img = ImageData::imread(img_path);
    const auto bbox = pick_face_bbox();

    constexpr int kRuns = 10;
    std::vector<TimerArray> runs;
    for (int i = 0; i < kRuns; ++i) {
        std::vector<std::array<float, 3>> pts;
        std::array<float, 3> pose{0, 0, 0};
        TimerArray t;
        REQUIRE(lmk->predict_3d68(img, bbox, &pts, &pose, &t));
        REQUIRE(pts.size() == 68);
        runs.push_back(t);
    }
    print_timers("1k3d68", runs);
}

// ==================== 单模型：w600k_r50 ====================
TEST_CASE("InsightFace benchmark w600k_r50", "[insightface][benchmark]") {
    const std::string model_dir = bm_data_dir() + "/test_data/test_models/onnx/insightface/buffalo_l";
    const std::string img_path = bm_data_dir() + "/test_data/test_images/test_person.jpg";
    if (!has_file(model_dir + "/w600k_r50.onnx")) return;
    if (!has_file(img_path)) return;

    auto rec = std::make_unique<face::InsightFaceRecognition>(model_dir + "/w600k_r50.onnx");
    REQUIRE(rec->is_initialized());
    auto img = ImageData::imread(img_path);
    // 用基准里的 kps（5 点）
    std::vector<std::array<float, 2>> kps(5);
    {
        const std::string ref_path = bm_data_dir() + "/tests/data/insightface_buffalo_l_ref.json";
        if (has_file(ref_path)) {
            std::ifstream f(ref_path);
            nlohmann::json j;
            f >> j;
            const auto& k = j["faces"][0]["kps"];
            for (int i = 0; i < 5; ++i) kps[i] = {k[i][0].get<float>(), k[i][1].get<float>()};
        }
    }

    constexpr int kRuns = 10;
    std::vector<TimerArray> runs;
    for (int i = 0; i < kRuns; ++i) {
        std::vector<float> emb;
        TimerArray t;
        REQUIRE(rec->predict(img, kps, &emb, &t));
        REQUIRE(emb.size() == 512);
        runs.push_back(t);
    }
    print_timers("w600k_r50", runs);
}

// ==================== 单模型：genderage ====================
TEST_CASE("InsightFace benchmark genderage", "[insightface][benchmark]") {
    const std::string model_dir = bm_data_dir() + "/test_data/test_models/onnx/insightface/buffalo_l";
    const std::string img_path = bm_data_dir() + "/test_data/test_images/test_person.jpg";
    if (!has_file(model_dir + "/genderage.onnx")) return;
    if (!has_file(img_path)) return;

    auto ga = std::make_unique<face::InsightFaceGenderAge>(model_dir + "/genderage.onnx");
    REQUIRE(ga->is_initialized());
    auto img = ImageData::imread(img_path);
    const auto bbox = pick_face_bbox();

    constexpr int kRuns = 10;
    std::vector<TimerArray> runs;
    for (int i = 0; i < kRuns; ++i) {
        face::GenderAgeResult r;
        TimerArray t;
        REQUIRE(ga->predict_gender_age(img, bbox, &r, &t));
        REQUIRE(r.gender >= 0);
        runs.push_back(t);
    }
    print_timers("genderage", runs);
}

// ==================== 完整 pipeline ====================
TEST_CASE("InsightFace benchmark full pipeline", "[insightface][benchmark]") {
    const std::string model_dir = bm_data_dir() + "/test_data/test_models/onnx/insightface/buffalo_l";
    const std::string img_path = bm_data_dir() + "/test_data/test_images/test_person.jpg";
    if (!has_file(model_dir + "/genderage.onnx")) return;
    if (!has_file(img_path)) return;

    auto analysis = face::InsightFaceAnalysis::create_from_dir(model_dir);
    REQUIRE(analysis->is_initialized());
    auto img = ImageData::imread(img_path);

    constexpr int kRuns = 10;
    std::vector<TimerArray> runs;
    for (int i = 0; i < kRuns; ++i) {
        std::vector<face::InsightFaceResult> results;
        TimerArray t;
        REQUIRE(analysis->analyze(img, &results, true, true, true, true, &t));
        REQUIRE(!results.empty());
        runs.push_back(t);
    }
    print_timers("full pipeline (det+2d106+3d68+rec+genderage)", runs);
}

// ==================== TRT 后端对比（engine 存在时） ====================
#ifdef ENABLE_TRT
TEST_CASE("InsightFace benchmark det_10g TRT", "[insightface][benchmark][gpu]") {
    const std::string model_dir = bm_data_dir() + "/test_data/test_models/trt/insightface/buffalo_l";
    const std::string img_path = bm_data_dir() + "/test_data/test_images/test_person.jpg";
    if (!has_file(model_dir + "/det_10g.engine")) return;
    if (!has_file(img_path)) return;

    RuntimeOption opt;
    opt.use_gpu(0);
    opt.use_trt_backend();
    auto det = std::make_unique<face::InsightFaceDet>(model_dir + "/det_10g.engine", opt);
    REQUIRE(det->is_initialized());
    auto img = ImageData::imread(img_path);

    constexpr int kRuns = 20;
    std::vector<TimerArray> runs;
    for (int i = 0; i < kRuns; ++i) {
        std::vector<face::InsightFaceBox> boxes;
        TimerArray t;
        REQUIRE(det->predict(img, &boxes, &t));
        REQUIRE(!boxes.empty());
        runs.push_back(t);
    }
    print_timers("det_10g TRT", runs);
}

TEST_CASE("InsightFace benchmark full pipeline TRT", "[insightface][benchmark][gpu]") {
    const std::string model_dir = bm_data_dir() + "/test_data/test_models/trt/insightface/buffalo_l";
    const std::string img_path = bm_data_dir() + "/test_data/test_images/test_person.jpg";
    if (!has_file(model_dir + "/genderage.engine")) return;
    if (!has_file(img_path)) return;

    RuntimeOption opt;
    opt.use_gpu(0);
    opt.use_trt_backend();
    face::InsightFaceAnalysis analysis(
        model_dir + "/det_10g.engine", model_dir + "/w600k_r50.engine",
        model_dir + "/2d106det.engine", model_dir + "/1k3d68.engine",
        opt, model_dir + "/genderage.engine");
    REQUIRE(analysis.is_initialized());
    auto img = ImageData::imread(img_path);

    constexpr int kRuns = 20;
    std::vector<TimerArray> runs;
    for (int i = 0; i < kRuns; ++i) {
        std::vector<face::InsightFaceResult> results;
        TimerArray t;
        REQUIRE(analysis.analyze(img, &results, true, true, true, true, &t));
        REQUIRE(!results.empty());
        runs.push_back(t);
    }
    print_timers("full pipeline TRT (det+2d106+3d68+rec+genderage)", runs);
}
#endif // ENABLE_TRT
