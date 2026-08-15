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
#include "capi/common/md_types.h"
#include "capi/common/md_decl.h"
#include "capi/utils/md_image_capi.h"
#include "capi/vision/face/insightface_capi.h"
#include <opencv2/opencv.hpp>

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

    // genderage：与 python 逐人脸对齐（gender 精确、age 精确）
    if (results[match].gender >= 0 && j["faces"][best_idx].contains("gender")) {
        REQUIRE(results[match].gender == j["faces"][best_idx]["gender"].get<int>());
    }
    if (results[match].age >= 0 && j["faces"][best_idx].contains("age")) {
        REQUIRE(results[match].age == j["faces"][best_idx]["age"].get<int>());
    }
}

// ==================== MNN 后端 ====================
#ifdef ENABLE_MNN
TEST_CASE("InsightFace det_10g on MNN backend", "[insightface][model][backend:mnn]") {
    const std::string model_dir = test_data_dir() + "/test_data/test_models/mnn/insightface/buffalo_l";
    const std::string img_path = test_data_dir() + "/test_data/test_images/test_person.jpg";
    if (!std::filesystem::exists(model_dir + "/det_10g.mnn")) return;
    if (!std::filesystem::exists(img_path)) return;

    // .mnn 路径自动选 MNN 后端
    auto det = std::make_unique<face::InsightFaceDet>(model_dir + "/det_10g.mnn");
    REQUIRE(det->is_initialized());
    auto img = ImageData::imread(img_path);

    std::vector<face::InsightFaceBox> boxes;
    REQUIRE(det->predict(img, &boxes));
    // MNN 动态 shape 推理应至少检出人脸（结果数量可与 ORT 不同，但不应为空）
    REQUIRE(!boxes.empty());
}

TEST_CASE("InsightFace full pipeline on MNN backend", "[insightface][model][backend:mnn]") {
    const std::string model_dir = test_data_dir() + "/test_data/test_models/mnn/insightface/buffalo_l";
    const std::string img_path = test_data_dir() + "/test_data/test_images/test_person.jpg";
    if (!std::filesystem::exists(model_dir + "/det_10g.mnn")) return;
    if (!std::filesystem::exists(model_dir + "/2d106det.mnn")) return;
    if (!std::filesystem::exists(model_dir + "/1k3d68.mnn")) return;
    if (!std::filesystem::exists(model_dir + "/w600k_r50.mnn")) return;
    if (!std::filesystem::exists(img_path)) return;

    // 用 MNN 模型路径构造 pipeline（各子模型自动选 MNN 后端）
    face::InsightFaceAnalysis analysis(
        model_dir + "/det_10g.mnn", model_dir + "/w600k_r50.mnn",
        model_dir + "/2d106det.mnn", model_dir + "/1k3d68.mnn",
        RuntimeOption(), model_dir + "/genderage.mnn");
    REQUIRE(analysis.is_initialized());
    auto img = ImageData::imread(img_path);

    std::vector<face::InsightFaceResult> results;
    REQUIRE(analysis.analyze(img, &results, true, true, true, true));
    REQUIRE(!results.empty());
    // 关键点/embedding 应被填充
    bool has_lmk = false, has_emb = false, has_ga = false;
    for (const auto& r : results) {
        if (!r.landmark_2d_106.empty()) has_lmk = true;
        if (!r.embedding.empty()) has_emb = true;
        if (r.gender >= 0 && r.age >= 0) has_ga = true;
    }
    REQUIRE(has_lmk);
    REQUIRE(has_emb);
    REQUIRE(has_ga);
}
#endif // ENABLE_MNN

// ==================== TRT 后端 ====================
#ifdef ENABLE_TRT
TEST_CASE("InsightFace det_10g on TRT backend", "[insightface][model][backend:trt][gpu]") {
    const std::string model_dir = test_data_dir() + "/test_data/test_models/trt/insightface/buffalo_l";
    const std::string img_path = test_data_dir() + "/test_data/test_images/test_person.jpg";
    if (!std::filesystem::exists(model_dir + "/det_10g.engine")) return;
    if (!std::filesystem::exists(img_path)) return;

    // .engine 路径自动选 TRT 后端，需指定 GPU
    RuntimeOption opt;
    opt.use_gpu(0);
    opt.use_trt_backend();
    auto det = std::make_unique<face::InsightFaceDet>(model_dir + "/det_10g.engine", opt);
    REQUIRE(det->is_initialized());
    auto img = ImageData::imread(img_path);

    std::vector<face::InsightFaceBox> boxes;
    REQUIRE(det->predict(img, &boxes));
    REQUIRE(!boxes.empty());
}

TEST_CASE("InsightFace genderage on TRT backend", "[insightface][model][backend:trt][gpu]") {
    const std::string model_dir = test_data_dir() + "/test_data/test_models/trt/insightface/buffalo_l";
    const std::string img_path = test_data_dir() + "/test_data/test_images/test_person.jpg";
    if (!std::filesystem::exists(model_dir + "/genderage.engine")) return;
    if (!std::filesystem::exists(model_dir + "/det_10g.engine")) return;
    if (!std::filesystem::exists(img_path)) return;

    RuntimeOption opt;
    opt.use_gpu(0);
    opt.use_trt_backend();
    face::InsightFaceAnalysis analysis(
        model_dir + "/det_10g.engine", "", "", "",
        opt, model_dir + "/genderage.engine");
    REQUIRE(analysis.is_initialized());
    auto img = ImageData::imread(img_path);

    std::vector<face::InsightFaceResult> results;
    REQUIRE(analysis.analyze(img, &results, false, false, false, true));
    REQUIRE(!results.empty());
    bool has_ga = false;
    for (const auto& r : results) {
        if (r.gender >= 0 && r.age >= 0) has_ga = true;
    }
    REQUIRE(has_ga);
}
#endif // ENABLE_TRT

// ==================== SOPHGO 后端（Linux + Sophon-Sail，需 ENABLE_SOPHGO 编译） ====================
// bmodel 命名: <name>_f16.bmodel / <name>_int8.bmodel（见 tools/docker/sophgo/convert_all.sh）
#ifdef ENABLE_SOPHGO
TEST_CASE("InsightFace det_10g on SOPHGO backend", "[insightface][model][backend:sophgo]") {
    const std::string model_dir = test_data_dir() + "/test_data/test_models/sophgo/insightface/buffalo_l";
    const std::string img_path = test_data_dir() + "/test_data/test_images/test_person.jpg";
    if (!std::filesystem::exists(model_dir + "/det_10g_f16.bmodel") &&
        !std::filesystem::exists(model_dir + "/det_10g.bmodel")) return;
    if (!std::filesystem::exists(img_path)) return;
    std::string bmodel = std::filesystem::exists(model_dir + "/det_10g_f16.bmodel")
        ? model_dir + "/det_10g_f16.bmodel" : model_dir + "/det_10g.bmodel";

    RuntimeOption opt;
    opt.use_sophgo_backend(0);
    auto det = std::make_unique<face::InsightFaceDet>(bmodel, opt);
    REQUIRE(det->is_initialized());
    auto img = ImageData::imread(img_path);

    std::vector<face::InsightFaceBox> boxes;
    REQUIRE(det->predict(img, &boxes));
    REQUIRE(!boxes.empty());
}

TEST_CASE("InsightFace full pipeline on SOPHGO backend", "[insightface][model][backend:sophgo]") {
    const std::string model_dir = test_data_dir() + "/test_data/test_models/sophgo/insightface/buffalo_l";
    const std::string img_path = test_data_dir() + "/test_data/test_images/test_person.jpg";
    std::string ext = std::filesystem::exists(model_dir + "/det_10g_f16.bmodel") ? "_f16" : "";
    if (!std::filesystem::exists(model_dir + "/det_10g" + ext + ".bmodel")) return;
    if (!std::filesystem::exists(img_path)) return;

    RuntimeOption opt;
    opt.use_sophgo_backend(0);
    face::InsightFaceAnalysis analysis(
        model_dir + "/det_10g" + ext + ".bmodel", model_dir + "/w600k_r50" + ext + ".bmodel",
        model_dir + "/2d106det" + ext + ".bmodel", model_dir + "/1k3d68" + ext + ".bmodel",
        opt, model_dir + "/genderage" + ext + ".bmodel");
    REQUIRE(analysis.is_initialized());
    auto img = ImageData::imread(img_path);

    std::vector<face::InsightFaceResult> results;
    REQUIRE(analysis.analyze(img, &results, true, true, true, true));
    REQUIRE(!results.empty());
    bool has_lmk = false, has_emb = false, has_ga = false;
    for (const auto& r : results) {
        if (!r.landmark_2d_106.empty()) has_lmk = true;
        if (!r.embedding.empty()) has_emb = true;
        if (r.gender >= 0 && r.age >= 0) has_ga = true;
    }
    REQUIRE(has_lmk);
    REQUIRE(has_emb);
    REQUIRE(has_ga);
}
#endif // ENABLE_SOPHGO

// ==================== C API 绑定（含 genderage） ====================
TEST_CASE("InsightFace C API analyze returns gender/age", "[insightface][capi][model]") {
    const std::string model_dir = test_data_dir() + "/test_data/test_models/onnx/insightface/buffalo_l";
    const std::string img_path = test_data_dir() + "/test_data/test_images/test_person.jpg";
    if (!std::filesystem::exists(model_dir + "/genderage.onnx")) return;
    if (!std::filesystem::exists(img_path)) return;

    MDModel model{};
    model.model_name = nullptr;
    model.model_content = nullptr;
    MDRuntimeOption c_option{};
    c_option.device = MD_DEVICE_CPU;
    const auto status = md_create_insightface_model(
        &model, (model_dir + "/det_10g.onnx").c_str(),
        (model_dir + "/w600k_r50.onnx").c_str(),
        (model_dir + "/2d106det.onnx").c_str(),
        (model_dir + "/1k3d68.onnx").c_str(),
        (model_dir + "/genderage.onnx").c_str(),
        &c_option);
    REQUIRE(status == MDStatusCode::Success);
    REQUIRE(model.model_content != nullptr);

    auto img_mat = cv::imread(img_path);
    REQUIRE(!img_mat.empty());
    REQUIRE(img_mat.isContinuous());
    std::vector<uint8_t> bgr(img_mat.data, img_mat.data + img_mat.total() * 3);
    MDImage c_image = md_from_bgr24_data(bgr.data(), img_mat.cols, img_mat.rows);
    REQUIRE(c_image.data != nullptr);

    MDInsightFaceResults c_results{};
    REQUIRE(md_insightface_analyze(&model, &c_image, &c_results) == MDStatusCode::Success);
    REQUIRE(c_results.size > 0);
    bool has_ga = false;
    for (int i = 0; i < c_results.size; ++i) {
        if (c_results.data[i].gender >= 0 && c_results.data[i].age >= 0) has_ga = true;
        REQUIRE(c_results.data[i].embedding_size == 512);
    }
    REQUIRE(has_ga);

    md_free_insightface_result(&c_results);
    md_free_image(&c_image);
    md_free_insightface_model(&model);
}
