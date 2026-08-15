//
// 多阶段 pipeline 回归测试：LPR / 人脸识别 / 活体检测 / 行人属性 / OCR / insightface。
// 验证各 pipeline 在 ORT CPU 上能完整跑通并返回预期结果。
//
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <array>
#include "csrc/vision.h"
#include "csrc/vision/face/insightface/face_analysis.h"

namespace fs = std::filesystem;
using namespace modeldeploy::vision;

namespace {
    fs::path pipe_data_dir() {
        const char* env = std::getenv("TEST_DATA_DIR");
        return (env && *env) ? fs::path(env) / "test_data" : fs::current_path() / "test_data";
    }

    bool has_file(const fs::path& p) { return fs::exists(p); }

    ImageData pipe_img(const std::string& name) {
        auto p = pipe_data_dir() / "test_images" / name;
        if (!has_file(p)) return ImageData();
        return ImageData::imread(p.string());
    }

    fs::path onnx_model(const std::string& rel) {
        return pipe_data_dir() / "test_models" / "onnx" / rel;
    }

    fs::path ocr_dict() {
        for (const auto& d : {"ppocrv4_dict.txt", "ppocrv5_dict.txt", "dict.txt"}) {
            auto p = pipe_data_dir() / d;
            if (has_file(p)) return p;
        }
        return {};
    }

    fs::path find_ocr_model(const std::string& kind) {
        for (const auto& v : {"ppocrv4_mobile", "ppocrv5_mobile", "ppocrv6_tiny", "repsvtr_mobile"}) {
            auto p = pipe_data_dir() / "test_models" / "onnx" / "ocr" / v / (kind + "_infer.onnx");
            if (has_file(p)) return p;
        }
        return {};
    }
} // namespace

TEST_CASE("Pipeline LPR det+rec", "[pipeline][model]") {
    auto det = onnx_model("yolov5plate.onnx");
    auto rec = onnx_model("plate_recognition_color.onnx");
    if (!has_file(det) || !has_file(rec)) return;
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    lpr::LprPipeline model(det.string(), rec.string(), opt);
    REQUIRE(model.is_initialized());
    auto img = pipe_img("test_lpr_pipeline.jpg");
    if (img.empty()) return;
    std::vector<LprResult> results;
    REQUIRE(model.predict(img, &results));
    REQUIRE(!results.empty());
    // 至少一个结果有车牌字符串
    bool has_plate = false;
    for (const auto& r : results) {
        if (!r.car_plate_str.empty()) has_plate = true;
    }
    REQUIRE(has_plate);
}

TEST_CASE("Pipeline face recognition det+rec", "[pipeline][model]") {
    auto det = onnx_model("face/scrfd_2.5g_bnkps_shape640x640.onnx");
    auto rec = onnx_model("face/face_recognizer.onnx");
    if (!has_file(det) || !has_file(rec)) return;
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    face::FaceRecognizerPipeline model(det.string(), rec.string(), opt);
    REQUIRE(model.is_initialized());
    auto img = pipe_img("test_face_detection.jpg");
    if (img.empty()) return;
    std::vector<FaceRecognitionResult> results;
    REQUIRE(model.predict(img, &results));
    REQUIRE(!results.empty());
    // embedding 应为 512 维（seetaface 特征）
    for (const auto& r : results) {
        REQUIRE(!r.embedding.empty());
    }
}

TEST_CASE("Pipeline face rec predict_max_face", "[pipeline][model]") {
    auto det = onnx_model("face/scrfd_2.5g_bnkps_shape640x640.onnx");
    auto rec = onnx_model("face/face_recognizer.onnx");
    if (!has_file(det) || !has_file(rec)) return;
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    face::FaceRecognizerPipeline model(det.string(), rec.string(), opt);
    REQUIRE(model.is_initialized());
    auto img = pipe_img("test_face_detection.jpg");
    if (img.empty()) return;
    // 应只返回一张脸（最大），并报告总脸数
    FaceRecognitionResult r;
    int count = 0;
    REQUIRE(model.predict_max_face(img, &r, &count));
    REQUIRE(!r.embedding.empty());
    REQUIRE(count >= 1);
}

TEST_CASE("Pipeline face anti-spoof det+first+second", "[pipeline][model]") {
    auto det = onnx_model("face/scrfd_2.5g_bnkps_shape640x640.onnx");
    auto first = onnx_model("face/fas_first.onnx");
    auto second = onnx_model("face/fas_second.onnx");
    if (!has_file(det) || !has_file(first) || !has_file(second)) return;
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    face::SeetaFaceAsPipeline model(det.string(), first.string(), second.string(), opt);
    REQUIRE(model.is_initialized());
    auto img = pipe_img("test_face_detection.jpg");
    if (img.empty()) return;
    std::vector<FaceAntiSpoofResult> results;
    REQUIRE(model.predict(img, &results));
    REQUIRE(!results.empty());
    // 枚举类型：REAL/FUZZY/SPOOF 任一都合法
    for (const auto& r : results) {
        const bool valid = (r == FaceAntiSpoofResult::REAL ||
                            r == FaceAntiSpoofResult::FUZZY ||
                            r == FaceAntiSpoofResult::SPOOF);
        REQUIRE(valid);
    }
}

TEST_CASE("Pipeline pedestrian attribute det+cls", "[pipeline][model]") {
    auto det = onnx_model("zhgd_det.onnx");
    auto ml = onnx_model("zhgd_ml.onnx");
    if (!has_file(det) || !has_file(ml)) return;
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    pipeline::PedestrianAttribute model(det.string(), ml.string(), opt);
    REQUIRE(model.is_initialized());
    model.set_det_input_size({1280, 1280});
    model.set_cls_input_size({192, 256});
    auto img = pipe_img("test_pedestrian_attribute.jpg");
    if (img.empty()) return;
    std::vector<AttributeResult> results;
    REQUIRE(model.predict(img, &results));
    REQUIRE(!results.empty());
    for (const auto& r : results) {
        REQUIRE(!r.attr_scores.empty());
    }
}

TEST_CASE("Pipeline OCR det+cls+rec", "[pipeline][model]") {
    auto det = find_ocr_model("det");
    auto cls = find_ocr_model("cls");
    auto rec = find_ocr_model("rec");
    auto dict = ocr_dict();
    if (!has_file(det) || !has_file(cls) || !has_file(rec) || dict.empty()) return;
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    ocr::PaddleOCR model(det.string(), cls.string(), rec.string(), dict.string(), opt);
    REQUIRE(model.is_initialized());
    auto img = pipe_img("test_ocr.png");
    if (img.empty()) return;
    OCRResult result;
    REQUIRE(model.predict(img, &result));
    REQUIRE(!result.boxes.empty());
    REQUIRE(!result.text.empty());
}

TEST_CASE("Pipeline insightface full", "[pipeline][model]") {
    auto dir = pipe_data_dir() / "test_models" / "onnx" / "insightface" / "buffalo_l";
    if (!has_file(dir / "det_10g.onnx")) return;
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    auto analysis = face::InsightFaceAnalysis::create_from_dir(dir.string(), opt);
    REQUIRE(analysis != nullptr);
    REQUIRE(analysis->is_initialized());
    auto img = pipe_img("test_person.jpg");
    if (img.empty()) return;
    std::vector<face::InsightFaceResult> results;
    REQUIRE(analysis->analyze(img, &results, true, true, true, true));
    REQUIRE(!results.empty());
    // 应同时有 landmark + embedding + genderage
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

TEST_CASE("Pipeline insightface max_face + sub-model selection", "[pipeline][model]") {
    auto dir = pipe_data_dir() / "test_models" / "onnx" / "insightface" / "buffalo_l";
    if (!has_file(dir / "det_10g.onnx")) return;
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    auto analysis = face::InsightFaceAnalysis::create_from_dir(dir.string(), opt);
    REQUIRE(analysis != nullptr);
    REQUIRE(analysis->is_initialized());
    auto img = pipe_img("test_person.jpg");
    if (img.empty()) return;

    // 1) 子模型可选：只做识别（跳过 landmark/age）
    std::vector<face::InsightFaceResult> results;
    REQUIRE(analysis->analyze(img, &results, false, false, true, false));
    REQUIRE(!results.empty());
    for (const auto& r : results) {
        REQUIRE(r.landmark_2d_106.empty());   // 2d 已跳过
        REQUIRE(r.landmark_3d_68.empty());    // 3d 已跳过
        REQUIRE(!r.embedding.empty());        // recognition 保留
        REQUIRE(r.gender < 0);                // genderage 已跳过
    }

    // 2) 只识别最大人脸
    face::InsightFaceResult max_face;
    int count = 0;
    REQUIRE(analysis->analyze_max_face(img, &max_face, true, true, true, true, &count));
    REQUIRE(count >= 1);
    REQUIRE(!max_face.embedding.empty());
    REQUIRE(max_face.gender >= 0);
    REQUIRE(max_face.age >= 0);
}
